"""Rumor candidate filtering and event clustering (plan §6.2 stage 1, §6.3).

Two steps, both cheap and deterministic, so they can be re-run at any time:

  1. Keyword pre-filter — a post is a *rumor candidate* if its title or body
     mentions any `rumor_keywords` term. High recall, low precision by design;
     the LLM triage in §6.2 stage 2 does the expensive judging afterwards, and
     it only has to look at what survives here.
  2. Clustering — candidates for the same ticker belong to the same event while
     no more than `event.gap_hours` of silence separates consecutive posts. An
     event is kept when it has >= `event.min_posts` posts, or a single post that
     already cleared `event.min_solo_score`.

A post mentioning two tickers becomes a candidate for each: events are
per-ticker, because labeling and market features are per-ticker.

`claim_type` is not known until LLM triage runs, so this pass clusters on
ticker alone. `cluster()` takes the grouping key as an argument so §6.2 can
re-run it as (ticker, claim_type) later without changing this logic.

Usage:
  python -m src.pipeline.events [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from src import db
from src.pipeline.tickers import TickerExtractor
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, ts_to_iso

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    """One (post, ticker) pair that passed the keyword pre-filter."""
    post_id: str
    ticker: str
    created_utc: int
    score: int
    subreddit: str


@dataclass
class Event:
    ticker: str
    t0_utc: int
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def event_id(self) -> str:
        """Deterministic, so re-running clustering upserts rather than duplicates."""
        return f"{self.ticker}-{self.t0_utc}"

    @property
    def max_score(self) -> int:
        return max(c.score or 0 for c in self.candidates)

    def to_row(self) -> dict:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "t0_utc": self.t0_utc,
            "claim_summary": None,   # filled by LLM triage (§6.2 stage 2)
            "claim_type": None,
            "post_ids": json.dumps([c.post_id for c in self.candidates]),
            "n_posts": len(self.candidates),
            "subreddits": json.dumps(
                sorted({c.subreddit for c in self.candidates if c.subreddit})
            ),
        }


def keyword_regex(keywords: Iterable[str]) -> re.Pattern[str]:
    """Whole-word, case-insensitive alternation over the config keyword list.

    Multi-word terms ("chapter 11", "short squeeze") tolerate any whitespace
    run, so line breaks inside a post body still match. A trailing "s" is
    optional because the config lists singulars while posts say "mergers",
    "lawsuits", "investigations" — 18% more matching posts, all of them the
    same kinds of claim.
    """
    parts = [re.escape(k.strip()).replace(r"\ ", r"\s+") for k in keywords if k.strip()]
    return re.compile(r"\b(?:" + "|".join(parts) + r")s?\b", re.IGNORECASE)


def iter_candidates(
    conn, cfg: dict, pattern: re.Pattern[str], extractor=None
) -> Iterator[Candidate]:
    """Every (post, ticker) pair inside the backtest window that hits a keyword.

    A post that mentions several tickers seeds an event for each, which is how
    a post about $BBAI whose body name-drops PLTR ends up creating a bogus PLTR
    event. When `event.title_ticker_priority` is on (and an extractor is given),
    a post that names any ticker in its *title* only seeds events for those —
    the title is what the post is about. Posts that mention tickers only in the
    body (30% of links, typically long DD) are unaffected.
    """
    start_ts = date_str_to_ts(cfg["backtest_window"]["start"])
    end_ts = date_str_to_ts(cfg["backtest_window"]["end"])
    etfs = _etf_symbols(cfg) if cfg["event"]["exclude_etfs"] else set()
    title_priority = cfg["event"]["title_ticker_priority"] and extractor is not None

    rows = conn.execute(
        """
        SELECT p.id, p.created_utc, p.score, p.subreddit, p.title, p.selftext,
               GROUP_CONCAT(pt.ticker) AS tickers
        FROM posts p
        JOIN post_tickers pt ON pt.post_id = p.id
        WHERE p.created_utc >= ? AND p.created_utc < ?
        GROUP BY p.id
        ORDER BY p.created_utc
        """,
        (start_ts, end_ts),
    )
    for post_id, created, score, subreddit, title, selftext, tickers in rows:
        if not pattern.search(f"{title or ''} {selftext or ''}"):
            continue
        candidates = {t for t in (tickers or "").split(",") if t and t not in etfs}
        if title_priority:
            in_title = candidates & set(extractor.extract(title or ""))
            candidates = in_title or candidates
        for ticker in sorted(candidates):
            yield Candidate(post_id, ticker, created, score or 0, subreddit)


def _etf_symbols(cfg: dict) -> set[str]:
    """ETF tickers from the generated universe.

    An index fund has no company-specific claim for news to confirm or deny, so
    "$SPY will crash" is not a rumor in the §6.2 sense however often it is
    posted. Turn off with event.exclude_etfs if you want them back.
    """
    import csv

    with open(cfg["tickers"]["listed_csv"], newline="", encoding="utf-8") as fh:
        return {r["ticker"] for r in csv.DictReader(fh) if r.get("etf") == "1"}


def cluster(
    candidates: Iterable[Candidate],
    gap_hours: float,
    key: Callable[[Candidate], str] = lambda c: c.ticker,
) -> list[Event]:
    """Group candidates into events; a gap of >gap_hours starts a new one."""
    gap_s = gap_hours * 3600
    by_key: dict[str, list[Candidate]] = {}
    for cand in candidates:
        by_key.setdefault(key(cand), []).append(cand)

    events: list[Event] = []
    for group in by_key.values():
        group.sort(key=lambda c: c.created_utc)
        current: Event | None = None
        previous_ts = 0
        for cand in group:
            if current is None or cand.created_utc - previous_ts > gap_s:
                current = Event(ticker=cand.ticker, t0_utc=cand.created_utc)
                events.append(current)
            current.candidates.append(cand)
            previous_ts = cand.created_utc
    return sorted(events, key=lambda e: e.t0_utc)


def keep_event(event: Event, cfg: dict) -> bool:
    """§6.3 minimum size: >=min_posts posts, or one post that already went big."""
    ecfg = cfg["event"]
    if len(event.candidates) >= ecfg["min_posts"]:
        return True
    return event.max_score >= ecfg["min_solo_score"]


def build_events(conn, cfg: dict, extractor=None) -> tuple[list[Event], dict]:
    """Full pass: keyword pre-filter, cluster, apply the minimum-size rule."""
    pattern = keyword_regex(cfg["rumor_keywords"])
    if extractor is None and cfg["event"]["title_ticker_priority"]:
        extractor = TickerExtractor.from_config(cfg)
    candidates = list(iter_candidates(conn, cfg, pattern, extractor))
    clustered = cluster(candidates, cfg["event"]["gap_hours"])
    kept = [e for e in clustered if keep_event(e, cfg)]

    stats = {
        "candidates": len(candidates),
        "candidate_posts": len({c.post_id for c in candidates}),
        "clusters": len(clustered),
        "kept": len(kept),
        "dropped_too_small": len(clustered) - len(kept),
        "tickers": len({e.ticker for e in kept}),
        "posts_in_kept": sum(len(e.candidates) for e in kept),
    }
    return kept, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the event set without writing to SQLite")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    events, stats = build_events(conn, cfg)

    log.info("keyword pre-filter: %d (post, ticker) candidates from %d posts",
             stats["candidates"], stats["candidate_posts"])
    log.info("clustering (%sh gap): %d clusters -> %d events kept, %d below "
             "minimum size", cfg["event"]["gap_hours"], stats["clusters"],
             stats["kept"], stats["dropped_too_small"])
    # LLM triage (§6.2 stage 2) only has to look at posts inside kept events.
    log.info("posts inside kept events: %d (the LLM triage budget)",
             stats["posts_in_kept"])

    sizes = Counter(len(e.candidates) for e in events)
    log.info("event sizes: %s", ", ".join(
        f"{n} post{'s' if n > 1 else ''}: {sizes[n]}" for n in sorted(sizes)[:6]))
    log.info("%d distinct tickers; top: %s", stats["tickers"], ", ".join(
        f"{t}({n})" for t, n in
        Counter(e.ticker for e in events).most_common(8)))
    if events:
        log.info("span %s .. %s", ts_to_iso(events[0].t0_utc),
                 ts_to_iso(events[-1].t0_utc))

    if args.dry_run:
        log.info("DRY RUN — nothing written")
        return
    new = db.upsert_events(conn, [e.to_row() for e in events])
    stale = db.prune_events(conn, {e.event_id for e in events})
    total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    log.info("wrote %d events (%d new, %d stale removed); events table now has "
             "%d rows", len(events), new, stale, total)


if __name__ == "__main__":
    main()
