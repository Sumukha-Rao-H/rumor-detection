"""Machine label proposals — plan §6.4 stage 2, prompt from Appendix D.2.

Stage 1 (event_news.py) fetched the headlines published around each event. This
stage decides what they mean: it shows the model the claim and the credible
headlines from its window and asks whether any of them confirmed or denied it,
then applies §6.4's arithmetic rule to the events where nothing did.

    TRUE         a whitelist headline confirms the claim -> label 1, t_official
                 is that headline's timestamp.
    FALSE        a whitelist headline denies it (t_official = denial), OR
                 nothing confirms it *and* the stock never moved
                 (|3-day return| < threshold and volume z-score < threshold),
                 in which case t_official = t0 + horizon.
    UNVERIFIED   everything else, including every event whose market data is
                 too thin to apply the quiet rule. Excluded from train/eval.

Three decisions worth knowing about:

  proposals    nothing here writes events.label. Rows land in `label_proposals`
               and a human promotes them (stage 3). The plan is emphatic that
               every label gets reviewed — keeping the machine's opinion in a
               separate table means an unreviewed one cannot reach training by
               accident, rather than merely by discipline.
  quota        an event with no whitelist headline needs no model call: "no
               credible source published anything" is already the answer, and
               the quiet rule decides it. Gemini's free daily quota is the
               binding constraint on this project, so those events are settled
               offline and the calls are spent where there is something to read.
  timestamps   t_official comes from matching the model's quoted headline back
               to the news row it came from, not from the timestamp the model
               writes. t_official drives the reward's early-commit bonus, so it
               has to be a time we observed rather than one a model formatted.

A confirmation that lands *before* t0 is reported separately (rule
`confirmed_pre_t0`): the claim was already public when Reddit posted it, so the
event is a repost of known news rather than a rumor anyone could have been early
on. They stay TRUE but the count is worth watching — Phase 3 should not learn to
"predict" news that had already broken.

Usage:
  python -m src.pipeline.labeling --dry-run    # workload + how many need no call
  python -m src.pipeline.labeling --limit 50   # pilot
  python -m src.pipeline.labeling              # everything outstanding
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass

from src import db
from src.utils.config import load_config
from src.utils.llm import LLMClient, LLMError, LLMRefused
from src.utils.timeutils import ts_to_iso, utc_now_ts

log = logging.getLogger(__name__)

HOUR = 3600
DAY = 86400
VERDICTS = ("TRUE", "FALSE", "UNVERIFIED")
MAX_CONSECUTIVE_SKIPS = 20

PROMPT = """CLAIM (from Reddit, posted {t0}): {claim}
CANDIDATE NEWS HEADLINES (source, UTC time):
{headlines}

Did credible news CONFIRM or DENY the claim within {horizon} hours?
Respond ONLY with JSON:
{{"verdict": "TRUE|FALSE|UNVERIFIED", \
"t_official": "<UTC of earliest deciding headline or null>",
 "deciding_headline": "<text or null>", "confidence": 0.0-1.0}}"""


@dataclass
class Event:
    event_id: str
    ticker: str
    t0_utc: int
    claim_summary: str


@dataclass
class Headline:
    title: str
    domain: str
    seen_utc: int


def whitelisted(headlines: list, whitelist: list[str]) -> list[Headline]:
    """Keep only headlines from credible sources (config news.whitelist).

    Matched on the domain suffix so `feeds.reuters.com` counts as Reuters while
    `reuters.com.example.net` does not.
    """
    allowed = tuple(d.lower().lstrip(".") for d in whitelist)
    kept = []
    for row in headlines:
        domain = (row["source_domain"] or "").lower()
        if any(domain == a or domain.endswith("." + a) for a in allowed):
            kept.append(Headline(row["title"] or "", domain, row["seen_utc"]))
    return kept


def event_headlines(conn, cfg: dict, event: Event) -> tuple[list[Headline], int]:
    """(whitelist headlines in the event window, count before filtering)."""
    pre = int(cfg["news"]["event_pre_hours"]) * HOUR
    post = int(cfg["event"]["label_horizon_hours"]) * HOUR
    rows = db.news_in_window(conn, event.ticker, event.t0_utc - pre,
                             event.t0_utc + post)
    return whitelisted(rows, cfg["news"]["whitelist"]), len(rows)


def _daily_volume(bars: list) -> dict[int, float]:
    """Volume summed per UTC day, keyed by day index."""
    days: dict[int, float] = {}
    for bar in bars:
        days[bar["ts_utc"] // DAY] = days.get(bar["ts_utc"] // DAY, 0.0) + (
            bar["volume"] or 0.0)
    return days


def market_move(conn, cfg: dict, event: Event) -> tuple[float | None, float | None]:
    """(3-day return, volume z-score) around t0, or (None, None) if unmeasurable.

    Labeling is the one place future data is legitimate — the oracle is allowed
    to look at what happened after t0, which is exactly what the features in
    Phase 3 must never do.

    Returns None rather than guessing when the bars are too thin: a missing
    baseline would let a quiet-looking stock be labeled FALSE on no evidence.
    """
    lcfg = cfg["labeling"]
    baseline_days = int(lcfg["volume_baseline_days"])
    ret_days = int(lcfg["return_days"])
    bars = db.bars_in_range(conn, event.ticker,
                            event.t0_utc - (baseline_days + 2) * DAY,
                            event.t0_utc + (ret_days + 1) * DAY,
                            cfg["market"]["interval"])
    if not bars:
        return None, None

    before = [b for b in bars if b["ts_utc"] <= event.t0_utc]
    after = [b for b in bars if b["ts_utc"] <= event.t0_utc + ret_days * DAY]
    ret = None
    if before and after and len(after) > len(before) and before[-1]["close"]:
        ret = after[-1]["close"] / before[-1]["close"] - 1.0

    volumes = _daily_volume(bars)
    t0_day = event.t0_utc // DAY
    baseline = [v for day, v in volumes.items() if t0_day - baseline_days <= day < t0_day]
    window = [v for day, v in volumes.items() if t0_day <= day <= t0_day + ret_days]
    z = None
    if len(baseline) >= int(lcfg["min_baseline_days"]) and window:
        spread = statistics.pstdev(baseline)
        if spread > 0:
            z = (max(window) - statistics.fmean(baseline)) / spread
    return ret, z


def is_quiet(ret: float | None, z: float | None, cfg: dict) -> bool:
    """§6.4's 'no abnormal sustained price move' test. Unknown is not quiet."""
    lcfg = cfg["labeling"]
    if ret is None or z is None:
        return False
    return abs(ret) < float(lcfg["abnormal_return"]) and z < float(lcfg["volume_z"])


def format_headlines(headlines: list[Headline], cfg: dict) -> str:
    lcfg = cfg["labeling"]
    chars = int(lcfg["headline_chars"])
    shown = headlines[:int(lcfg["max_headlines"])]
    return "\n".join(f"- ({h.domain}, {ts_to_iso(h.seen_utc)}) {h.title[:chars]}"
                     for h in shown)


def build_prompt(event: Event, headlines: list[Headline], cfg: dict) -> str:
    return PROMPT.format(t0=ts_to_iso(event.t0_utc), claim=event.claim_summary,
                         headlines=format_headlines(headlines, cfg),
                         horizon=int(cfg["event"]["label_horizon_hours"]))


def content_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def match_headline(quoted: str, headlines: list[Headline]) -> Headline | None:
    """Find the news row the model quoted, so t_official is an observed time.

    Models paraphrase and truncate, so an exact match is tried first and a
    containment match second. No match means no t_official — better a missing
    timestamp than an invented one.
    """
    target = _normalize_text(quoted)
    if not target:
        return None
    exact = [h for h in headlines if _normalize_text(h.title) == target]
    loose = [h for h in headlines
             if target in _normalize_text(h.title)
             or _normalize_text(h.title) in target]
    # Wire services republish the same headline for hours; §6.4 wants the
    # earliest deciding one, not whichever copy came back first.
    candidates = exact or loose
    return min(candidates, key=lambda h: h.seen_utc) if candidates else None


def normalize(data: dict) -> dict:
    """Coerce a model reply into the fields we store."""
    if not isinstance(data, dict):
        raise LLMRefused(f"reply was {type(data).__name__}, not an object")
    verdict = str(data.get("verdict") or "").strip().upper()
    if verdict not in VERDICTS:
        raise LLMRefused(f"unusable verdict {data.get('verdict')!r}")
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        confidence = None
    headline = data.get("deciding_headline")
    return {"llm_verdict": verdict,
            "deciding_headline": str(headline).strip() if headline else None,
            "confidence": confidence}


def decide(event: Event, fields: dict, headlines: list[Headline],
           ret: float | None, z: float | None, cfg: dict) -> dict:
    """Turn a model verdict plus the market check into a §6.4 proposal."""
    horizon = int(cfg["event"]["label_horizon_hours"]) * HOUR
    verdict = fields["llm_verdict"]
    match = match_headline(fields.get("deciding_headline") or "", headlines)
    t_official = match.seen_utc if match else None

    if verdict == "TRUE" and match is None:
        # The model claims a confirmation it cannot point at. Without a real
        # headline there is no t_official, and a TRUE with no timestamp is
        # useless to the reward — hand it to the human instead.
        return {**fields, "verdict": "UNVERIFIED", "rule": "unmatched_headline",
                "t_official_utc": None}
    if verdict == "TRUE":
        rule = "confirmed_pre_t0" if t_official < event.t0_utc else "confirmed"
        return {**fields, "verdict": "TRUE", "rule": rule,
                "t_official_utc": t_official}
    if verdict == "FALSE":
        # A denial we cannot locate still means "no credible confirmation", so
        # the claim expires at the horizon rather than at an unknown moment.
        return {**fields, "verdict": "FALSE",
                "rule": "denied" if match else "denied_unmatched",
                "t_official_utc": t_official or event.t0_utc + horizon}
    if is_quiet(ret, z, cfg):
        return {**fields, "verdict": "FALSE", "rule": "quiet",
                "t_official_utc": event.t0_utc + horizon}
    return {**fields, "verdict": "UNVERIFIED",
            "rule": "no_bars" if ret is None or z is None else "moved",
            "t_official_utc": None}


def pending_events(conn, cfg: dict, prompt_version: str) -> list[Event]:
    """Rumor events with no proposal yet, oldest first."""
    done = db.proposed_event_ids(conn, prompt_version)
    rows = conn.execute(
        """SELECT event_id, ticker, t0_utc, claim_summary FROM events
           WHERE n_rumor_posts > 0 AND label IS NULL AND claim_summary IS NOT NULL
           ORDER BY t0_utc"""
    ).fetchall()
    return [Event(*row) for row in rows if row[0] not in done]


def propose(conn, cfg: dict, client: LLMClient | None, limit: int | None = None,
            dry_run: bool = False) -> dict:
    """Propose a label for every outstanding event. Resumable."""
    prompt_version = cfg["labeling"]["prompt_version"]
    events = pending_events(conn, cfg, prompt_version)
    if limit is not None:
        events = events[:limit]
    log.info("%d events to label under prompt %s", len(events), prompt_version)

    stats = Counter()
    consecutive_skips = 0
    for i, event in enumerate(events, 1):
        headlines, n_all = event_headlines(conn, cfg, event)
        ret, z = market_move(conn, cfg, event)
        stats["headlines"] += len(headlines)

        if not headlines:
            # Nothing credible was published: no question left for a model.
            stats["no_call"] += 1
            fields = {"llm_verdict": "UNVERIFIED", "deciding_headline": None,
                      "confidence": None}
            proposal = decide(event, fields, headlines, ret, z, cfg)
            provider = model = None
        else:
            if dry_run:
                stats["would_call"] += 1
                continue
            prompt = build_prompt(event, headlines, cfg)
            try:
                reply = client.complete_json(prompt, cache_key=content_key(prompt))
                fields = normalize(reply.data)
            except LLMRefused as exc:
                log.warning("skipping %s: %s", event.event_id, exc)
                stats["skipped"] += 1
                consecutive_skips += 1
                if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                    log.error("%d unusable replies in a row — stopping",
                              consecutive_skips)
                    stats["aborted"] = 1
                    break
                continue
            except LLMError as exc:
                log.error("giving up at event %d/%d: %s", i, len(events), exc)
                stats["aborted"] = 1
                break
            consecutive_skips = 0
            provider, model = reply.provider, reply.model
            proposal = decide(event, fields, headlines, ret, z, cfg)

        if dry_run:
            stats[f"verdict:{proposal['verdict']}"] += 1
            continue
        db.upsert_label_proposals(conn, [{
            **proposal, "event_id": event.event_id, "ret_3d": ret, "volume_z": z,
            "n_headlines": len(headlines), "n_headlines_all": n_all,
            "provider": provider, "model": model,
            "prompt_version": prompt_version, "created_utc": utc_now_ts(),
        }])
        stats["done"] += 1
        stats[f"verdict:{proposal['verdict']}"] += 1
        stats[f"rule:{proposal['rule']}"] += 1
        if client is not None and i % 25 == 0:
            log.info("%d/%d events (%d api calls, %d cached)", i, len(events),
                     client.calls, client.cache_hits)
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="propose at most N events")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the workload without calling the model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    client = None if args.dry_run else LLMClient(cfg)
    stats = propose(conn, cfg, client, limit=args.limit, dry_run=args.dry_run)

    verdicts = {k[8:]: v for k, v in stats.items() if k.startswith("verdict:")}
    rules = {k[5:]: v for k, v in stats.items() if k.startswith("rule:")}
    log.info("proposed %d (%d needed no call, %d skipped)", stats.get("done", 0),
             stats.get("no_call", 0), stats.get("skipped", 0))
    log.info("verdicts: %s", ", ".join(f"{k}={v}" for k, v in sorted(verdicts.items())))
    if rules:
        log.info("rules: %s", ", ".join(
            f"{k}={v}" for k, v in sorted(rules.items(), key=lambda x: -x[1])))
    if client is not None:
        log.info("%d api calls, %d cached, %d key rotations",
                 client.calls, client.cache_hits, client.key_rotations)


if __name__ == "__main__":
    main()
