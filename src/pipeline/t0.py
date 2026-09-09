"""The t0 correction — when did the market actually learn?

The SEC's acceptance time is not the moment news became public. Companies wire
a press release first and file the 8-K with that release attached minutes to
hours later; acceptance times cluster at 20:00-21:00 UTC while earnings
releases hit the wire at ~16:05 ET. Using acceptance alone silently counts time
the market already knew as "advance warning", which is the single easiest way
to report a result that is too good.

    t0 = min(8-K acceptanceDateTime, earliest credible article)

This module finds the second term. Storing all three columns is P4-02's job;
here we match, and measure how well the matching works.

Two limits are inherent and belong in the report rather than in a fix:

  * We match on ticker and time only. Nothing here confirms the article is
    ABOUT this filing — checking that would mean reading article text, which
    the project's framing excludes.
  * Finnhub's free tier carries no wire services, so a tier-1 t0 falls back to
    filing time almost everywhere. `--report` states that as a number.

Usage:
  python -m src.pipeline.t0 --report          # match rate, gap distribution
  python -m src.pipeline.t0 --report --tier 1 # wires only, the sensitivity check
  python -m src.pipeline.t0 --sample 20       # matched headlines, to eyeball

`--tier` is a REPORTING switch and is refused alongside `--build`: there is one
`events` table and one `t0_utc` column, so building at tier 1 would not produce
a second variant, it would overwrite the study with the uncorrected one.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter
from typing import NamedTuple

from src import db
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, ts_to_iso, utc_now_ts

log = logging.getLogger(__name__)

HOUR_S = 3600

#: Columns of `events` that this module does NOT own. Item filtering (P4-03)
#: and materiality (P4-04) write them with UPDATE, and `db.upsert_events`
#: writes EVERY column of every row it is handed — so a key missing from a row
#: built here lands as NULL and erases their work. `build_events` therefore
#: reads them back and carries them forward; see `_carry_forward`.
DOWNSTREAM_COLUMNS = ("is_scheduled", "abs_return", "is_material", "usable",
                      "exclude_reason")

#: The honest state of a genuinely new event: not yet proven usable, because
#: the filters that decide have not run. Explicit 0 rather than the column
#: default, because NULL would also hide the row from a `WHERE usable = 0`
#: query looking for rejected events.
NEW_EVENT_DEFAULTS = {"is_scheduled": None, "abs_return": None,
                      "is_material": None, "usable": 0, "exclude_reason": None}

#: Why an event kept filing time as its t0. Two very different causes that a
#: bare `t0_source = 'filing'` hides from each other — see `match_filing`.
NO_NEWS_IN_LOOKBACK = "no_news_in_lookback"
NO_NEWS_FOR_TICKER = "no_news_for_ticker"

#: Provenance stamped on every build, so a later reader of `events` can tell
#: which tier and lookback produced the stored t0 values.
META_BUILT_UTC = "events:t0_built_utc"
META_MAX_TIER = "events:t0_max_tier"
META_LOOKBACK_HOURS = "events:t0_lookback_hours"


class Match(NamedTuple):
    """One filing's t0 candidates."""
    ticker: str
    acceptance_utc: int
    news_utc: int | None      # earliest credible article in the lookback
    t0_utc: int               # min(acceptance, news)
    source: str               # 'news' | 'filing'
    reason: str | None = None  # why 'filing', when it is 'filing'

    @property
    def gap_hours(self) -> float:
        """How much earlier the news moment is than acceptance."""
        return (self.acceptance_utc - self.t0_utc) / HOUR_S


def lookback_window(cfg: dict, acceptance_utc: int) -> tuple[int, int]:
    """The span searched for the release: `t0_lookback_hours` before acceptance.

    Ends AT acceptance and never reaches past it. An article published later
    cannot lower t0, because t0 is a minimum — and reaching forward would be
    exactly the look-ahead the leakage tests exist to prevent.
    """
    return (acceptance_utc - cfg["news"]["t0_lookback_hours"] * HOUR_S,
            acceptance_utc)


def _ticker_has_any_news(conn, ticker: str) -> bool:
    """Does this ticker have ANY article stored, at any time, at any tier?"""
    return conn.execute(
        "SELECT 1 FROM news WHERE ticker = ? LIMIT 1", (ticker,)
    ).fetchone() is not None


def match_filing(cfg: dict, conn, ticker: str, acceptance_utc: int,
                 max_tier: int | None = 2) -> Match:
    """Find the earliest credible article before one filing.

    `max_tier=1` restricts to wires and top-tier outlets — the release itself;
    `max_tier=2` also admits fast republishers of wire copy. Running both and
    reporting the difference is the sensitivity analysis P1-13b promised, and
    on this data it is stark: no tier-1 article exists anywhere.

    Publication time only, never the aggregator's crawl time (P1-14): crawl
    time lags publication by an unknown amount and would push t0 later.

    A filing-time fallback carries a `reason`, because two very different
    situations produce it and the report must not conflate them:

      `no_news_in_lookback`  the ticker is covered, nothing landed in the
                             window — the intended, uninteresting baseline.
      `no_news_for_ticker`   nothing is stored for this ticker at all. That is
                             a COVERAGE HOLE, not a measurement, so it is
                             counted separately — it reads as a defect rather
                             than as a normal fallback.
                             (It USED to double as the symptom of a lost
                             article, back when `news` was keyed on url alone
                             and a story first stored under another ticker was
                             invisible here. The key is `(url, ticker)` now and
                             36,836 URLs in the database are genuinely shared
                             across tickers, so that failure mode is gone and
                             this reason means only what it says.)
    """
    lo, hi = lookback_window(cfg, acceptance_utc)
    news_utc = db.earliest_news_ts(conn, ticker, lo, hi, max_tier=max_tier)
    if news_utc is None:
        reason = (NO_NEWS_IN_LOOKBACK if _ticker_has_any_news(conn, ticker)
                  else NO_NEWS_FOR_TICKER)
        return Match(ticker, acceptance_utc, None, acceptance_utc, "filing",
                     reason)
    return Match(ticker, acceptance_utc, news_utc,
                 min(acceptance_utc, news_utc), "news")


def match_all(cfg: dict, conn, max_tier: int | None = 2,
              universe_only: bool = True) -> list[Match]:
    """Match every in-window filing. Read-only; P4-02 stores the result."""
    lo = date_str_to_ts(cfg["study_window"]["start"])
    hi = date_str_to_ts(cfg["study_window"]["end"])
    universe = set(db.universe_tickers(conn)) if universe_only else None

    filings = db.filing_acceptance_times(conn, lo, hi, cfg["edgar"]["forms"])
    if universe is not None:
        in_universe = [(t, a) for t, a in filings if t in universe]
        log.info("t0: %d in-window filings, %d outside the liquidity universe "
                 "(skipped), %d to match",
                 len(filings), len(filings) - len(in_universe), len(in_universe))
        filings = in_universe
    if not filings:
        raise SystemExit(
            "no in-window filings to match — run the EDGAR collector and the "
            "liquidity filter first."
        )
    return [match_filing(cfg, conn, t, a, max_tier=max_tier) for t, a in filings]


def event_id_for(accession_no: str) -> str:
    """Accession number with punctuation stripped, per the `events` schema.

    Accession numbers are globally unique and immutable, so keying on them
    makes the event builder re-runnable: a config change updates rows in place
    instead of creating a second copy of the study.
    """
    return accession_no.replace("-", "").replace(".", "").strip()


def _carry_forward(conn) -> dict[str, dict]:
    """Current values of the columns this module does not own, by event_id."""
    cols = ", ".join(DOWNSTREAM_COLUMNS)
    return {
        r["event_id"]: {c: r[c] for c in DOWNSTREAM_COLUMNS}
        for r in conn.execute(f"SELECT event_id, {cols} FROM events")
    }


def build_events(cfg: dict, conn, max_tier: int | None = 2) -> list[dict]:
    """One row per in-window, in-universe filing, carrying all three clocks.

    All three are kept apart on purpose — it is a frozen decision. Collapsing
    them into one column would push a variant switch into every metric, and
    would throw away the acceptance-minus-news gap, which is itself a result.

    Unmatched filings are built too, with t0 falling back to acceptance. That
    is not a failure: it is the uncorrected baseline every prior paper uses,
    and both variants must cover the same population or they describe two
    different studies.

    `is_scheduled`, `abs_return`, `is_material`, `usable` and `exclude_reason`
    belong to P4-03 and P4-04, and are genuinely left alone: whatever those
    phases stored is read back and re-emitted unchanged, because
    `db.upsert_events` writes every column and would otherwise NULL them out
    on every rebuild. Only a genuinely new event gets `usable = 0`.
    """
    lo = date_str_to_ts(cfg["study_window"]["start"])
    hi = date_str_to_ts(cfg["study_window"]["end"])
    universe = set(db.universe_tickers(conn))
    stored = _carry_forward(conn)

    rows, skipped = [], 0
    # A Counter, not a dict literal with two fixed keys: the old form only
    # incremented reasons it already knew, so a third fallback reason added to
    # `match_filing` later would have been dropped from the breakdown with no
    # error, under-reporting the true fallback total. Anything unexpected is
    # counted here and surfaced below rather than silently discarded.
    reasons: Counter[str] = Counter()
    for f in db.filings_in_window(conn, lo, hi, cfg["edgar"]["forms"]):
        if f["ticker"] not in universe:
            skipped += 1
            continue
        m = match_filing(cfg, conn, f["ticker"], f["acceptance_utc"],
                         max_tier=max_tier)
        if m.reason:
            reasons[m.reason] += 1
        eid = event_id_for(f["accession_no"])
        rows.append({
            "event_id": eid,
            "accession_no": f["accession_no"],
            "ticker": f["ticker"],
            "items": f["items"],
            "t0_filing_utc": f["acceptance_utc"],
            "t0_news_utc": m.news_utc,
            "t0_utc": m.t0_utc,
            "t0_source": m.source,
            **stored.get(eid, NEW_EVENT_DEFAULTS),
        })
    # The universe filter drops roughly two thirds of in-window 8-Ks. That is
    # by design, but a drop nobody counts is indistinguishable from a bug.
    log.info("events: %d in-window filings, %d skipped (ticker not in the "
             "liquidity universe), %d built", len(rows) + skipped, skipped,
             len(rows))
    log.info("events: filing-time fallback — %d with no article in the "
             "%sh lookback, %d with NO news stored for the ticker at all "
             "(a coverage hole, not a measurement)",
             reasons[NO_NEWS_IN_LOOKBACK], cfg["news"]["t0_lookback_hours"],
             reasons[NO_NEWS_FOR_TICKER])
    unexpected = set(reasons) - {NO_NEWS_IN_LOOKBACK, NO_NEWS_FOR_TICKER}
    if unexpected:
        # The line above names its two reasons, so a new one added to
        # `match_filing` would otherwise vanish from the breakdown entirely.
        log.warning("events: %d fallback(s) carried a reason this report does "
                    "not break out: %s — add it to the line above",
                    sum(reasons[r] for r in unexpected), sorted(unexpected))
    if not rows:
        raise SystemExit(
            "no events could be built — check that the EDGAR collection and "
            "the liquidity filter have both run."
        )
    return rows


def prune_events(conn, keep_ids: set[str]) -> int:
    """Drop stored events that the current window/universe no longer contains.

    Without this a rebuild is additive: `study_window.start` moved a year
    forward on 2026-08-29, and every event built before that would otherwise
    sit in `events` forever with its old `usable` flag — inflating the
    denominator of every match-rate number read back from the table, and
    handing `features.py` events with no price coverage.
    """
    stale = [(r[0],) for r in conn.execute("SELECT event_id FROM events")
             if r[0] not in keep_ids]
    if stale:
        conn.executemany("DELETE FROM events WHERE event_id = ?", stale)
        conn.commit()
    return len(stale)


def write_events(cfg: dict, conn, max_tier: int | None = 2) -> tuple[int, int]:
    """Build and store. Returns (rows written, newly created).

    Safe to re-run after a config change, in both directions: `upsert_events`
    corrects existing rows rather than duplicating them, `build_events` carries
    the downstream filter columns through untouched, and `prune_events` removes
    events the new config no longer covers. Not hypothetical — the lookback
    moved 24h -> 3h and the study window moved a year forward, so events built
    before either change carry a stale t0 or no longer belong at all.
    """
    rows = build_events(cfg, conn, max_tier=max_tier)
    matched = sum(1 for r in rows if r["t0_source"] == "news")
    new = db.upsert_events(conn, rows)
    pruned = prune_events(conn, {r["event_id"] for r in rows})
    now = utc_now_ts()
    db.set_meta(conn, META_BUILT_UTC, str(now), now)
    db.set_meta(conn, META_MAX_TIER,
                "any" if max_tier is None else str(max_tier), now)
    db.set_meta(conn, META_LOOKBACK_HOURS,
                str(cfg["news"]["t0_lookback_hours"]), now)
    log.info("events: %d rows written (%d new); %d matched to news",
             len(rows), new, matched)
    if pruned:
        # A deletion nobody announces is how a study quietly changes size.
        log.warning("events: %d stored events dropped — the current study "
                    "window/universe no longer covers them.", pruned)
    if not matched:
        # Rule 6: a run that produces zero records must say so. The whole point
        # of this module is the news correction; storing a study with none of
        # it is a decision, not a detail.
        log.warning("events: NOT ONE event took its t0 from news at tier %s. "
                    "The stored study is now the UNCORRECTED baseline "
                    "(t0 = acceptance everywhere). If that was not intended, "
                    "check the news table and the tier ceiling before using "
                    "these rows.", "any" if max_tier is None else max_tier)
    return len(rows), new


def build_provenance(conn) -> dict[str, str | None]:
    """Which tier and lookback produced the rows currently in `events`."""
    return {"built_utc": db.get_meta(conn, META_BUILT_UTC),
            "max_tier": db.get_meta(conn, META_MAX_TIER),
            "lookback_hours": db.get_meta(conn, META_LOOKBACK_HOURS)}


def stored_gap_report(cfg: dict, conn) -> None:
    """The Done-when: median and tail of acceptance minus news, from `events`.

    Read back from the table rather than recomputed, so the number reported is
    the number stored — together with the tier and lookback that produced it,
    because "0 matched" at tier 1 and "0 matched" from a broken news table look
    identical without the provenance.
    """
    rows = conn.execute(
        "SELECT t0_filing_utc, t0_news_utc, t0_source FROM events").fetchall()
    if not rows:
        print("\nNo events stored — run `python -m src.pipeline.t0 --build` "
              "first.")
        return
    gaps = sorted((r["t0_filing_utc"] - r["t0_news_utc"]) / HOUR_S
                  for r in rows if r["t0_source"] == "news")
    matched = len(gaps)

    prov = build_provenance(conn)
    print("\n=== t0 variants, as stored in `events` ===")
    if prov["built_utc"]:
        print(f"built               : {ts_to_iso(int(prov['built_utc']))} "
              f"at tier {prov['max_tier']}, "
              f"{prov['lookback_hours']}h lookback")
    else:
        print("built               : unknown (rows predate provenance "
              "stamping — rebuild to record it)")
    print(f"events              : {len(rows):,}")
    print(f"t0 from news        : {matched:,}  ({matched/len(rows):.1%})")
    print(f"t0 from filing time : {len(rows)-matched:,}")
    if not matched:
        print("\nNo event took its t0 from news at this tier.")
        return

    def q(p: float) -> float:
        return _percentile(gaps, p)
    print(f"\nacceptance minus news, for the {matched:,} corrected events:")
    for label, v in (("min", gaps[0]), ("p25", q(.25)), ("median", q(.50)),
                     ("p75", q(.75)), ("p90", q(.90)), ("p99", q(.99)),
                     ("max", gaps[-1])):
        print(f"  {label:<7}: {v*60:7.1f} min  ({v:5.2f} h)")
    print(f"  {'mean':<7}: {sum(gaps)/matched*60:7.1f} min  "
          f"({sum(gaps)/matched:5.2f} h)")
    print("\nThis gap is the correction: time the market already knew that "
          "acceptance-only\nt0 would have counted as advance warning.")


def _percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile. One copy, used by both gap reports."""
    n = len(sorted_values)
    return sorted_values[min(int(n * p), n - 1)]


def gap_buckets(cfg: dict) -> list[float]:
    """The report's histogram edges, in hours, derived from the lookback.

    Fixed 2h/6h/12h edges were left behind by the 24h -> 3h narrowing and went
    dead: a gap cannot exceed the lookback by construction, so "beyond 6h" was
    printing a tautological 0% that would stay 0% even if every match were a
    coincidence. Scaling with the window keeps the buckets measuring something.
    """
    look = cfg["news"]["t0_lookback_hours"]
    return [look * f for f in cfg["news"]["gap_bucket_fractions"]]


def gap_percentiles(matches: list[Match], cfg: dict) -> dict:
    """Where the matched articles actually sit, relative to acceptance.

    This is the diagnostic behind issue 27. A press release precedes its 8-K by
    minutes to a few hours; a median far into the window means the rule is
    matching unrelated daily coverage rather than the release.
    """
    gaps = sorted(m.gap_hours for m in matches if m.source == "news")
    if not gaps:
        return {}
    return {"p10": _percentile(gaps, .10), "p25": _percentile(gaps, .25),
            "p50": _percentile(gaps, .50), "p75": _percentile(gaps, .75),
            "p90": _percentile(gaps, .90), "mean": sum(gaps) / len(gaps),
            "within": {h: sum(g <= h for g in gaps) / len(gaps)
                       for h in gap_buckets(cfg)}}


def print_report(cfg: dict, conn, max_tier: int | None = 2) -> None:
    """Match rate and gap distribution — the evidence, not an assertion."""
    matches = match_all(cfg, conn, max_tier=max_tier)
    matched = [m for m in matches if m.source == "news"]
    no_window = sum(1 for m in matches if m.reason == NO_NEWS_IN_LOOKBACK)
    no_ticker = sum(1 for m in matches if m.reason == NO_NEWS_FOR_TICKER)
    look = cfg["news"]["t0_lookback_hours"]
    tier = "any" if max_tier is None else f"<= {max_tier}"

    print(f"\n=== t0 matching (tier {tier}, lookback {look}h) ===")
    print(f"filings          : {len(matches):,}")
    print(f"matched to news  : {len(matched):,}  ({len(matched)/len(matches):.1%})")
    print(f"fall back to filing time: {len(matches)-len(matched):,}")
    print(f"  no article in the {look}h lookback : {no_window:,}")
    print(f"  no news stored for the ticker     : {no_ticker:,}"
          f"  <- coverage hole, not a measurement")

    pct = gap_percentiles(matches, cfg)
    if not pct:
        print("\nNo article matched at this tier — t0 equals filing time for "
              "every event. That is the headline limitation, not a bug.")
        return
    print("\ngap, acceptance minus matched article:")
    for k in ("p10", "p25", "p50", "p75", "p90", "mean"):
        print(f"  {k:<5}: {pct[k]:6.1f} h")
    print(f"\nwhere in the {look}h window the matches sit "
          f"(every match is inside it by construction):")
    for h, frac in pct["within"].items():
        print(f"  within {h:5.2f}h of acceptance: {frac:.0%}")
    outer = max(pct["within"])
    if pct["p50"] > outer:
        print(f"\n⚠ The median match sits beyond {outer:.2f}h before "
              f"acceptance — the outer edge of\n  the {look}h window, later "
              f"than a press release plausibly runs. See issue 27:\n  widen "
              f"the window far enough and the earliest article is unrelated "
              f"daily\n  coverage rather than the release.")


def print_sample(cfg: dict, conn, n: int, max_tier: int | None = 2,
                 seed: int = 0) -> None:
    """Matched headlines next to their filing — how issue 27 was spotted.

    A distribution says something is wrong; the headlines say what. The lookup
    repeats the tier ceiling the match was made under and orders
    deterministically: without both, a same-second untiered blog post could be
    printed as "the release", which defeats the only tool for eyeballing match
    quality.
    """
    matches = [m for m in match_all(cfg, conn, max_tier=max_tier)
               if m.source == "news"]
    random.Random(seed).shuffle(matches)
    print(f"\n=== {min(n, len(matches))} matched filings ===")
    tier_clause = ("" if max_tier is None else
                   " AND source_tier IS NOT NULL AND source_tier <= ?")
    for m in matches[:n]:
        args = [m.ticker, m.news_utc]
        if max_tier is not None:
            args.append(max_tier)
        row = conn.execute(
            "SELECT title, source_name FROM news WHERE ticker = ? AND "
            f"published_utc = ?{tier_clause} "
            "ORDER BY source_tier IS NULL, source_tier, url LIMIT 1",
            args).fetchone()
        title = (row["title"] or "")[:78] if row else "(headline not found)"
        who = (row["source_name"] if row else None) or "?"
        print(f"\n  {m.ticker:<6} filed {ts_to_iso(m.acceptance_utc)}")
        print(f"  {'':<6} news  {ts_to_iso(m.news_utc)}  "
              f"({m.gap_hours:.1f}h earlier, {who})")
        print(f"  {'':<6} {title}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true",
                        help="match rate and gap distribution")
    parser.add_argument("--sample", type=int, metavar="N",
                        help="print N matched filings with their headlines")
    parser.add_argument("--build", action="store_true",
                        help="build/refresh `events` rows with all three t0 "
                             "columns. Safe to re-run after a config change: "
                             "rows are corrected in place, the P4-03/P4-04 "
                             "columns are preserved, and events the config no "
                             "longer covers are pruned.")
    parser.add_argument("--gap", action="store_true",
                        help="median and tail of acceptance minus news, read "
                             "back from the stored events")
    parser.add_argument("--tier", choices=("1", "2", "any"),
                        help="credibility tier ceiling for REPORTING "
                             "(default 2). 'any' includes untiered "
                             "publishers. Refused with --build: there is only "
                             "one t0_utc column, so a tier-1 build would "
                             "overwrite the study with the uncorrected t0 "
                             "rather than produce a second variant.")
    args = parser.parse_args()

    if args.build and args.tier is not None:
        raise SystemExit(
            "--tier is a reporting switch and cannot be combined with --build: "
            "it would overwrite `events` with the tier-restricted t0, not store "
            "a second variant. Use `--report --tier 1` for the sensitivity "
            "check, and `--build` alone to store the study."
        )

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    # The default comes from config, not from a literal repeated across five
    # signatures. `--tier` still overrides it, and `--tier any` still means no
    # ceiling; the function defaults below stay as they are so no test or
    # caller changes behaviour.
    tier = (None if args.tier == "any"
            else int(args.tier if args.tier else cfg["events"]["t0_max_tier"]))
    if args.build:
        write_events(cfg, conn, max_tier=tier)
        stored_gap_report(cfg, conn)
    elif args.gap:
        stored_gap_report(cfg, conn)
    elif args.sample is not None:
        print_sample(cfg, conn, args.sample, max_tier=tier)
    else:
        print_report(cfg, conn, max_tier=tier)


if __name__ == "__main__":
    main()
