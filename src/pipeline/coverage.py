"""Coverage audit — does every event have the price history it needs?

A model decides at hour t using the hours before it. An event whose stock has
no bars across its padded span cannot be scored at all: features come back NaN,
and NaNs that reach a feature matrix get imputed, dropped, or silently treated
as zero somewhere downstream. Far better to name those events here, with a
reason, while the reason is still recoverable.

The audit is read-only. Dropping events belongs to Phase 4, once `events`
exists; this module's product is the failure list.

Anchored on acceptance time rather than t0, because t0 does not exist until
P4-02 and Phase 3 has to certify the price data before Phase 4 builds labels on
it. Since t0 = min(acceptance, earliest article), t0 can only be EARLIER than
acceptance, and at most `news.t0_lookback_hours` earlier — so the required span
is widened by that lookback and the audit can never pass an event the real t0
would fail.

Usage:
  python -m src.pipeline.coverage            # audit and print the report
  python -m src.pipeline.coverage --failures # list every failing event
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from typing import NamedTuple

import pandas as pd

from src import db
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, get_market_calendar, ts_to_iso

log = logging.getLogger(__name__)

DAY_S = 86400


class Verdict(NamedTuple):
    """One event's coverage outcome.

    Carries the accession number so the result can be joined to `events` by
    key. (ticker, acceptance_utc) is not guaranteed unique — a company can file
    twice in the same second — and a silent mis-join would exclude the wrong
    event.
    """
    accession_no: str
    ticker: str
    acceptance_utc: int
    outcome: str          # ok | no_bars | starts_late | ends_early | gaps
    sessions_expected: int
    sessions_present: int
    clamped: bool         # the pad was cut short by the window start


def required_span(cfg: dict, acceptance_utc: int) -> tuple[int, int, bool]:
    """The bar span an event needs, and whether it was clamped.

    Widened at the front by `news.t0_lookback_hours`: t0 will land somewhere in
    that window, so auditing only from acceptance could pass an event that the
    real t0 later fails.

    Clamped at `study_window.start`, because a filing in the window's first days
    has a 30-day pad reaching into a period we deliberately never collected.
    Failing those events would be reporting our own scope decision as a data
    defect, so the clamp is applied and counted.
    """
    mcfg = cfg["market"]
    start = (acceptance_utc
             - mcfg["pad_days_before"] * DAY_S
             - cfg["news"]["t0_lookback_hours"] * 3600)
    end = acceptance_utc + mcfg["pad_days_after"] * DAY_S
    window_start = date_str_to_ts(cfg["study_window"]["start"])
    clamped = start < window_start
    return (max(start, window_start), end, clamped)


def session_count(cfg: dict, start_ts: int, end_ts: int) -> int:
    """Exchange sessions in a span, from the calendar rather than by division.

    Sessions, not bars: a 09:30-16:00 session yields seven bars (six hours plus
    the 15:30 half-hour stub) against 6.5 trading hours, so any bar-count ratio
    scores a perfectly covered event above 1.0 and the threshold ends up
    absorbing an artefact instead of measuring a gap. A half-day counts as one
    session, so early closes cannot fail the ratio either.
    """
    cal = get_market_calendar(cfg["market"]["calendar"])
    a = pd.Timestamp(start_ts, unit="s", tz="UTC").normalize().tz_localize(None)
    b = pd.Timestamp(end_ts, unit="s", tz="UTC").normalize().tz_localize(None)
    a = max(a, cal.first_session)
    b = min(b, cal.last_session)
    if b < a:
        return 0
    return len(cal.sessions_in_range(a, b))


def audit(cfg: dict, conn) -> list[Verdict]:
    """Check every in-window 8-K of an in-universe company. Read-only."""
    mcfg = cfg["market"]
    interval = mcfg["interval"]
    tol = mcfg["coverage_tolerance_days"] * DAY_S
    min_cov = mcfg["min_session_coverage"]
    lo = date_str_to_ts(cfg["study_window"]["start"])
    hi = date_str_to_ts(cfg["study_window"]["end"])

    universe = set(db.universe_tickers(conn))
    if not universe:
        raise SystemExit(
            "no company has in_universe = 1 — run "
            "`python -m src.pipeline.universe` first."
        )

    # Bar dates per ticker, once. Per-event queries would be ~18,000 round
    # trips over 2.6 million rows.
    by_ticker: dict[str, list[int]] = defaultdict(list)
    for ticker, ts in conn.execute(
            "SELECT ticker, ts_utc FROM bars WHERE interval = ? ORDER BY ts_utc",
            (interval,)):
        if ticker in universe:
            by_ticker[ticker].append(ts)

    events = [f for f in db.filings_in_window(conn, lo, hi, cfg["edgar"]["forms"])
              if f["ticker"] in universe]
    if not events:
        raise SystemExit(
            "no in-window filings for any in-universe company — nothing to "
            "audit. Check that the EDGAR collection and the liquidity filter "
            "have both run."
        )

    verdicts = []
    for f in events:
        ticker, acceptance = f["ticker"], f["acceptance_utc"]
        start, end, clamped = required_span(cfg, acceptance)
        expected = session_count(cfg, start, end)
        stamps = by_ticker.get(ticker, [])
        # Compared on whole DATES, not exact timestamps. The span boundaries
        # land mid-day, so a session at the edge would have its bars fall
        # outside the span while `session_count` still counted the session —
        # costing up to two days at each end. On a clamped span of five
        # sessions that alone drops the ratio to 0.8 and reports a healthy
        # large-cap as having gaps.
        lo_d = pd.Timestamp(start, unit="s", tz="UTC").date()
        hi_d = pd.Timestamp(end, unit="s", tz="UTC").date()
        dates = {pd.Timestamp(ts, unit="s", tz="UTC").date() for ts in stamps}
        span_dates = {d for d in dates if lo_d <= d <= hi_d}

        if not span_dates:
            outcome, present = "no_bars", 0
        else:
            present = len(span_dates)
            # Cause before symptom: a ticker whose hourly history begins inside
            # the span would otherwise be reported as a gap.
            if stamps[0] > start + tol:
                outcome = "starts_late"
            elif stamps[-1] < end - tol:
                outcome = "ends_early"
            elif expected and present / expected < min_cov:
                outcome = "gaps"
            else:
                outcome = "ok"
        verdicts.append(Verdict(f["accession_no"], ticker, acceptance, outcome,
                                expected, present, clamped))
    return verdicts


def print_report(cfg: dict, verdicts: list[Verdict], list_failures: bool = False,
                 max_listed: int = 25) -> None:
    """The acceptance check: the failure list, empty or explained."""
    counts: dict[str, int] = defaultdict(int)
    for v in verdicts:
        counts[v.outcome] += 1
    failures = [v for v in verdicts if v.outcome != "ok"]
    clamped = sum(1 for v in verdicts if v.clamped)

    print(f"\n=== Coverage audit [{cfg['market']['interval']}] ===")
    print(f"pad: -{cfg['market']['pad_days_before']}d / "
          f"+{cfg['market']['pad_days_after']}d, widened by "
          f"{cfg['news']['t0_lookback_hours']}h for t0; "
          f"min session coverage {cfg['market']['min_session_coverage']:.0%}")
    print(f"\nevents audited : {len(verdicts):,}")
    print(f"  ok           : {counts['ok']:,}  ({counts['ok']/len(verdicts):.1%})")
    for outcome in ("starts_late", "ends_early", "gaps", "no_bars"):
        if counts[outcome]:
            print(f"  {outcome:<13}: {counts[outcome]:,}")
    print(f"\npad clamped at the window start for {clamped:,} early events "
          f"(our scope, not a data defect)")

    if failures:
        worst = defaultdict(int)
        for v in failures:
            worst[(v.ticker, v.outcome)] += 1
        print(f"\nFailing tickers — worst {min(len(worst), max_listed)} "
              f"of {len(worst)}:")
        for (ticker, outcome), n in sorted(worst.items(), key=lambda kv: -kv[1])[:max_listed]:
            print(f"  {ticker:<8} {outcome:<13} {n} event(s)")
    if list_failures:
        print(f"\nAll {len(failures)} failing events:")
        for v in failures:
            print(f"  {v.ticker:<8} {ts_to_iso(v.acceptance_utc)} {v.outcome:<13} "
                  f"{v.sessions_present}/{v.sessions_expected} sessions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", action="store_true",
                        help="list every failing event, not just the summary")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    print_report(cfg, audit(cfg, conn), list_failures=args.failures)


if __name__ == "__main__":
    main()
