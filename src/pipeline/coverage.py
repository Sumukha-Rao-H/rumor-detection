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
    outcome: str          # ok | no_bars | no_sessions | starts_late | ends_early | gaps
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
    # `end` has no matching clamp. The front clamp is safe only because
    # `window_start` is a pure config value that is also, by construction,
    # the earliest date bars were ever collected for. There is no equivalent
    # config value for "latest date bars were collected through" — that
    # lives in the DB's `meta` table as `snapshot_frozen_{interval}` (see
    # src/collectors/market.py), not in `cfg`, and this function only gets
    # `cfg`. `study_window["end"]` is NOT that bound: pad_days_after already
    # reaches past it today, and the collector's snapshot freeze date is
    # later still (config.yaml: frozen 2026-08-30, well past study_window.end
    # of 2026-08-01), so clamping `end` at study_window.end would misreport
    # genuinely-collected late events as clamped/short. If study_window.end
    # is ever moved close to or past the snapshot freeze date, a late event's
    # tail could be reported `ends_early`/`gaps` with no `clamped`-style flag
    # to say "that's our scope, not a data defect" — same failure shape as
    # the front clamp guards against, just currently unreachable.
    end = acceptance_utc + mcfg["pad_days_after"] * DAY_S
    window_start = date_str_to_ts(cfg["study_window"]["start"])
    clamped = start < window_start
    return (max(start, window_start), end, clamped)


def session_dates(cfg: dict, start_ts: int, end_ts: int) -> set:
    """Real exchange session dates in a span — the calendar's ground truth.

    `audit()` intersects a ticker's bar dates against this set so that a bar
    stamped on a non-session date (a weekend, a market holiday, or an
    upstream timestamp/timezone bug that shifts a bar onto the wrong
    calendar day) can never substitute for a missing trading day. Before this
    intersection existed, `audit()` counted "distinct calendar dates with a
    bar" directly, so a ticker padded with weekend-dated bars could be
    reported fully covered while missing most of its real trading sessions.
    """
    cal = get_market_calendar(cfg["market"]["calendar"])
    a = pd.Timestamp(start_ts, unit="s", tz="UTC").normalize().tz_localize(None)
    b = pd.Timestamp(end_ts, unit="s", tz="UTC").normalize().tz_localize(None)
    clipped_a = max(a, cal.first_session)
    clipped_b = min(b, cal.last_session)
    if clipped_b < clipped_a:
        return set()
    if clipped_a != a or clipped_b != b:
        # The required span reaches outside the calendar's own known range.
        # `expected` (len of the result) silently shrinks when this fires —
        # same "denominator quietly gets easier" risk as an unclamped `end`
        # above, just bounded by the calendar library instead of the study
        # window. Not reachable with `exchange_calendars`' XNYS schedule
        # against today's config; logged so it is visible if that changes.
        log.debug("session_dates: span [%s, %s] exceeds calendar bounds, "
                  "clipped to [%s, %s] — expected sessions reduced",
                  a.date(), b.date(), clipped_a.date(), clipped_b.date())
    return {s.date() for s in cal.sessions_in_range(clipped_a, clipped_b)}


def session_count(cfg: dict, start_ts: int, end_ts: int) -> int:
    """Exchange sessions in a span, from the calendar rather than by division.

    Sessions, not bars: a 09:30-16:00 session yields seven bars (six hours plus
    the 15:30 half-hour stub) against 6.5 trading hours, so any bar-count ratio
    scores a perfectly covered event above 1.0 and the threshold ends up
    absorbing an artefact instead of measuring a gap. A half-day counts as one
    session, so early closes cannot fail the ratio either.
    """
    return len(session_dates(cfg, start_ts, end_ts))


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
        # Compared on whole DATES, not exact timestamps. The span boundaries
        # land mid-day, so a session at the edge would have its bars fall
        # outside the span while a naive count still counted the session —
        # costing up to two days at each end. `session_dates()` truncates to
        # dates the same way, so this stays aligned automatically.
        sess_dates = session_dates(cfg, start, end)
        expected = len(sess_dates)
        stamps = by_ticker.get(ticker, [])
        bar_dates = {pd.Timestamp(ts, unit="s", tz="UTC").date() for ts in stamps}
        # Intersected with the real calendar sessions, not just date-range
        # filtered: a bar dated on a weekend/holiday inside the span must
        # never count toward coverage, only a bar on an actual session date.
        span_dates = bar_dates & sess_dates

        if expected == 0:
            # The required span holds no exchange sessions at all (e.g. pads
            # and t0 lookback shrink to nothing and the span lands entirely on
            # a weekend/holiday, or the span falls outside the calendar's own
            # range). Distinct from `no_bars`: nothing was required here, so
            # nothing was measured against `min_session_coverage` — this must
            # never be reported as `ok` on the strength of a stray bar. Not
            # reachable with today's config (pad_days_before=30 alone
            # guarantees expected > 0).
            outcome, present = "no_sessions", 0
        elif not span_dates:
            outcome, present = "no_bars", 0
        else:
            present = len(span_dates)
            # Cause before symptom: a ticker whose hourly history begins inside
            # the span would otherwise be reported as a gap.
            if stamps[0] > start + tol:
                outcome = "starts_late"
            elif stamps[-1] < end - tol:
                outcome = "ends_early"
            elif present / expected < min_cov:
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
    for outcome in ("starts_late", "ends_early", "gaps", "no_bars", "no_sessions"):
        if counts[outcome]:
            print(f"  {outcome:<13}: {counts[outcome]:,}")
    print(f"\npad clamped at the window start for {clamped:,} early events "
          f"(our scope, not a data defect)")

    if failures:
        # Grouped by (ticker, outcome), not by ticker alone: one ticker can
        # fail two different events for two different reasons (e.g. a
        # no_bars filing and a separate starts_late filing), which would
        # otherwise be double-counted under a "failing tickers" label.
        worst = defaultdict(int)
        for v in failures:
            worst[(v.ticker, v.outcome)] += 1
        n_tickers = len({ticker for ticker, _ in worst})
        print(f"\nWorst offenders — {min(len(worst), max_listed)} of "
              f"{len(worst)} (ticker, outcome) pairs, {n_tickers:,} distinct "
              f"ticker{'s' if n_tickers != 1 else ''}:")
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
