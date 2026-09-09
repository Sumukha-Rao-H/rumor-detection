"""Market data collector — yfinance OHLCV into the `bars` table.

Backtest granularity is hourly ('60m') bars: yfinance serves ~730 days of
those, matching the hourly decision step. Daily bars are supplementary
context, and are what the Phase 3 liquidity filter ranks companies by.
1m/5m/15m/30m bars are NOT used (30-60 day history is useless for backtests).

The hourly window is ROLLING — bars available today silently disappear later —
so coverage must be downloaded broadly and early, then frozen. Broad coverage
(not just event windows) is required because negative sampling and trailing
z-scores both need continuous history. Budget hours of wall-clock time for a
full 6,000-ticker pull, and record the download date with --stamp-snapshot.

Fetches are incremental: each run resumes from the latest cached bar per
(ticker, interval), and every ticker's outcome is recorded in `fetch_state`
so `--resume` continues an interrupted run. All bar timestamps are stored as
UTC epoch seconds of the bar's open.

Usage:
  python -m src.collectors.market --tickers TSLA,AAPL --start 2025-09-01 --end 2026-08-01
  python -m src.collectors.market --candidates --interval 1d --resume
  python -m src.collectors.market --universe              # every liquid ticker + benchmark
  python -m src.collectors.market --universe --stamp-snapshot
  python -m src.collectors.market --candidates --interval 1d --report
"""

from __future__ import annotations

import argparse
import logging
from typing import NamedTuple

import pandas as pd
import yfinance as yf

from src import db
from src.utils.config import load_config
from src.utils.ratelimit import RateLimiter
from src.utils.timeutils import (
    date_str_to_ts, iso_utc_to_ts, ts_to_dt, ts_to_iso, utc_now_ts,
)

log = logging.getLogger(__name__)

DAY_S = 86400

#: Namespace prefix for this collector's rows in `fetch_state`. The interval is
#: appended, because a ticker finished for daily bars is NOT finished for
#: hourly ones — a shared namespace would make the hourly run skip all 6,000
#: tickers on its first --resume.
FETCH_SOURCE_PREFIX = "market"


def fetch_source(interval: str) -> str:
    """`fetch_state.source` for one bar interval, e.g. 'market:1d'."""
    return f"{FETCH_SOURCE_PREFIX}:{interval}"


class FetchResult(NamedTuple):
    """One ticker's outcome.

    `attempted` is False when the cache already covers the window and no
    request was issued — that is not an attempt, and must not arm the
    zero-record guard.
    """
    attempted: bool
    parsed: int
    written: int


def default_start_ts(cfg: dict, interval: str) -> int:
    """Start of the download window for one interval, when --start is absent.

    Daily bars reach back `universe.min_history_days` BEFORE the study window,
    because the liquidity filter applies its history requirement and averages
    traded value *as of the window start* — it can do neither from bars that
    begin on that same day. Derived from the knob the filter itself reads,
    rather than duplicated into a second one that could drift away from it.
    """
    start = date_str_to_ts(cfg["study_window"]["start"])
    if interval == cfg["market"]["daily_interval"]:
        return start - cfg["universe"]["min_history_days"] * DAY_S
    return start


def df_to_rows(df: pd.DataFrame, ticker: str, interval: str) -> list[tuple]:
    """History DataFrame -> bars rows. Index is converted to UTC epoch secs."""
    if df is None or df.empty:
        return []
    idx = df.index
    if getattr(idx, "tz", None) is None:
        # Verified against the installed yfinance 1.5.2
        # (`yfinance/utils.py::set_df_tz`) that both intervals this collector
        # issues ('1d' and '60m') always come back tz-aware, localized to the
        # exchange timezone, so this is not a path a healthy fetch reaches.
        # If a future yfinance or a different feed does hand back a naive
        # index, the value is EXCHANGE-LOCAL time, not UTC — calling it UTC
        # shifts every bar four or five hours, and `upsert_bars` sets OHLCV
        # unconditionally on conflict, so those wrong bars would overwrite the
        # frozen snapshot with nothing in the data to show it happened.
        # Refuse, the way `timeutils.iso_utc_to_ts` refuses to guess UTC
        # (AGENTS rule 3). `collect_many`'s per-ticker `except Exception`
        # turns this into a recorded `failed` state for one ticker rather
        # than a dead run.
        raise ValueError(
            f"{ticker} [{interval}]: yfinance returned a tz-naive index. "
            f"That value is exchange-local time, not UTC — refusing to guess, "
            f"because storing it as UTC shifts every bar by the exchange's "
            f"offset and silently restates the frozen snapshot."
        )
    idx = idx.tz_convert("UTC")
    rows = {}
    skipped_nan = 0
    dup = 0
    for ts, row in zip(idx, df.itertuples(index=False)):
        # A valid Close with a NaN Open/High/Low is a real, if rare, yfinance
        # shape (illiquid names, bars around halts). Storing it would put NULLs
        # into columns nothing downstream currently reads (features.py only
        # reads close/volume) — but the whole point of "every NaN explained" is
        # not to leave a NULL sitting in `bars` for some future feature to trip
        # over silently. Drop the whole bar rather than store a partial one.
        if pd.isna(row.Open) or pd.isna(row.High) or pd.isna(row.Low) or pd.isna(row.Close):
            skipped_nan += 1
            continue
        key = int(ts.timestamp())
        if key in rows:
            dup += 1
        rows[key] = (
            ticker,
            key,
            float(row.Open),
            float(row.High),
            float(row.Low),
            float(row.Close),
            float(row.Volume) if not pd.isna(row.Volume) else 0.0,
            interval,
        )
    if skipped_nan:
        log.warning("%s [%s]: skipped %d bar(s) with NaN in open/high/low/close",
                    ticker, interval, skipped_nan)
    if dup:
        log.warning("%s [%s]: %d duplicate bar timestamp(s) in one fetch — "
                    "kept the last occurrence of each", ticker, interval, dup)
    return list(rows.values())


def clamp_start(cfg: dict, start_ts: int, interval: str, now_ts: int) -> int:
    """Enforce yfinance's history window for intraday bars.

    Both the interval and the lookback come from config. The interval used to
    be the literal `"60m"`, which meant changing `market.interval` alone turned
    the clamp off without a word — the one edit most likely to need it.
    """
    if interval != cfg["market"]["interval"]:
        return start_ts
    floor = now_ts - cfg["market"]["hourly_max_lookback_days"] * DAY_S
    if start_ts < floor:
        log.warning("%s bars only go back %d days — clamping start %s -> %s",
                    interval, cfg["market"]["hourly_max_lookback_days"],
                    ts_to_iso(start_ts), ts_to_iso(floor))
        return floor
    return start_ts


def assert_not_frozen(cfg: dict, conn, interval: str, requested_start_ts: int,
                      force: bool = False) -> None:
    """Refuse to re-download a frozen snapshot, while still allowing appends.

    The hazard is not wasted time: bars are fetched with `auto_adjust=True`, so
    a split or dividend **restates** every earlier price. Re-pull a stock after
    a 2-for-1 split and its whole pre-split history silently halves, making any
    earlier result unreproducible with nothing in the data to explain why.

    The rule keys on the REQUESTED start, not on the resolved incremental one.
    `collect_ticker` always resumes from `last_bar + 1s`, which is before the
    stamp, so keying on that would refuse every append too — and quietly break
    the Phase 7 live monitor weeks later, in a different phase.
    """
    if not cfg["market"]["snapshot_frozen"] or force:
        if force:
            log.warning("--force: bypassing the %s snapshot freeze. yfinance "
                        "restates history after splits, so bars already stored "
                        "may change and earlier results stop reproducing.",
                        interval)
        return
    stamp = db.get_meta(conn, f"snapshot_frozen_{interval}")
    if stamp is None:
        raise SystemExit(
            f"market.snapshot_frozen is true but no snapshot_frozen_{interval} "
            f"stamp exists in `meta`. This function runs BEFORE --stamp-snapshot "
            f"ever gets a chance to write that stamp, so passing --stamp-snapshot "
            f"alone will hit this same error again — pass --force together with "
            f"--stamp-snapshot to do the initial download and stamp it in one run "
            f"(there is nothing frozen yet to restate), or set snapshot_frozen: "
            f"false if you don't want the freeze protection yet — a flag with no "
            f"evidence behind it is worse than no flag."
        )
    stamp_ts = iso_utc_to_ts(stamp)
    if requested_start_ts < stamp_ts:
        raise SystemExit(
            f"{interval} bars were frozen at {stamp}. Refusing to re-download "
            f"from {ts_to_iso(requested_start_ts)}: yfinance restates history "
            f"after splits, so this would silently change bars already used in "
            f"results. Append newer bars with --start after the stamp, or pass "
            f"--force if you genuinely mean to replace the snapshot."
        )


def _earliest_bar_ts(conn, ticker: str, interval: str) -> int | None:
    """MIN(ts_utc) for one ticker/interval — the mirror of `db.latest_bar_ts`.

    Kept local to this module rather than added to `src/db.py` (out of scope
    for this unit); it is the other half of the incremental-resume check in
    `collect_ticker` below, needed to tell "the cache reaches back far enough"
    apart from "the cache merely has SOME bar at or after the requested start".
    """
    row = conn.execute(
        "SELECT MIN(ts_utc) FROM bars WHERE ticker = ? AND interval = ?",
        (ticker, interval),
    ).fetchone()
    return row[0]


def collect_ticker(conn, ticker: str, start_ts: int, end_ts: int,
                   interval: str,
                   limiter: RateLimiter | None = None,
                   force: bool = False) -> FetchResult:
    """Fetch and upsert bars for one ticker, resuming from the cache.

    The rate limiter is waited immediately before the request and not before
    the cache check, so a run that skips thousands of already-cached tickers
    does not also sleep a second for each of them.

    The "already covered" shortcut only fires when the requested start falls
    INSIDE the cached range (`cached_min <= start_ts <= cached_max`) — not
    merely when *some* cached bar is at or after `start_ts`. Comparing against
    `MAX(ts_utc)` alone used to treat a backfill window entirely BEFORE the
    earliest cached bar as "already covered" (because the latest bar happened
    to be later than the requested end), silently skipping the fetch. `force`
    bypasses this cache check entirely, matching its documented purpose of
    re-downloading even data that looks already covered.

    `start_ts` arrives already clamped to yfinance's intraday history window:
    `collect_many` does that once, before its range check, so a window lying
    entirely outside that history fails there loudly instead of reaching here
    and being reported as "cache already covers window".
    """
    if not force:
        cached_min = _earliest_bar_ts(conn, ticker, interval)
        cached_max = db.latest_bar_ts(conn, ticker, interval)
        if (cached_min is not None and cached_max is not None
                and cached_min <= start_ts <= cached_max):
            start_ts = cached_max + 1  # incremental: refetch nothing we already have
    if start_ts >= end_ts:
        log.info("%s [%s]: cache already covers window", ticker, interval)
        return FetchResult(attempted=False, parsed=0, written=0)
    if limiter is not None:
        limiter.wait()
    hist = yf.Ticker(ticker).history(
        start=ts_to_dt(start_ts), end=ts_to_dt(end_ts),
        interval=interval, auto_adjust=True,
    )
    rows = df_to_rows(hist, ticker, interval)
    n = db.upsert_bars(conn, rows)
    log.info("%s [%s]: %d bars upserted (%s -> %s)", ticker, interval,
             len(rows), ts_to_iso(start_ts), ts_to_iso(end_ts))
    return FetchResult(attempted=True, parsed=len(rows), written=n)


def collect_many(cfg: dict, conn, tickers: list[str], start_ts: int,
                 end_ts: int, interval: str, resume: bool = False,
                 force: bool = False) -> int:
    """Collect bars for many tickers. Returns rows parsed across the run.

    Per-ticker failures are logged and the run continues — one delisted symbol
    must not cost the other 6,000. Every outcome is committed to `fetch_state`
    as it happens, so `--resume` continues an interrupted run.
    `KeyboardInterrupt` is deliberately not caught (`except Exception` does not
    cover it), so Ctrl-C stops the run with everything collected so far saved.
    """
    # Clamped HERE, before the range check, not per ticker inside
    # `collect_ticker`. The clamp can push `start_ts` past `end_ts` — a 60m
    # window entirely older than yfinance's intraday history does exactly that
    # — and done per ticker that turned every ticker into attempted=False,
    # which the guard below excludes from its denominator by design. The run
    # then logged "cache already covers window" for every symbol against an
    # empty database and exited 0. Clamping once makes it the same malformed
    # range as a swapped --start/--end, which the check below already refuses.
    start_ts = clamp_start(cfg, start_ts, interval, utc_now_ts())
    if start_ts >= end_ts:
        # A swapped/typo'd --start/--end used to be indistinguishable from a
        # legitimate "cache already covers window" no-op: every ticker would
        # come back attempted=False, which the zero-record guard below also
        # ignores, so the whole run "succeeded" having fetched nothing, for
        # every ticker, forever. Fail loudly on the malformed range instead.
        raise SystemExit(
            f"--start ({ts_to_iso(start_ts)}) is not before --end "
            f"({ts_to_iso(end_ts)}) — refusing an inverted or empty date "
            f"range instead of silently fetching nothing. For {interval} bars "
            f"the start is first clamped to yfinance's "
            f"{cfg['market']['hourly_max_lookback_days']}-day intraday "
            f"history, so a window entirely older than that lands here too."
        )
    source = fetch_source(interval)
    skip = db.completed_keys(conn, source) if resume else set()
    if skip:
        log.info("resume: skipping %d tickers already collected",
                 sum(1 for t in tickers if t in skip))
    # Tickers a previous run already found to return nothing. They are excluded
    # from the guard's denominator below: a --resume mop-up attempts exactly
    # these, and all of them coming back empty a second time is the expected
    # outcome, not evidence that yfinance has stopped answering.
    known_empty = db.keys_with_status(conn, source, "empty")

    limiter = RateLimiter(cfg["market"]["min_interval_s"])
    attempted = parsed_total = written_total = failed = empty = 0
    guard_attempts = guard_parsed = 0
    for ticker in tickers:
        if ticker in skip:
            continue
        guarded = ticker not in known_empty
        try:
            res = collect_ticker(conn, ticker, start_ts, end_ts, interval,
                                 limiter=limiter, force=force)
        except Exception as exc:
            failed += 1
            attempted += 1
            guard_attempts += guarded
            log.exception("failed to collect %s [%s] — continuing",
                          ticker, interval)
            db.set_fetch_state(conn, source, ticker, "failed",
                               error=f"{type(exc).__name__}: {exc}"[:500])
            continue
        if not res.attempted:
            continue  # cache already covered it; nothing was tried
        attempted += 1
        parsed_total += res.parsed
        written_total += res.written
        if guarded:
            guard_attempts += 1
            guard_parsed += res.parsed
        # After the upsert, never before: a crash between the two re-fetches
        # one ticker, which is free. The reverse order would mark a ticker done
        # whose bars never landed.
        #
        # Zero bars is 'empty', not 'ok', so --resume comes back to it. For
        # EDGAR a company that files no 8-Ks was a permanent, legitimate 'ok';
        # a listed company with no daily bars at all is not the same thing — it
        # is a delisted symbol or a transient Yahoo failure, and the two look
        # identical from here.
        if res.parsed:
            db.set_fetch_state(conn, source, ticker, "ok",
                               records=res.parsed, rows_written=res.written)
        else:
            empty += 1
            db.set_fetch_state(conn, source, ticker, "empty", records=0,
                               rows_written=0,
                               error="yfinance returned no bars for this window")

    n_bars = conn.execute("SELECT COUNT(*) FROM bars WHERE interval = ?",
                          (interval,)).fetchone()[0]
    log.info("Done [%s]. %d tickers attempted (%d failed, %d empty); "
             "%d bars parsed, %d rows written; bars[%s] now holds %d.",
             interval, attempted, failed, empty, parsed_total, written_total,
             interval, n_bars)

    # The silent-failure guard, at run level. It deliberately does NOT look at
    # the size of the bars table: the old `total == 0 and n_bars == 0` form
    # could only ever fire on a virgin database, so from the second run onward
    # a completely broken yfinance would have passed quietly.
    #
    # An empty frame arrives with no exception, which is precisely the "HTTP 200
    # carrying nothing" shape this project keeps guarding against, so unlike the
    # EDGAR guard this one counts empty responses and not only raised errors.
    if cfg["logging"]["fail_on_zero_records"] and guard_attempts and not guard_parsed:
        raise SystemExit(
            f"ZERO bars parsed across all {guard_attempts} tickers attempted "
            f"[{interval}] that were not already known to be empty — yfinance "
            f"is returning nothing. Do not treat this run as successful."
        )
    return parsed_total


# --------------------------------------------------------------------------
# coverage report
# --------------------------------------------------------------------------

def coverage_report(cfg: dict, conn, tickers: list[str],
                    interval: str) -> dict:
    """Does every candidate have bars spanning the required window?

    The required span is [default_start_ts, study_window.end] — for daily bars
    that reaches back before the window, because the liquidity filter needs the
    history. A boundary date can legitimately fall on a weekend or in a holiday
    week, so `market.coverage_tolerance_days` of slack is allowed at each end.
    """
    required_start = default_start_ts(cfg, interval)
    required_end = date_str_to_ts(cfg["study_window"]["end"])
    tol = cfg["market"]["coverage_tolerance_days"] * DAY_S
    coverage = db.bar_coverage(conn, interval)
    state = {
        row[0]: (row[1], row[2]) for row in conn.execute(
            "SELECT key, status, error FROM fetch_state WHERE source = ?",
            (fetch_source(interval),),
        )
    }

    report = {"interval": interval, "required_start": required_start,
              "required_end": required_end, "candidates": len(tickers),
              "covered": [], "missing": [], "late_start": [], "early_end": []}
    for ticker in tickers:
        got = coverage.get(ticker)
        if not got or not got[2]:
            status, error = state.get(ticker, ("never fetched", None))
            report["missing"].append((ticker, status, error))
            continue
        first_ts, last_ts, _ = got
        if first_ts > required_start + tol:
            report["late_start"].append((ticker, first_ts))
        elif last_ts < required_end - tol:
            report["early_end"].append((ticker, last_ts))
        else:
            report["covered"].append(ticker)
    return report


def print_coverage_report(report: dict, max_listed: int = 20) -> None:
    """Human-readable form of `coverage_report` — the P3-01 acceptance check."""
    print(f"\n=== Bar coverage [{report['interval']}] ===")
    print(f"Required span : {ts_to_iso(report['required_start'])} -> "
          f"{ts_to_iso(report['required_end'])}")
    print(f"Candidates    : {report['candidates']}")
    print(f"  covered     : {len(report['covered'])}")
    print(f"  late start  : {len(report['late_start'])}   "
          f"(IPO or relisting inside the span)")
    print(f"  early end   : {len(report['early_end'])}   "
          f"(delisted or acquired inside the span)")
    print(f"  missing     : {len(report['missing'])}   (no bars at all)")

    for label, key in (("Late start", "late_start"), ("Early end", "early_end")):
        rows = report[key]
        if rows:
            print(f"\n{label} — first {min(len(rows), max_listed)} of {len(rows)}:")
            for ticker, ts in rows[:max_listed]:
                print(f"  {ticker:<8} {ts_to_iso(ts)}")
    if report["missing"]:
        rows = report["missing"]
        print(f"\nMissing — first {min(len(rows), max_listed)} of {len(rows)}:")
        for ticker, status, error in rows[:max_listed]:
            print(f"  {ticker:<8} {status:<8} {error or ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tickers", help="comma-separated, e.g. TSLA,AAPL")
    group.add_argument("--candidates", action="store_true",
                       help="every ticker in `companies` — the pre-filter list, "
                            "which is what the liquidity filter is computed from")
    group.add_argument("--universe", action="store_true",
                       help="every ticker that passed the liquidity filter, "
                            "plus the benchmark")
    parser.add_argument("--start", help="YYYY-MM-DD (default: per interval — "
                                       "daily reaches back min_history_days)")
    parser.add_argument("--end", help="YYYY-MM-DD (default: now)")
    parser.add_argument("--interval", choices=["60m", "1d"],
                        help="60m (default) or 1d — the only two intervals "
                             "this collector fetches or stores (see module "
                             "docstring: 1m/5m/15m/30m are NOT used)")
    parser.add_argument("--resume", action="store_true",
                        help="skip tickers already collected for this interval")
    parser.add_argument("--report", action="store_true",
                        help="print the coverage report and exit, fetching nothing")
    parser.add_argument("--force", action="store_true",
                        help="re-download even though the snapshot is frozen. "
                             "yfinance restates history after splits, so this "
                             "can change bars already used in results.")
    parser.add_argument("--stamp-snapshot", action="store_true",
                        help="record this run's date as the frozen snapshot date")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    mcfg = cfg["market"]
    interval = args.interval or mcfg["interval"]
    start_ts = (date_str_to_ts(args.start) if args.start
                else default_start_ts(cfg, interval))
    end_ts = date_str_to_ts(args.end) if args.end else utc_now_ts()

    conn = db.get_conn(cfg["paths"]["db"])
    if args.candidates:
        tickers = db.candidate_tickers(conn)
        if not tickers:
            raise SystemExit(
                "companies table is empty — run `python -m src.collectors.edgar "
                "--build-universe` first."
            )
        if mcfg["benchmark"] not in tickers:
            tickers.append(mcfg["benchmark"])  # SPY market control
    elif args.universe:
        tickers = db.universe_tickers(conn)
        if not tickers:
            raise SystemExit(
                "no company has in_universe = 1 — run the liquidity filter "
                "first, or use --candidates for the unfiltered list."
            )
        if mcfg["benchmark"] not in tickers:
            tickers.append(mcfg["benchmark"])
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if args.report:
        print_coverage_report(coverage_report(cfg, conn, tickers, interval))
        return

    assert_not_frozen(cfg, conn, interval, start_ts, force=args.force)
    if args.force and mcfg["snapshot_frozen"] and len(tickers) > 1:
        # --force now genuinely bypasses the per-ticker incremental-cache
        # check (see collect_ticker), not just the frozen-snapshot assertion.
        # The freeze is whole-run, not per-ticker: --force on a multi-ticker
        # batch (--universe/--candidates, or a --tickers list mixing new and
        # already-frozen symbols) will re-download and potentially restate
        # EVERY ticker in it, not just the new ones. Scope --tickers to just
        # the new symbols to force-refresh only those.
        log.warning("--force with %d tickers: every one of them may be "
                    "re-downloaded and restated, not just new ones. If you "
                    "only meant to backfill/add specific tickers, re-run with "
                    "--tickers limited to those symbols.", len(tickers))
    collect_many(cfg, conn, tickers, start_ts, end_ts, interval,
                 resume=args.resume, force=args.force)

    if args.stamp_snapshot:
        stamp = ts_to_iso(utc_now_ts())
        db.set_meta(conn, f"snapshot_frozen_{interval}", stamp, utc_now_ts())
        log.info("Snapshot date for %s bars recorded as %s. Do not re-download.",
                 interval, stamp)


if __name__ == "__main__":
    main()
