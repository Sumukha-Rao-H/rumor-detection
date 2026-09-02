"""Materiality — did the market actually react?

An 8-K is only a positive if something moved. Without this filter, routine
dividend declarations and credit-facility amendments become "events" and the
target the model learns is mostly paperwork. Roughly two thirds of 8-Ks are
administrative.

The move is measured benchmark-relative: if a stock rose 4% on a day the whole
market rose 4%, nothing happened to that company in particular.

⚠ This is a LABEL, computed deliberately from prices AFTER t0. It must never
become a feature — a model that can see the outcome has already won. The
leakage tests guard the feature path; this module is on the label path and is
the one place where looking forward is correct.

⚠ The 3% threshold was fixed in config before any model existed and is applied
here exactly as written. If the surviving count is low, that is a finding to
report (P4-05), never a reason to move the bar.

Usage:
  python -m src.pipeline.materiality            # apply and report
  python -m src.pipeline.materiality --report   # report only, write nothing
"""

from __future__ import annotations

import argparse
import bisect
import logging
from collections import Counter
from typing import NamedTuple

from src import db
from src.utils.config import load_config

log = logging.getLogger(__name__)

HOUR_S = 3600


def interval_seconds(interval: str) -> int:
    """Bar duration in seconds, e.g. '60m' -> 3600, '1d' -> 86400."""
    unit, n = interval[-1], int(interval[:-1])
    try:
        return n * {"m": 60, "h": 3600, "d": 86400}[unit]
    except KeyError:
        raise ValueError(f"unrecognized interval: {interval!r}") from None


class Move(NamedTuple):
    """One event's price reaction."""
    raw: float | None            # the stock's own return
    adjusted: float | None       # benchmark-subtracted, when possible
    benchmark_applied: bool
    reason: str | None           # why it could not be measured


class BarSeries:
    """Sorted (timestamp, open, close) for one ticker, with as-of lookups.

    Loaded once per run: a per-event query over 2.6 million bars would be
    ~15,000 round trips for numbers we already have in memory.

    Bars are keyed by their OPEN time (src/collectors/market.py), each
    nominally covering [open, open + interval_s). `interval_s` is needed by
    `at_or_before` to tell a bar that has fully finished trading before `ts`
    from one that is still live when `ts` happens — see there.
    """

    def __init__(self, rows: list[tuple[int, float, float]],
                interval_s: int) -> None:
        self.ts = [r[0] for r in rows]
        self.open = [r[1] for r in rows]
        self.close = [r[2] for r in rows]
        self.interval_s = interval_s

    def at_or_before(self, ts: int) -> float | None:
        """The pre-news price as of `ts`.

        Two cases, both keyed off the same bar — the last one whose open is
        at or before `ts` (`bisect_right(self.ts, ts) - 1`):

        - That bar's whole window already finished before `ts`
          (`open + interval_s <= ts`): its CLOSE is a clean pre-`ts` price.
        - `ts` falls inside that bar's still-live window, including exactly
          at its open: the bar's CLOSE reflects trading that happens AFTER
          `ts` too, so it is not safe to use. Its OPEN is — it is the last
          price printed at or before the instant `ts` occurs. Using the
          close here would silently fold part of the real reaction into the
          "before" side and understate, or erase, the measured move.
        """
        i = bisect.bisect_right(self.ts, ts)
        if i == 0:
            return None
        idx = i - 1
        if self.ts[idx] + self.interval_s <= ts:
            return self.close[idx]
        return self.open[idx]

    def at_or_after(self, ts: int) -> float | None:
        """Close of the first bar at or after `ts`.

        This is what makes the horizon skip a closed market. For a Friday
        evening t0 the 24-hour mark lands on Saturday, and the first bar at or
        after it is Monday's — which is when the reaction could first happen.
        """
        i = bisect.bisect_left(self.ts, ts)
        return self.close[i] if i < len(self.ts) else None


def load_series(conn, interval: str) -> dict[str, BarSeries]:
    """Every ticker's open/close series, in one pass over `bars`."""
    interval_s = interval_seconds(interval)
    rows: dict[str, list[tuple[int, float, float]]] = {}
    for ticker, ts, o, close in conn.execute(
            "SELECT ticker, ts_utc, open, close FROM bars WHERE interval = ? "
            "ORDER BY ticker, ts_utc", (interval,)):
        rows.setdefault(ticker, []).append((ts, o, close))
    return {t: BarSeries(v, interval_s) for t, v in rows.items()}


def measure(cfg: dict, series: dict[str, BarSeries], ticker: str,
            t0_utc: int) -> Move:
    """The benchmark-relative move over the materiality horizon.

    The horizon is a MINIMUM, not a hard cutoff: `window_hours` after t0, then
    the first available close. 8.1% of events sit where no trading happens in
    the next 24 wall-clock hours — 1,138 of them Friday filings — and a literal
    cutoff would score every one as a zero move and drop it. Companies release
    bad news on Friday afternoon precisely because the market cannot answer
    until Monday, so deleting that category would be a selection bias, not a
    rounding error.

    Note: there is deliberately no upper bound on how far `at_or_after` may
    have to look — a multi-week trading halt would silently stretch a "24h"
    label over however long it takes the market to reopen. Real data tops out
    at a 4.75-day gap (long weekends/holidays), so this has not bitten in
    practice; a hard ceiling would need its own exclude_reason and is not
    implemented here.
    """
    mcfg = cfg["materiality"]
    horizon = t0_utc + mcfg["window_hours"] * HOUR_S

    s = series.get(ticker)
    if s is None:
        return Move(None, None, False, "no_bars")
    start = s.at_or_before(t0_utc)
    if start is None or not (start > 0):    # `> 0` also excludes NaN, unlike `<= 0`
        return Move(None, None, False, "no_pre_t0_bar")
    end = s.at_or_after(horizon)
    if end is None or not (end > 0):
        return Move(None, None, False, "no_post_t0_bar")

    raw = end / start - 1.0
    if not mcfg["benchmark_relative"]:
        return Move(raw, raw, False, None)

    bench = series.get(cfg["market"]["benchmark"])
    b_start = bench.at_or_before(t0_utc) if bench else None
    b_end = bench.at_or_after(horizon) if bench else None
    if b_start is None or not (b_start > 0) or b_end is None or not (b_end > 0):
        # Recorded rather than silently treated as adjusted: a raw return
        # labelled benchmark-relative would misstate what was measured.
        return Move(raw, raw, False, None)
    return Move(raw, raw - (b_end / b_start - 1.0), True, None)


def apply_filter(cfg: dict, conn) -> list[dict]:
    """Measure every unexcluded event and decide `usable`.

    `usable` is the AND of every gate — no exclusion, and a real reaction. It
    is set here and only here, because materiality is the last gate.

    Already-excluded events are not measured: no price move turns an
    attachment-only filing into an event.
    """
    mcfg = cfg["materiality"]
    series = load_series(conn, cfg["market"]["interval"])
    rows = conn.execute(
        "SELECT event_id, ticker, t0_utc, exclude_reason FROM events").fetchall()
    if not rows:
        raise SystemExit(
            "no events — run `python -m src.pipeline.t0 --build` and "
            "`python -m src.pipeline.events` first."
        )

    out = []
    for r in rows:
        # The mirror of events.py's owned-vs-preserved rule. A reason from an
        # UPSTREAM pass (item filtering, coverage, universe) is preserved:
        # this module is not entitled to overturn it. A reason this module
        # itself wrote is RE-MEASURED, because the inputs behind it move —
        # `no_pre_t0_bar` becomes measurable once the market collector
        # backfills that bar, and `immaterial` flips the moment
        # `min_abs_return` is retuned. Skipping those re-persisted the first
        # run's verdict forever, silently freezing the positive count against
        # data and config that had since changed.
        if r["exclude_reason"] and r["exclude_reason"] not in _OWNED_REASONS:
            out.append({"event_id": r["event_id"], "abs_return": None,
                        "is_material": 0, "usable": 0,
                        "exclude_reason": r["exclude_reason"]})
            continue
        move = measure(cfg, series, r["ticker"], r["t0_utc"])
        if move.reason:
            out.append({"event_id": r["event_id"], "abs_return": None,
                        "is_material": 0, "usable": 0,
                        "exclude_reason": move.reason})
            continue
        abs_ret = abs(move.adjusted)
        material = int(abs_ret >= mcfg["min_abs_return"])   # inclusive
        out.append({"event_id": r["event_id"], "abs_return": abs_ret,
                    "is_material": material, "usable": material,
                    "exclude_reason": None if material else "immaterial"})

    if not any(r["usable"] for r in out):
        raise SystemExit(
            f"NO event out of {len(out)} cleared the materiality bar "
            f"({mcfg['min_abs_return']:.0%} over {mcfg['window_hours']}h). "
            f"Check the benchmark bars before trusting this."
        )
    return out


def write_filter(cfg: dict, conn) -> int:
    """Apply and store. Returns the usable count."""
    rows = apply_filter(cfg, conn)
    conn.executemany(
        "UPDATE events SET abs_return = :abs_return, is_material = :is_material,"
        " usable = :usable, exclude_reason = :exclude_reason "
        "WHERE event_id = :event_id", rows)
    conn.commit()
    usable = sum(r["usable"] for r in rows)
    log.info("materiality: %d events measured, %d usable", len(rows), usable)
    return usable


#: exclude_reasons that `measure()` itself assigns for missing/bad price data,
#: as opposed to reasons assigned earlier by universe.py/events.py/t0.py. Kept
#: distinct in `print_report` so "excluded earlier" only ever means the latter.
_NO_PRICE_REASONS = {"no_bars", "no_pre_t0_bar", "no_post_t0_bar"}

#: Every exclude_reason THIS module writes. `apply_filter` re-measures these
#: on each run rather than trusting the stored verdict, because both halves
#: are inputs-dependent: the price reasons resolve when the missing bars are
#: backfilled, and `immaterial` is a function of `min_abs_return`, a config
#: knob. Any reason NOT in this set belongs to an earlier pass and is left
#: exactly as found.
_OWNED_REASONS = _NO_PRICE_REASONS | {"immaterial"}


def print_report(cfg: dict, conn) -> None:
    """The kept/dropped counts the acceptance criterion asks for."""
    mcfg = cfg["materiality"]
    rows = conn.execute(
        "SELECT is_scheduled, usable, abs_return, exclude_reason "
        "FROM events").fetchall()
    if not rows:
        raise SystemExit(
            "no events — run `python -m src.pipeline.t0 --build` and "
            "`python -m src.pipeline.events` first."
        )
    usable = [r for r in rows if r["usable"]]
    immaterial = [r for r in rows if r["exclude_reason"] == "immaterial"]
    no_price = [r for r in rows if r["exclude_reason"] in _NO_PRICE_REASONS]
    excluded_earlier = [r for r in rows if r["exclude_reason"]
                        and r["exclude_reason"] not in _NO_PRICE_REASONS
                        and r["exclude_reason"] != "immaterial"]

    print(f"\n=== Materiality ({mcfg['min_abs_return']:.0%} over "
          f"{mcfg['window_hours']}h, "
          f"{'benchmark-relative' if mcfg['benchmark_relative'] else 'raw'}) ===")
    print(f"threshold fixed in config before any model existed — applied as written\n")
    print(f"events            : {len(rows):,}")
    print(f"  excluded earlier: {len(excluded_earlier):,}  "
          f"(universe/events/t0 gates — decided before this module ran)")
    print(f"  no price data   : {len(no_price):,}  "
          f"(this module's own bar lookup found nothing usable)")
    if no_price:
        for reason, n in sorted(Counter(r["exclude_reason"] for r in no_price).items()):
            print(f"    {reason:<15}: {n:,}")
    print(f"  immaterial      : {len(immaterial):,}")
    print(f"  USABLE          : {len(usable):,}  ({len(usable)/len(rows):.1%} of all events)")

    if usable:
        sched = sum(r["is_scheduled"] for r in usable)
        print(f"\nusable, split as every headline number must be:")
        print(f"  scheduled   : {sched:,}  ({sched/len(usable):.1%})")
        print(f"  unscheduled : {len(usable)-sched:,}  "
              f"({1-sched/len(usable):.1%})")
        moves = sorted(r["abs_return"] for r in usable)
        print(f"\n|move| among usable events: median "
              f"{moves[len(moves)//2]:.1%}, p90 {moves[int(len(moves)*.9)]:.1%}, "
              f"max {moves[-1]:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true",
                        help="report only, writing nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    if not args.report:
        write_filter(cfg, conn)
    print_report(cfg, conn)


if __name__ == "__main__":
    main()
