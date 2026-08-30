"""Liquidity filter — decides which companies the study actually covers.

A volume spike only carries information in a stock whose volume is normally
steady. Where 400 shares trade a day, 4,000 shares is a tenfold spike and also
one person opening a small position; left in, those companies teach the
detector noise. So the universe is cut down to companies with real, continuous
trading.

Everything here is computed from bars at or before `study_window.start`, and
nothing after it is ever read. That is the entire point of the module. A
universe built from today's prices silently drops every company that was
acquired or delisted during the window — precisely the dramatic, market-moving
population this project exists to detect. A company that was liquid on the
window-start date and delisted eight months later belongs in the study; its
bars simply stop early, which is the coverage audit's problem, not this one's.

Usage:
  python -m src.pipeline.universe            # apply the filter and write flags
  python -m src.pipeline.universe --report   # explain the cut, write nothing
"""

from __future__ import annotations

import argparse
import logging
from typing import NamedTuple

from src import db
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, ts_to_iso

log = logging.getLogger(__name__)

DAY_S = 86400


class Candidate(NamedTuple):
    """One company's liquidity picture, measured as of the window start."""
    ticker: str
    first_ts: int | None      # earliest daily bar ever, at or before as-of
    last_close: float | None  # last close at or before as-of
    adv_usd: float | None     # mean close*volume over the lookback
    n_bars: int               # bars inside the lookback


def as_of_ts(cfg: dict) -> int:
    """The date every measurement is taken at.

    Only `start` is accepted. "As of today" is the exact mistake this module
    exists to prevent, so an unrecognised value raises rather than quietly
    meaning something else.
    """
    mode = cfg["universe"]["as_of"]
    if mode != "start":
        raise ValueError(
            f"universe.as_of is {mode!r}; only 'start' is supported. Measuring "
            f"liquidity as of today would drop every company acquired or "
            f"delisted during the window — the events this project detects."
        )
    return date_str_to_ts(cfg["study_window"]["start"])


def gather_candidates(cfg: dict, conn) -> dict[str, Candidate]:
    """Liquidity statistics per ticker, from bars at or before the as-of date.

    Aggregated in SQL: 2.7 million bars is not worth pulling into Python to
    take a mean. Three passes — earliest bar ever, the lookback average, and
    the last close — each bounded above by the as-of date so no future bar can
    reach the result.
    """
    ucfg = cfg["universe"]
    interval = cfg["market"]["daily_interval"]
    cutoff = as_of_ts(cfg)
    lookback_start = cutoff - ucfg["min_history_days"] * DAY_S

    first = {r[0]: r[1] for r in conn.execute(
        "SELECT ticker, MIN(ts_utc) FROM bars "
        "WHERE interval = ? AND ts_utc <= ? GROUP BY ticker",
        (interval, cutoff))}

    window = {r[0]: (r[1], r[2]) for r in conn.execute(
        "SELECT ticker, AVG(close * volume), COUNT(*) FROM bars "
        "WHERE interval = ? AND ts_utc BETWEEN ? AND ? GROUP BY ticker",
        (interval, lookback_start, cutoff))}

    last = {r[0]: r[1] for r in conn.execute(
        """SELECT b.ticker, b.close FROM bars b
             JOIN (SELECT ticker, MAX(ts_utc) AS mx FROM bars
                    WHERE interval = ? AND ts_utc <= ? GROUP BY ticker) m
               ON b.ticker = m.ticker AND b.ts_utc = m.mx
            WHERE b.interval = ?""",
        (interval, cutoff, interval))}

    out = {}
    for ticker in db.candidate_tickers(conn):
        adv, n = window.get(ticker, (None, 0))
        out[ticker] = Candidate(ticker, first.get(ticker), last.get(ticker),
                                adv, n)
    return out


def classify(cfg: dict, cand: Candidate) -> str | None:
    """None if the company qualifies, else the reason it does not.

    Checked in a fixed order so every exclusion has exactly one reason, and the
    report adds up.
    """
    ucfg = cfg["universe"]
    cutoff = as_of_ts(cfg)
    # The history boundary lands on a calendar date that may be a weekend or a
    # holiday, so the earliest possible bar sits days after it. Reuses the
    # coverage tolerance rather than adding a second knob for the same fact —
    # a strict comparison here returns zero companies.
    tolerance = cfg["market"]["coverage_tolerance_days"] * DAY_S
    history_by = cutoff - ucfg["min_history_days"] * DAY_S + tolerance

    if cand.first_ts is None or not cand.n_bars:
        return "no_bars"
    if cand.first_ts > history_by:
        return "short_history"
    if cand.last_close is None or cand.last_close < ucfg["min_price_usd"]:
        return "below_min_price"
    if cand.adv_usd is None or cand.adv_usd < ucfg["min_adv_usd"]:
        return "below_min_adv"
    return None


def select_universe(cfg: dict, conn) -> tuple[list[Candidate], dict[str, int]]:
    """Apply the filter. Returns (survivors, exclusion counts).

    `max_tickers` is a budget applied to whatever clears the thresholds, not a
    fifth threshold: the result is "the most liquid companies that qualify",
    never an arbitrary 1,500. Ordered by ADV then ticker so the cut at the
    boundary is the same on every run.
    """
    candidates = gather_candidates(cfg, conn)
    reasons: dict[str, int] = {}
    passed = []
    for cand in candidates.values():
        reason = classify(cfg, cand)
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            passed.append(cand)

    passed.sort(key=lambda c: (-c.adv_usd, c.ticker))
    cap = cfg["universe"]["max_tickers"]
    if len(passed) > cap:
        reasons["over_max_tickers"] = len(passed) - cap
        passed = passed[:cap]
    return passed, reasons


def apply_filter(cfg: dict, conn) -> list[Candidate]:
    """Select the universe and write the flags. Returns the survivors."""
    survivors, reasons = select_universe(cfg, conn)
    if not survivors:
        raise SystemExit(
            "liquidity filter kept ZERO companies — check that daily bars are "
            "collected for the window. Every later phase would download "
            "nothing while reporting success."
        )
    cutoff = as_of_ts(cfg)
    db.clear_universe_flags(conn)   # not additive: a company that no longer
                                    # qualifies must be demoted, not left set
    written = db.set_universe_flags(conn, [
        {"ticker": c.ticker, "adv_usd": c.adv_usd,
         "last_price": c.last_close, "as_of_utc": cutoff}
        for c in survivors
    ])
    log.info("Universe: %d companies flagged (as of %s); excluded %s",
             written, ts_to_iso(cutoff),
             ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    return survivors


def print_report(cfg: dict, conn, max_listed: int = 15) -> None:
    """Explain the cut — how many, why the rest went, and where the line fell."""
    survivors, reasons = select_universe(cfg, conn)
    ucfg = cfg["universe"]
    total = len(db.candidate_tickers(conn))
    print(f"\n=== Liquidity filter (as of {ts_to_iso(as_of_ts(cfg))}) ===")
    print(f"thresholds: price >= ${ucfg['min_price_usd']}, "
          f"ADV >= ${ucfg['min_adv_usd']:,}, "
          f"history >= {ucfg['min_history_days']}d, cap {ucfg['max_tickers']}")
    print(f"\ncandidates : {total}")
    print(f"in universe: {len(survivors)}")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  excluded {reason:<18} {n}")

    if survivors:
        print(f"\nMost liquid — top {min(len(survivors), max_listed)}:")
        for c in survivors[:max_listed]:
            print(f"  {c.ticker:<8} ADV ${c.adv_usd/1e6:>10,.1f}M  "
                  f"close ${c.last_close:,.2f}")
        cut = survivors[-1]
        print(f"\nThe line fell at: {cut.ticker} — ADV ${cut.adv_usd/1e6:,.1f}M, "
              f"close ${cut.last_close:,.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true",
                        help="explain the cut and exit, writing nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    if args.report:
        print_report(cfg, conn)
        return
    apply_filter(cfg, conn)
    print_report(cfg, conn)


if __name__ == "__main__":
    main()
