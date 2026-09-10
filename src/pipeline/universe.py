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
A company that had already stopped trading BEFORE that date does not belong:
its last bar sits months behind the cutoff and it never traded during the
window at all.

**KNOWN LIMITATION — the guarantee above is not delivered end-to-end.**
Measured 2026-09-01, decided 2026-09-02: keep it and report it, do not
re-collect. Everything this module does is correct, and it is not enough.
`db.candidate_tickers` reads `companies`, which `edgar.py` builds from SEC's
company_tickers_exchange.json — a LIVE register listing who holds a ticker
TODAY. A company acquired or delisted during the window has already fallen
off it, so it was never ingested and can never reach this filter to be kept.
The evidence: across 5,540 tickers over eleven months, **zero** have daily
history ending more than 90 days before the snapshot. The delisted population
is not thinned, it is absent.

So the as-of arithmetic here guards a door that is already open one stage
upstream, and the honest statement of scope is: **results hold for companies
that were listed at the window start and still listed at the snapshot date.**
Acquisitions and delistings — dramatic, market-moving, and exactly the
population the project most wants — are outside it. Closing this needs a
point-in-time company register, a full EDGAR re-collection, and a new price
snapshot; that is a bigger change than the finding warrants right now, so it
is a stated bound on the claim rather than a silent one. It belongs in the
report next to the Finnhub twelve-month bound, which is handled the same way.

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
    last_ts: int | None       # latest daily bar at or before as-of
    last_close: float | None  # close of that bar
    adv_usd: float | None     # mean close*volume over the lookback
    n_bars: int               # bars inside the lookback
    files_8k: bool            # filed a RECENT 8-K before the window (P3-02b).
    # No default on purpose: it decides membership, and a default of True would
    # let a Candidate built without filing evidence qualify on silence.


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

    last = {r[0]: (r[1], r[2]) for r in conn.execute(
        """SELECT b.ticker, b.ts_utc, b.close FROM bars b
             JOIN (SELECT ticker, MAX(ts_utc) AS mx FROM bars
                    WHERE interval = ? AND ts_utc <= ? GROUP BY ticker) m
               ON b.ticker = m.ticker AND b.ts_utc = m.mx
            WHERE b.interval = ?""",
        (interval, cutoff, interval))}

    # Measured strictly before the window: keying on in-window filings would
    # build the universe out of the outcome and hand every member a positive.
    # Bounded below as well: "ever filed an 8-K since 1994" admits a trust on
    # two administrative filings from a decade ago. One query with both bounds
    # rather than `db.tickers_with_filings_before` (which has no lower bound)
    # intersected with a second set — that intersection would count a ticker
    # whose only recent filing lands INSIDE the window, i.e. look-ahead.
    forms = cfg["edgar"]["forms"]
    marks = ",".join("?" * len(forms))
    since = cutoff - ucfg["prior_8k_lookback_days"] * DAY_S
    prior_filers = {r[0] for r in conn.execute(
        f"SELECT DISTINCT ticker FROM filings "
        f"WHERE acceptance_utc >= ? AND acceptance_utc < ? "
        f"AND form IN ({marks}) AND ticker IS NOT NULL",
        (since, cutoff, *forms))}

    out = {}
    for ticker in db.candidate_tickers(conn):
        adv, n = window.get(ticker, (None, 0))
        last_ts, last_close = last.get(ticker, (None, None))
        out[ticker] = Candidate(ticker, first.get(ticker), last_ts, last_close,
                                adv, n, ticker in prior_filers)
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

    # Checked first: "this entity cannot produce an event at all" is more
    # fundamental than "it trades too thinly to measure", and the report then
    # reads in order of severity. Foreign private issuers file 6-K/20-F and are
    # exempt from 8-K; ETFs file neither. Either way the answer key is silent
    # about them forever, so they can only contribute negatives while occupying
    # a capped slot that a real filer would otherwise hold.
    if ucfg["require_prior_8k"] and not cand.files_8k:
        return "no_prior_8k"
    if cand.first_ts is None or not cand.n_bars:
        return "no_bars"
    if cand.first_ts > history_by:
        return "short_history"
    # Liquid a year ago and gone before the window opened is not a member. The
    # docstring's promise is the other direction — a company alive AT the start
    # and delisted during the window stays, and its last bar is at the cutoff.
    # Checked before the bar count because a name that died mid-lookback is
    # short of bars as well, and "it stopped trading" is the truer reason.
    if (cand.last_ts is None
            or cutoff - cand.last_ts > ucfg["max_bar_staleness_days"] * DAY_S):
        return "not_trading_at_as_of"
    # ADV is a mean, so the denominator matters: over four bars it measures four
    # days, not a year. Without a floor a name that traded one week outranks
    # every continuously-traded company and takes a capped slot from it.
    if cand.n_bars < ucfg["min_bars_in_lookback"]:
        return "too_few_bars"
    # Bars exist but carry no price or volume. Distinct from "too cheap" and
    # from "too thin": those are facts about the company, this is a data fault,
    # and reporting it as thin trading would hide a broken collector cycle.
    if cand.last_close is None or cand.adv_usd is None:
        return "no_usable_bars"
    if cand.last_close < ucfg["min_price_usd"]:
        return "below_min_price"
    if cand.adv_usd < ucfg["min_adv_usd"]:
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


def write_flags(conn, survivors: list[Candidate], cutoff: int) -> int:
    """Clear every flag and set the survivors' — in ONE transaction.

    The transaction, the SQL and the all-or-nothing count check now live in
    `db.replace_universe_flags`, which is where they belong: this used to
    duplicate both statements here precisely because the two db-layer functions
    committed separately, and a crash between them left `in_universe = 0` on
    every row while every downstream stage read an empty universe and reported
    success. `SystemExit` rather than the `ValueError` the db layer raises,
    because this is what a CLI run should exit on.
    """
    rows = [{"ticker": c.ticker, "adv_usd": c.adv_usd,
             "last_price": c.last_close, "as_of_utc": cutoff}
            for c in survivors]
    try:
        written = db.replace_universe_flags(conn, rows, expected=len(survivors))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return written


def check_drift(cfg: dict, conn) -> dict:
    """Does the STORED universe still match what this config computes?

    `companies.in_universe` is written once and then read by every later stage.
    Nothing re-derives it, so a config change after the write leaves the code
    and the data disagreeing silently — which is exactly what happened:
    `prior_8k_lookback_days` was added on 2026-09-02 and the filter was never
    re-run, so the flags in the database predate the knob and the comment
    beside it describes an effect that is not in force.

    Reported, never repaired. Re-running the filter today would swap 14 flagged
    members for 14 that have **zero** hourly bars — the hourly snapshot was
    collected for the universe as it stood, and `market.snapshot_frozen` means
    the missing history cannot be fetched without `--force` restating the
    frozen prices. So the honest state is "config and data differ, here is the
    difference", not a silent re-write in either direction.

    Returns `{"drift": n, "flagged_only": [...], "computed_only": [...]}`.
    """
    survivors, _ = select_universe(cfg, conn)
    computed = {c.ticker for c in survivors}
    flagged = {r[0] for r in conn.execute(
        "SELECT ticker FROM companies WHERE in_universe = 1")}
    return {"drift": len(computed ^ flagged),
            "flagged_only": sorted(flagged - computed),
            "computed_only": sorted(computed - flagged),
            "n_computed": len(computed), "n_flagged": len(flagged)}


def print_drift(cfg: dict, conn) -> None:
    """The drift guard's report. Says which way the difference runs and why it
    is not simply corrected."""
    d = check_drift(cfg, conn)
    print("\n=== Universe drift: stored flags vs this config ===")
    print(f"computed {d['n_computed']:,}  flagged {d['n_flagged']:,}  "
          f"differing tickers {d['drift']}")
    if not d["drift"]:
        print("  in agreement — the stored universe is what this config selects")
        return
    print(f"  flagged in the DB, NOT selected now ({len(d['flagged_only'])}): "
          f"{', '.join(d['flagged_only'])}")
    print(f"  selected now, NOT flagged ({len(d['computed_only'])}): "
          f"{', '.join(d['computed_only'])}")
    missing = [t for t in d["computed_only"] if not conn.execute(
        "SELECT 1 FROM bars WHERE ticker = ? AND interval = ? LIMIT 1",
        (t, cfg["market"]["interval"])).fetchone()]
    if missing:
        print(f"\n  ⚠ {len(missing)} of the newly-selected have NO "
              f"{cfg['market']['interval']} bars at all: {', '.join(missing)}")
        print("  Re-running the filter would seat them in capped slots they "
              "cannot fill, and the frozen snapshot cannot be extended to "
              "cover them without --force restating it. Reported, not fixed.")


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
    written = write_flags(conn, survivors, cutoff)
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
    print(f"            bars >= {ucfg['min_bars_in_lookback']} in the lookback, "
          f"last bar within {ucfg['max_bar_staleness_days']}d of the as-of date, "
          f"8-K within {ucfg['prior_8k_lookback_days']}d before it")
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
    parser.add_argument("--check", action="store_true",
                        help="report whether the stored universe still matches "
                             "this config, and write nothing")
    parser.add_argument("--report", action="store_true",
                        help="explain the cut and exit, writing nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    if args.check:
        # Read-only, and deliberately BEFORE the write paths: this is the mode
        # you run to find out whether the stored universe is still the one this
        # config selects, without changing either.
        print_drift(cfg, conn)
        return
    if args.report:
        print_report(cfg, conn)
        print_drift(cfg, conn)
        return
    apply_filter(cfg, conn)
    print_report(cfg, conn)


if __name__ == "__main__":
    main()
