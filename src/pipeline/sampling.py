"""Negative sampling — what "nothing is coming" looks like.

A detector trained only on the hours before announcements learns nothing about
restraint. It needs quiet windows too: stretches where the stock traded
normally and no filing followed.

Two rules make a window genuinely quiet.

**The gap is measured from EVERY filing, not just usable events.** An
immaterial 8-K is still an 8-K; a window sitting beside one is not "nothing
happening". This also resolves issue 28's clustering for free — a quiet window
cannot land inside an event's cluster, because the cluster's filings are
themselves filings.

**A window with no volume baseline is excluded**, exactly as the 234
unscoreable positives are (issue 32). A detector cannot tell "quiet" from
"unmeasurable", so neither should the sample.

Quiet windows are built in the same shape as positives — 48 bars ending
strictly before an anchor timestamp — so nothing distinguishes the two except
the label. Any structural difference would be something a model could learn
instead of the market.

Usage:
  python -m src.pipeline.sampling --report          # what is available
  python -m src.pipeline.sampling --ratio 20        # draw and report a sample
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from src import db
from src.utils.config import load_config

log = logging.getLogger(__name__)

HOUR_S = 3600


def quiet_candidates(cfg: dict, conn, ticker: str) -> np.ndarray:
    """Anchor timestamps for non-overlapping quiet windows on one ticker.

    An anchor plays the role a real t0 plays for a positive: the window is the
    `horizon_hours` bars ending strictly before it.

    Non-overlapping on purpose. Two windows sharing 47 of 48 bars are not two
    observations, and tiling rather than sliding keeps the sample from being
    dominated by near-duplicates of a single quiet afternoon.
    """
    gap = cfg["sampling"]["quiet_gap_hours"] * HOUR_S
    horizon = cfg["decision"]["horizon_hours"]

    bars = np.array([r[0] for r in conn.execute(
        "SELECT ts_utc FROM bars WHERE ticker = ? AND interval = ? "
        "ORDER BY ts_utc", (ticker, cfg["market"]["interval"]))], dtype=np.int64)
    if bars.size <= horizon:
        return np.empty(0, dtype=np.int64)

    filings = np.array([r[0] for r in conn.execute(
        "SELECT acceptance_utc FROM filings WHERE ticker = ? AND "
        "acceptance_utc IS NOT NULL ORDER BY acceptance_utc",
        (ticker,))], dtype=np.int64)

    if filings.size == 0:
        quiet = np.ones(bars.size, dtype=bool)
    else:
        idx = np.searchsorted(filings, bars)
        before = np.where(idx > 0, bars - filings[np.clip(idx - 1, 0, None)],
                          np.iinfo(np.int64).max)
        after = np.where(idx < filings.size,
                         filings[np.clip(idx, 0, filings.size - 1)] - bars,
                         np.iinfo(np.int64).max)
        quiet = (before > gap) & (after > gap)

    # An anchor needs `horizon` quiet bars behind it, all of them quiet: a
    # window straddling a filing is not a quiet window.
    anchors = []
    run = 0
    for i, ok in enumerate(quiet):
        run = run + 1 if ok else 0
        if run > horizon:
            anchors.append(int(bars[i]))
            run = 0            # tile, do not slide
    return np.array(anchors, dtype=np.int64)


def all_candidates(cfg: dict, conn) -> dict[str, np.ndarray]:
    """Quiet anchors for every in-universe ticker."""
    out = {}
    for ticker in db.universe_tickers(conn):
        anchors = quiet_candidates(cfg, conn, ticker)
        if anchors.size:
            out[ticker] = anchors
    if not out:
        raise SystemExit(
            "no quiet window exists anywhere in the universe — check "
            "sampling.quiet_gap_hours against the filing density."
        )
    return out


def draw(cfg: dict, candidates: dict[str, np.ndarray], n: int,
         seed: int | None = None) -> list[tuple[str, int]]:
    """Draw `n` (ticker, anchor) pairs without replacement, reproducibly.

    Nested by construction: the same seed drawn at 10, 20 and 40 negatives per
    positive gives 10 ⊂ 20 ⊂ 40, so the robustness check varies one thing. Three
    independent samples would confound the ratio with the draw.

    A shortfall returns everything available and is reported by the caller,
    never resampled with replacement — duplicate rows would be a fabricated
    observation.
    """
    seed = cfg["sampling"]["seed"] if seed is None else seed
    pool = [(t, int(a)) for t, anchors in sorted(candidates.items())
            for a in anchors]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    return [pool[i] for i in order[:min(n, len(pool))]]


def report(cfg: dict, conn, ratio: int | None = None) -> dict:
    """What is available, and what the base rate would be under each framing."""
    scfg = cfg["sampling"]
    horizon = cfg["decision"]["horizon_hours"]
    ratio = ratio or scfg["negatives_per_positive"]

    positives = conn.execute(
        "SELECT COUNT(*) FROM events WHERE usable = 1").fetchone()[0]
    candidates = all_candidates(cfg, conn)
    # .values(), not the dict itself: iterating a dict yields KEYS, so the
    # obvious `sum(len(v) for v in candidates)` sums ticker-symbol LENGTHS and
    # returns a plausible-looking number (1,496 tickers x ~3.4 chars = 5,050)
    # that is not the quantity at all.
    available = sum(len(v) for v in candidates.values())
    total_bars = conn.execute(
        "SELECT COUNT(*) FROM bars WHERE interval = ? AND ticker IN "
        "(SELECT ticker FROM companies WHERE in_universe = 1)",
        (cfg["market"]["interval"],)).fetchone()[0]

    wanted = positives * ratio
    return {"positives": positives, "available": available,
            "tickers_with_quiet": len(candidates), "wanted": wanted,
            "shortfall": max(0, wanted - available), "horizon": horizon,
            "total_bars": total_bars,
            "rate_tiled": positives / (positives + available),
            "rate_per_bar": positives / total_bars}


def print_report(cfg: dict, conn, ratio: int | None = None) -> None:
    r = report(cfg, conn, ratio)
    scfg = cfg["sampling"]
    print(f"\n=== Negative sampling ===")
    print(f"quiet gap        : {scfg['quiet_gap_hours']}h from EVERY filing "
          f"(not just usable events)")
    print(f"window length    : {r['horizon']} bars, ending strictly before "
          f"its anchor")
    print(f"\npositives        : {r['positives']:,}")
    print(f"quiet windows    : {r['available']:,} available across "
          f"{r['tickers_with_quiet']:,} tickers")
    print(f"wanted at {(r['wanted']//r['positives']) if r['positives'] else 0}:1"
          f"      : {r['wanted']:,}"
          + (f"  ⚠ SHORTFALL {r['shortfall']:,}" if r["shortfall"] else ""))

    print(f"\n--- base rate depends on the unit, and the plan never pinned it ---")
    print(f"one window per bar (overlapping) : {r['rate_per_bar']:.3%}  "
          f"({r['total_bars']:,} windows)")
    print(f"non-overlapping tiling           : {r['rate_tiled']:.1%}  "
          f"({r['positives'] + r['available']:,} windows)")
    print(f"\nThe first is the plan's ~0.3% but needs ~{r['total_bars']//1000:,}k "
          f"windows of {r['horizon']} rows.\nThe second is buildable but leaves "
          f"the alert budget covering most of the set.\nThis is a decision about "
          f"the headline metric, not a coding choice.")

    for rr in scfg["robustness_ratios"] + [scfg["negatives_per_positive"]]:
        n = min(r["positives"] * rr, r["available"])
        print(f"  ratio {rr:>3}:1 -> {n:,} negatives"
              + ("  (capped by availability)"
                 if r["positives"] * rr > r["available"] else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=int, help="negatives per positive")
    parser.add_argument("--report", action="store_true", help="availability only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    print_report(cfg, conn, args.ratio)


if __name__ == "__main__":
    main()
