"""Negative sampling — what "nothing is coming" looks like.

A detector trained only on the hours before announcements learns nothing about
restraint. It needs quiet windows too: stretches where the stock traded
normally and no filing followed.

Two rules make a window genuinely quiet.

**The gap is measured from EVERY filing's t0, not just usable events'
acceptance time.** An immaterial 8-K is still an 8-K; a window sitting beside
one is not "nothing happening" (issue 28's clustering is resolved for free —
a quiet window cannot land inside an event's cluster, because the cluster's
filings are themselves filings). And per AGENTS.md rule 3, the instant that
matters is t0 = min(acceptance, matched news), not acceptance alone: a filing
that already has an `events` row uses that row's `t0_utc`; one that does not
(outside the study window, or not yet matched) falls back to its own
`acceptance_utc`, the same uncorrected baseline `t0.py` itself falls back to.

**A window with no volume baseline is excluded.** `features.py`'s
`volume_zscore()` needs `features.min_baseline_bars` prior bars before it is
defined at all — issue 32 found 234 positive windows entirely NaN on
volume_z for exactly this reason, unscoreable by any volume-based detector.
A detector cannot tell "quiet" from "unmeasurable", so an anchor without that
much history behind it is not offered as a candidate here either. (Nothing
yet excludes those 234 positive windows themselves — that is P5's call, not
this module's — but a quiet window claiming to be their negative counterpart
should not dodge the same standard.)

Quiet windows are built in the same shape as positives — 48 bars ending
strictly before an anchor timestamp — so nothing distinguishes the two except
the label. Any structural difference would be something a model could learn
instead of the market.

Usage:
  python -m src.pipeline.sampling --ratio 20        # availability report at this ratio
  python -m src.pipeline.sampling                   # same report, at the configured ratio
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from src import db
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

log = logging.getLogger(__name__)

HOUR_S = 3600


def quiet_candidates(cfg: dict, conn, ticker: str) -> np.ndarray:
    """Anchor timestamps for non-overlapping quiet windows on one ticker.

    An anchor plays the role a real t0 plays for a positive: the window is the
    `horizon_hours` bars ending strictly before it.

    Non-overlapping on purpose. Two windows sharing 47 of 48 bars are not two
    observations, and tiling rather than sliding keeps the sample from being
    dominated by near-duplicates of a single quiet afternoon.

    Anchors are confined to `study_window`, inclusive at both ends, exactly as
    `events.py` confines positives. The price snapshot deliberately runs past
    the window end (frozen 2026-08-30 against a 2026-08-01 end), so without
    this bound negatives could be drawn from a stretch of calendar no positive
    can ever occupy — handing a model "which month is this?" as a free
    separator instead of making it learn quiet-versus-about-to-file.
    """
    gap = cfg["sampling"]["quiet_gap_hours"] * HOUR_S
    horizon = cfg["decision"]["horizon_hours"]
    min_baseline = cfg["features"]["min_baseline_bars"]
    lo = date_str_to_ts(cfg["study_window"]["start"])
    hi = date_str_to_ts(cfg["study_window"]["end"])

    bars = np.array([r[0] for r in conn.execute(
        "SELECT ts_utc FROM bars WHERE ticker = ? AND interval = ? "
        "AND ts_utc BETWEEN ? AND ? ORDER BY ts_utc",
        (ticker, cfg["market"]["interval"], lo, hi))], dtype=np.int64)
    if bars.size <= horizon:
        return np.empty(0, dtype=np.int64)

    # AGENTS.md rule 3: the instant that matters is t0 = min(acceptance,
    # matched news), never acceptance alone. A filing already matched into
    # `events` (t0.py's job) contributes its stored `t0_utc`; one that is not
    # in `events` (outside the study window, or not yet built) falls back to
    # its own `acceptance_utc` — the same uncorrected baseline t0.py itself
    # falls back to when no news matched.
    rows = conn.execute(
        "SELECT f.acceptance_utc AS acceptance_utc, e.t0_utc AS t0_utc "
        "FROM filings f LEFT JOIN events e ON e.accession_no = f.accession_no "
        "WHERE f.ticker = ? AND f.acceptance_utc IS NOT NULL",
        (ticker,)).fetchall()
    filings = np.sort(np.array(
        [r["t0_utc"] if r["t0_utc"] is not None else r["acceptance_utc"]
         for r in rows], dtype=np.int64))

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
    # window straddling a filing is not a quiet window. It also needs
    # `min_baseline` bars of history behind THAT window: `volume_zscore()`
    # is undefined before then (min_periods=min_baseline_bars), so an anchor
    # any earlier would be "unmeasurable", not "quiet" — see the module
    # docstring's issue-32 note. `i` is the anchor's position in this
    # ticker's full bar history, matching the position `volume_zscore` counts
    # from, so `i - horizon` is the position of the window's earliest bar.
    anchors = []
    run = 0
    for i, ok in enumerate(quiet):
        run = run + 1 if ok else 0
        if run > horizon and i - horizon >= min_baseline:
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


def eval_decision_points(cfg: dict, conn) -> dict:
    """The EVALUATION population, at the true base rate. P4-12, decided 2026-09-01.

    Training and evaluation take deliberately different shapes, and conflating
    them is what made this task stall.

    TRAINING draws a balanced-ish sample: `negatives_per_positive` quiet
    windows per positive, tiled so no two negatives are near-duplicates. That
    is a sampling choice and it is allowed to distort the base rate, because
    the model needs enough positives to learn from.

    EVALUATION must not distort it. In live use the system sees EVERY trading
    hour for every covered company and must stay quiet through almost all of
    them, so the honest denominator is every in-universe bar:

        base rate = usable events / all in-universe bars
                  = 6,778 / 2,375,072 = 0.285%

    (Those figures were 6,737 / 2,584,872 = 0.26% when this was written, before
    the 2026-09-02 rebuild settled the positive count and before the
    denominator was bounded to the study window. The function has always
    computed them from the database; only this docstring went stale, which is
    the failure mode a hardcoded number in prose always has.)

    which is where the plan's "~0.3%, and always-quiet scores 99.7%" comes
    from. One positive per EVENT, not per pre-event hour: 48 positive hours
    per event would put the rate at 12.5% and always-quiet at 87.5%, which is
    not the number the plan is quoting.

    The rejected alternative was non-overlapping tiling of the quiet stretches
    too. Feasible, but it puts the base rate at 22.2% and then the alert
    budget covers most of the set, leaving precision-at-budget with almost
    nothing to discriminate.

    Storage was the original objection and it was based on a miscount: one
    window per bar materialised as 48 rows each is ~124M rows (~11 GB), but
    scoring per HOUR needs one row per bar — 2.58M rows, ~246 MB measured
    against the current matrix's bytes-per-row. A 48x difference, and the
    reason this framing is affordable after all.

    Returns the counts; materialising the frame is the evaluation harness's
    job in Phase 5, and it reads these definitions.
    """
    interval = cfg["market"]["interval"]
    lo = date_str_to_ts(cfg["study_window"]["start"])
    hi = date_str_to_ts(cfg["study_window"]["end"])

    total_bars = conn.execute(
        "SELECT COUNT(*) FROM bars WHERE interval = ? AND ts_utc BETWEEN ? AND ? "
        "AND ticker IN (SELECT ticker FROM companies WHERE in_universe = 1)",
        (interval, lo, hi)).fetchone()[0]
    positives = conn.execute(
        "SELECT COUNT(*) FROM events WHERE usable = 1").fetchone()[0]

    if not total_bars:
        raise SystemExit(
            "no in-universe bars in the study window — run the market "
            "collector and the liquidity filter before sizing the eval set."
        )
    return {
        "decision_points": total_bars,
        "positives": positives,
        "base_rate": positives / total_bars,
        "always_quiet_accuracy": 1 - positives / total_bars,
    }


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
    ratio = scfg["negatives_per_positive"] if ratio is None else ratio

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
            "rate_per_bar": positives / total_bars,
            "eval": eval_decision_points(cfg, conn)}


def print_report(cfg: dict, conn, ratio: int | None = None) -> None:
    r = report(cfg, conn, ratio)
    scfg = cfg["sampling"]
    print(f"\n=== Negative sampling ===")
    print(f"quiet gap        : {scfg['quiet_gap_hours']}h from EVERY filing "
          f"(not just usable events)")
    print(f"window length    : {r['horizon']} bars, ending strictly before "
          f"its anchor")
    ev = r["eval"]
    print(f"\n-- evaluation set (P4-12, true base rate) --")
    print(f"decision points  : {ev['decision_points']:,} in-universe bars")
    print(f"positives        : {ev['positives']:,}")
    print(f"base rate        : {ev['base_rate']:.3%}  "
          f"(always-quiet accuracy {ev['always_quiet_accuracy']:.1%})")
    print(f"\n-- training sample --")
    print(f"positives        : {r['positives']:,}")
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
    parser.add_argument("--ratio", type=int, help="negatives per positive "
                        "(default: sampling.negatives_per_positive)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    print_report(cfg, conn, args.ratio)


if __name__ == "__main__":
    main()
