"""Temporal split, and sealing the test set. P4-14.

The split is by CALENDAR, not by event count: the study window is cut at
`split.train` and `split.train + split.val` of its elapsed span, and whatever
events happen to fall in each period are that period's events. Choosing the
boundary from the event distribution instead — "cut where 85% of events have
happened" — would let the labels decide the split, which is a small
look-ahead and exactly the kind that is easy to defend and impossible to
un-ring later. The event counts per period are reported, never optimised.

Temporal and not random, because the alternative leaks. Two hours from the
same pre-announcement window landing on opposite sides of a random split
would let a model see part of an event it is then scored on, and 4,945 of
16,842 events share a ticker with another event within 30 days, so even
splitting by event would leak through the ticker's own price history.

**The test period is sealed.** `assert_not_test` raises on any timestamp at or
after the test boundary, and every read path that could reach evaluation data
should call it. The plan opens the seal once, in Phase 10, on the single final
evaluation run. That is what makes an accidental peek an error rather than a
mistake nobody notices: the guard fails loudly, and unsealing is a deliberate
act recorded in `meta` with a reason.

Usage:
  python -m src.pipeline.split --report     # boundaries and what falls where
  python -m src.pipeline.split --seal       # compute, store, and seal
  python -m src.pipeline.split --unseal "final evaluation, Phase 10"
"""

from __future__ import annotations

import argparse
import logging

from src import db
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, ts_to_iso, utc_now_ts

log = logging.getLogger(__name__)

#: Boundary timestamps and seal state, stamped so any later reader — a
#: notebook, a baseline, a future session — resolves the same split without
#: recomputing it from a config that may since have changed.
META_TRAIN_END = "split:train_end_utc"
META_VAL_END = "split:val_end_utc"
META_SEALED = "split:test_sealed"
META_SEALED_AT = "split:sealed_at_utc"
META_UNSEAL_REASON = "split:unseal_reason"

TRAIN, VAL, TEST = "train", "val", "test"
#: Anything after the study window. Not a split — the study never covered it.
#: Introduced 2026-09-04 for P7-01: `split_of` used to return TEST for every
#: timestamp at or after `val_end`, unbounded above, so today's bars were
#: classified as sealed test data and `assert_not_test` refused them. That
#: would have blocked the live monitor entirely. `boundaries()` already
#: documented the test period as bounded — "test is [val_end, end]" — so this
#: makes the code agree with its own docstring. The sealed period itself is
#: unchanged: every timestamp inside [val_end, study_end) is still refused.
LIVE = "live"


def boundaries(cfg: dict) -> tuple[int, int]:
    """(train_end, val_end) as UTC epoch seconds, from the config fractions.

    Half-open throughout: train is [start, train_end), val is
    [train_end, val_end), test is [val_end, end]. A bar exactly on a boundary
    belongs to the LATER period, so no hour is in two splits and none is in
    none.
    """
    scfg = cfg["split"]
    lo = date_str_to_ts(cfg["study_window"]["start"])
    hi = date_str_to_ts(cfg["study_window"]["end"])
    span = hi - lo
    if span <= 0:
        raise SystemExit(
            f"study_window is empty or inverted: {cfg['study_window']['start']} "
            f"-> {cfg['study_window']['end']}"
        )
    total = scfg["train"] + scfg["val"] + scfg["test"]
    if abs(total - 1.0) > 1e-9:
        raise SystemExit(
            f"split fractions must sum to 1.0, got {total:.4f} "
            f"(train={scfg['train']}, val={scfg['val']}, test={scfg['test']}). "
            f"A split that does not partition the window silently drops or "
            f"double-counts a stretch of calendar."
        )
    train_end = lo + int(span * scfg["train"])
    val_end = lo + int(span * (scfg["train"] + scfg["val"]))
    return train_end, val_end


def split_of(cfg: dict, ts_utc: int) -> str:
    """Which split one timestamp belongs to.

    Returns `LIVE` for anything at or after the study window's end. That data
    was never part of the study, so calling it TEST would both misdescribe it
    and — through `assert_not_test` — refuse the live monitor its own inputs.
    """
    train_end, val_end = boundaries(cfg)
    if ts_utc < train_end:
        return TRAIN
    if ts_utc < val_end:
        return VAL
    return TEST if ts_utc < date_str_to_ts(cfg["study_window"]["end"]) else LIVE


def is_sealed(conn) -> bool:
    """Is the test period currently closed? Absent meta counts as sealed.

    Fail-closed on purpose: a database that has never been sealed, or whose
    meta was lost, must not silently behave as though the test set were open.
    """
    return db.get_meta(conn, META_SEALED) != "0"


def assert_not_test(cfg: dict, conn, timestamps, context: str = "") -> None:
    """Raise if any timestamp falls in the sealed test period.

    The guard P4-14 exists for. Call it from anything that loads data for
    training, tuning, baseline comparison or feature inspection — anywhere a
    stray test row would quietly become part of a decision.

    Accepts one timestamp or any iterable of them.
    """
    if not is_sealed(conn):
        return
    _, val_end = boundaries(cfg)
    # Bounded ABOVE by the study window's end. Data after it was never in any
    # split, so refusing it would block Phase 7's live monitor while protecting
    # nothing — the sealed period is [val_end, study_end), exactly what
    # `boundaries()` documents.
    study_end = date_str_to_ts(cfg["study_window"]["end"])

    def sealed(t: int) -> bool:
        return val_end <= t < study_end

    try:
        offenders = [int(t) for t in timestamps if sealed(int(t))]
    except TypeError:                       # a single timestamp
        offenders = [int(timestamps)] if sealed(int(timestamps)) else []
    if offenders:
        where = f" in {context}" if context else ""
        raise SystemExit(
            f"SEALED TEST SET touched{where}: {len(offenders):,} timestamp(s) "
            f"at or after {ts_to_iso(val_end)}, earliest "
            f"{ts_to_iso(min(offenders))}.\n"
            f"The last {cfg['split']['test']:.0%} by date is sealed until the "
            f"final evaluation (plan Phase 10). Filter to "
            f"ts_utc < {val_end} for train/val work, or, if this really is "
            f"that final run, unseal deliberately:\n"
            f"  python -m src.pipeline.split --unseal \"<why>\""
        )


def counts(cfg: dict, conn) -> dict:
    """Events and decision points per split — reported, never optimised."""
    train_end, val_end = boundaries(cfg)
    lo = date_str_to_ts(cfg["study_window"]["start"])
    hi = date_str_to_ts(cfg["study_window"]["end"])
    iv = cfg["market"]["interval"]

    def events_between(a, b):
        return conn.execute(
            "SELECT COUNT(*) FROM events WHERE usable = 1 AND t0_utc >= ? "
            "AND t0_utc < ?", (a, b)).fetchone()[0]

    def bars_between(a, b):
        return conn.execute(
            "SELECT COUNT(*) FROM bars WHERE interval = ? AND ts_utc >= ? "
            "AND ts_utc < ? AND ticker IN "
            "(SELECT ticker FROM companies WHERE in_universe = 1)",
            (iv, a, b)).fetchone()[0]

    out = {}
    for name, a, b in ((TRAIN, lo, train_end), (VAL, train_end, val_end),
                       (TEST, val_end, hi + 1)):
        ev, bars = events_between(a, b), bars_between(a, b)
        out[name] = {"start": a, "end": b, "events": ev, "bars": bars,
                     "base_rate": (ev / bars) if bars else float("nan")}
    return out


def seal(cfg: dict, conn) -> dict:
    """Compute the boundaries, store them, and close the test period."""
    train_end, val_end = boundaries(cfg)
    now = utc_now_ts()
    db.set_meta(conn, META_TRAIN_END, str(train_end), now)
    db.set_meta(conn, META_VAL_END, str(val_end), now)
    db.set_meta(conn, META_SEALED, "1", now)
    db.set_meta(conn, META_SEALED_AT, str(now), now)
    log.info("split sealed: train < %s <= val < %s <= test",
             ts_to_iso(train_end), ts_to_iso(val_end))
    return counts(cfg, conn)


def unseal(cfg: dict, conn, reason: str) -> None:
    """Open the test period. Deliberate, and recorded with a reason."""
    if not reason.strip():
        raise SystemExit(
            "unsealing needs a reason — it is recorded in `meta` so the final "
            "evaluation can be shown to have happened once, on purpose."
        )
    now = utc_now_ts()
    db.set_meta(conn, META_SEALED, "0", now)
    db.set_meta(conn, META_UNSEAL_REASON, reason.strip(), now)
    log.warning("TEST SET UNSEALED: %s", reason.strip())


def print_report(cfg: dict, conn) -> None:
    train_end, val_end = boundaries(cfg)
    c = counts(cfg, conn)
    sealed = is_sealed(conn)
    print("\n=== Temporal split (P4-14) ===")
    print(f"method           : {cfg['split']['method']}, by calendar")
    print(f"train ends       : {ts_to_iso(train_end)}")
    print(f"val ends         : {ts_to_iso(val_end)}")
    print(f"test set         : {'SEALED' if sealed else '*** UNSEALED ***'}")
    print(f"\n{'split':<8}{'events':>10}{'bars':>14}{'base rate':>12}")
    for name in (TRAIN, VAL, TEST):
        s = c[name]
        print(f"{name:<8}{s['events']:>10,}{s['bars']:>14,}"
              f"{s['base_rate']:>11.3%}")
    total_ev = sum(c[n]["events"] for n in (TRAIN, VAL, TEST))
    print(f"{'total':<8}{total_ev:>10,}")
    if not sealed:
        print(f"\n  reason: {db.get_meta(conn, META_UNSEAL_REASON)}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Temporal split and test-set seal.")
    ap.add_argument("--seal", action="store_true",
                    help="compute boundaries, store them, close the test set")
    ap.add_argument("--unseal", metavar="REASON",
                    help="open the test set, recording why (Phase 10 only)")
    ap.add_argument("--report", action="store_true",
                    help="show boundaries and what falls in each split")
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    if args.unseal:
        unseal(cfg, conn, args.unseal)
    elif args.seal:
        seal(cfg, conn)
    print_report(cfg, conn)
    conn.close()


if __name__ == "__main__":
    main()
