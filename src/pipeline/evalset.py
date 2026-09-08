"""The evaluation population — every hour the system would actually face.

`sampling.eval_decision_points` fixed the *counts* for this frame in P4-12 and
said the frame itself was Phase 5's to build: *"materialising the frame is the
evaluation harness's job in Phase 5, and it reads these definitions."* This is
that module.

Training and evaluation take deliberately different shapes, and conflating
them is what made P4-12 stall and P5-03 stall again. Training draws a
balanced-ish 3:1 sample, which is allowed to distort the base rate because a
model needs positives to learn from. Evaluation must not distort it: in live
use the system sees every trading hour for every covered company and must stay
quiet through almost all of them.

Why the two sides are shaped differently
----------------------------------------
Decided by the user 2026-09-04, after a tuning run on a sampled frame came
back identical to the always-quiet floor.

* A **positive** is one episode: the `decision.horizon_hours` bars strictly
  before an event's t0, carrying one label. One positive per EVENT, never one
  per pre-event hour — 48 positive hours per event would put the base rate at
  12.5% and always-quiet at 87.5%, which is not the figure the plan quotes.
* A **negative** is a single bar. Every other in-universe hour is its own
  independent "do I alert now?" decision, because that is what the live system
  faces.

The asymmetry is the honest shape rather than an awkward compromise: an
episode is a thing that happens, a quiet hour is just an hour. It yields
~2.26M windows at a ~0.30% base rate, matching `eval_decision_points`, and —
the practical point — it puts the alert budget (~12k, denominated in
ticker-months) far BELOW the window count, so precision at that budget can
actually discriminate. On a 4,421-window sample the budget exceeded the
windows, bought every one of them, and collapsed precision to the base rate.

The rejected alternative was tiling the quiet stretches into 48-bar windows
too. Symmetric and simpler, but P4-12 measured it at a 19-22% base rate, where
the budget covers a third of the set and the metric has almost nothing left to
discriminate.

Usage:
  python -m src.pipeline.evalset --split val
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.pipeline.features import (_event_times, _news_arrays, _ticker_frame,
                                   ticker_features)
from src.pipeline.split import TEST, boundaries, split_of
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, ts_to_iso

#: Columns the contract needs, in the order the feature matrix carries them.
LABEL_COLS = ("window_id", "ticker", "t0_utc", "is_scheduled", "item_code")


#: The two t0 definitions the plan requires reporting side by side.
#: `t0_utc` is the corrected instant, min(acceptance, earliest matched news);
#: `t0_filing_utc` is the uncorrected acceptance time. Evaluating both answers
#: "what does the news correction actually buy?" — which is this project's
#: headline contribution, so it must be measurable, not asserted.
T0_COLUMNS = {"news_adjusted": "t0_utc", "filing": "t0_filing_utc"}


def _positive_windows(conn, cfg: dict, ticker: str, lo: int, hi: int,
                      t0_column: str = "t0_utc") -> list[dict]:
    """Usable events for one ticker whose t0 falls in [lo, hi).

    `t0_column` selects the variant. The chosen column is aliased to `t0_utc`
    so everything downstream — the contract, the metrics, this module — stays
    single-purpose, exactly as `contract.py` describes: "t0_utc means the
    variant currently being evaluated and the whole evaluation is run twice".
    """
    if t0_column not in set(T0_COLUMNS.values()):
        raise SystemExit(f"unknown t0 column {t0_column!r}; expected one of "
                         f"{sorted(set(T0_COLUMNS.values()))}")
    rows = conn.execute(
        f"SELECT event_id, items, {t0_column} AS t0_utc, is_scheduled FROM events "
        f"WHERE usable = 1 AND ticker = ? AND {t0_column} >= ? AND {t0_column} < ? "
        f"ORDER BY {t0_column}", (ticker, lo, hi)).fetchall()
    return [dict(r) for r in rows]


def build_eval_frame(cfg: dict, conn, lo: int, hi: int,
                     tickers: list[str] | None = None,
                     progress_every: int = 250,
                     t0_variant: str = "news_adjusted") -> pd.DataFrame:
    """Every decision point in [lo, hi), positives as episodes.

    Returns a frame in the feature matrix's shape, ready to hand to a
    `Baseline`. `t0_utc` is set on the bars of a positive episode and null
    everywhere else, which is the contract's definition of the label.

    `lo`/`hi` are UTC epoch seconds. Nothing here filters by split — pass the
    boundaries you mean, and note that `Baseline.predict` refuses the sealed
    test range independently.

    `t0_variant` picks which t0 definition labels the positives — see
    `T0_COLUMNS`. The whole evaluation is run once per variant, which is how
    the plan requires the news correction to be reported.
    """
    if t0_variant not in T0_COLUMNS:
        raise SystemExit(f"unknown t0_variant {t0_variant!r}; expected one of "
                         f"{sorted(T0_COLUMNS)}")
    t0_column = T0_COLUMNS[t0_variant]
    horizon = cfg["decision"]["horizon_hours"]
    interval = cfg["market"]["interval"]
    benchmark = _ticker_frame(conn, cfg["market"]["benchmark"], interval)

    if tickers is None:
        tickers = [r[0] for r in conn.execute(
            "SELECT ticker FROM companies WHERE in_universe = 1 ORDER BY ticker")]
    if not tickers:
        raise SystemExit(
            "no in-universe companies — run the liquidity filter (P3-02) "
            "before building an evaluation frame.")

    out: list[pd.DataFrame] = []
    n_pos = n_neg = 0
    for i, ticker in enumerate(tickers, 1):
        frame = _ticker_frame(conn, ticker, interval)
        if frame.empty:
            continue
        filings, earnings = _event_times(conn, cfg, ticker)
        articles, publishers = _news_arrays(conn, cfg, ticker)
        feats = ticker_features(frame, benchmark, filings, earnings, cfg,
                                article_times=articles,
                                publishers=publishers)
        stamps = feats.index.to_numpy()

        # Positive episodes first, so their bars can be excluded from the
        # negative pool. A bar cannot be both its own decision point and part
        # of an episode — it would be counted twice by every metric.
        claimed = np.zeros(len(stamps), dtype=bool)
        for event in _positive_windows(conn, cfg, ticker, lo, hi, t0_column):
            end = np.searchsorted(stamps, event["t0_utc"], side="left")
            start = max(end - horizon, 0)
            if end <= start:
                continue
            claimed[start:end] = True
            block = feats.iloc[start:end].copy()
            block.insert(0, "item_code", event["items"])
            block.insert(0, "is_scheduled", bool(event["is_scheduled"]))
            block.insert(0, "t0_utc", event["t0_utc"])
            block.insert(0, "ticker", ticker)
            block.insert(0, "window_id", event["event_id"])
            out.append(block.reset_index())
            n_pos += 1

        # Every remaining bar in range is its own single-row window.
        in_range = (stamps >= lo) & (stamps < hi)
        free = in_range & ~claimed
        if free.any():
            block = feats.loc[free].copy()
            block.insert(0, "item_code", None)
            block.insert(0, "is_scheduled", None)
            block.insert(0, "t0_utc", None)
            block.insert(0, "ticker", ticker)
            block.insert(0, "window_id",
                         [f"bar:{ticker}:{s}" for s in stamps[free]])
            out.append(block.reset_index())
            n_neg += int(free.sum())

        if progress_every and i % progress_every == 0:
            print(f"  evalset: {i}/{len(tickers)} tickers, "
                  f"{n_pos:,} episodes + {n_neg:,} bars")

    if not out:
        raise SystemExit(
            f"evaluation frame is empty for {ts_to_iso(lo)}..{ts_to_iso(hi)} — "
            f"check bar coverage and the split boundaries.")
    matrix = pd.concat(out, ignore_index=True)
    print(f"evalset: {n_pos:,} positive episodes + {n_neg:,} quiet bars "
          f"= {n_pos + n_neg:,} windows, base rate "
          f"{n_pos / max(n_pos + n_neg, 1):.4%}")
    return matrix


def split_bounds(cfg: dict, split: str, conn=None) -> tuple[int, int]:
    """[lo, hi) for 'train', 'val' or 'test'.

    Asking for the TEST bounds while the seal is on is refused here, when
    `conn` is supplied.

    The seal used to guard only TRAINING — `rl/train.py`, `base.py` and
    `gradient_boosting.py` all call `assert_not_test` on the frame they fit
    on. That stops a model being trained on the test set, which is the worse
    leak, but it left the EVALUATION path open: `compare --split test` built
    the test frame and would have scored on it without ever consulting the
    seal. Since evaluating once is precisely what the seal exists to ration,
    the guard belongs at the point the bounds are handed out.

    `conn` is optional only because several callers legitimately want the
    boundaries without touching data — reporting the split sizes, drawing the
    calendar. Those pass nothing and are unaffected. Anything that is about to
    READ rows passes the connection, and then the seal decides.
    """
    lo = date_str_to_ts(cfg["study_window"]["start"])
    hi = date_str_to_ts(cfg["study_window"]["end"])
    train_end, val_end = boundaries(cfg)
    bounds = {"train": (lo, train_end), "val": (train_end, val_end),
              TEST: (val_end, hi)}
    if split not in bounds:
        raise SystemExit(f"unknown split {split!r}; expected one of "
                         f"{sorted(bounds)}")
    if split == TEST and conn is not None:
        from src.pipeline.split import assert_not_test

        # The midpoint stands for the period: `assert_not_test` reports the
        # boundary and the offending instant, which is the message a caller
        # needs, and any timestamp inside the range proves the same point.
        assert_not_test(cfg, conn, (bounds[TEST][0] + bounds[TEST][1]) // 2,
                        "evaluation on the test split")
    return bounds[split]


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val", help="train | val | test")
    ap.add_argument("--limit-tickers", type=int, default=None,
                    help="build only the first N tickers (a smoke run)")
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    lo, hi = split_bounds(cfg, args.split)
    print(f"{args.split}: {ts_to_iso(lo)} .. {ts_to_iso(hi)}")

    tickers = None
    if args.limit_tickers:
        tickers = [r[0] for r in conn.execute(
            "SELECT ticker FROM companies WHERE in_universe = 1 "
            "ORDER BY ticker LIMIT ?", (args.limit_tickers,))]

    frame = build_eval_frame(cfg, conn, lo, hi, tickers)
    print(f"rows: {len(frame):,}  windows: {frame.window_id.nunique():,}")


if __name__ == "__main__":
    main()
