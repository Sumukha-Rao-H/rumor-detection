"""The number to beat — all four baselines, one evaluation path, one table.

The capstone of Phase 5 and the reason it was built before Phase 6. Without a
tuned baseline, a learned policy that scores 0.09 means nothing: it could be
excellent or it could be losing to a threshold on one column. With this table
it is a comparison.

Every baseline reaches the metrics through the identical route — the P5-01
`predict()` path, the same evaluation population, one alert budget, one
threshold per frame — so any difference between two rows is a difference
between the detectors and not between their harnesses.

Validation only. The test split stays sealed until Phase 10, and
`Baseline.predict` refuses it independently of anything decided here.

Both t0 variants are reported. `t0_utc` is the corrected instant,
min(acceptance, earliest matched news); `t0_filing_utc` is the uncorrected
acceptance time. The gap between the two tables is what the news correction
actually buys, which is the project's headline contribution and therefore has
to be measured rather than asserted.

Usage:
  python -m src.baselines.compare                      # validation, both variants
  python -m src.baselines.compare --variant news_adjusted
  python -m src.baselines.compare --skip-gb            # fast: no model to fit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.baselines.always_quiet import AlwaysQuiet
from src.baselines.random_noise import RandomNoise
from src.baselines.cusum import CUSUM
from src.baselines.gradient_boosting import GradientBoosting
from src.baselines.volume_zscore import VolumeZScore
from src.eval.report import evaluate
from src.eval.metrics import precision_at_alert_budget
from src.pipeline.evalset import T0_COLUMNS, build_eval_frame, split_bounds
from src.utils.config import load_config

#: The columns worth putting in front of a reader, in reading order.
REPORT_COLUMNS = [
    "baseline", "slice", "t0_variant", "n_windows", "n_positive", "base_rate",
    "precision", "max_precision", "lift", "recall", "n_alerts",
    "median_lead_trading_h", "degenerate",
]

CONTRACT_DTYPES = {
    "window_id": "string", "ticker": "string", "ts_utc": "Int64",
    "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string",
}


def conform(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast the label/identifier columns to the contract's dtypes."""
    out = frame.copy()
    for col, dtype in CONTRACT_DTYPES.items():
        out[col] = out[col].astype(dtype)
    return out


def build_training_frame(cfg: dict, conn) -> pd.DataFrame:
    """Positives plus 3:1 sampled quiet windows, from the TRAIN split only.

    Only gradient boosting needs this. `negatives_per_positive` is labelled
    "training only" in config for the reason P4-12 gives: fitted at the true
    0.46% base rate a model sees ~216 quiet rows per positive one and can reach
    99.5% accuracy by answering "quiet" forever.
    """
    from src.pipeline import sampling
    from src.pipeline.features import build_quiet_matrix

    lo, hi = split_bounds(cfg, "train")
    # `matrix_path` carries the ablation arm, so this reads the matrix that
    # THIS config built — the with-news arm cannot be scored against the
    # without-news matrix by forgetting to change a second path.
    from src.pipeline.features import matrix_path

    matrix = pd.read_parquet(matrix_path(cfg))
    positives = matrix[(matrix.ts_utc >= lo) & (matrix.ts_utc < hi)].copy()

    ratio = cfg["sampling"]["negatives_per_positive"]
    candidates = sampling.all_candidates(cfg, conn)
    in_split = {t: a[(a >= lo) & (a < hi)] for t, a in candidates.items()}
    in_split = {t: a for t, a in in_split.items() if len(a)}
    wanted = positives.window_id.nunique() * ratio
    pairs = sampling.draw(cfg, in_split, wanted)
    if len(pairs) < wanted:
        # Reported, never resampled with replacement — duplicate rows would be
        # a fabricated observation.
        print(f"  training draw short: {len(pairs):,} of {wanted:,} wanted")

    quiet = build_quiet_matrix(cfg, conn, pairs)
    return conform(pd.concat([positives, quiet], ignore_index=True))


def run_baselines(cfg: dict, conn, frame: pd.DataFrame,
                  skip_gb: bool = False,
                  fitted_gb: "GradientBoosting | None" = None,
                  policy_runs: list[str] | None = None
                  ) -> dict[str, pd.DataFrame]:
    """Every detector's prediction frame, from one evaluation frame.

    Each is scored through `Baseline.predict`, so all of them get the same
    contract validation, the same unscoreable handling and the same seal check.

    `policy_runs` are P6-03 run directories. Each becomes its **own row**
    rather than being averaged into one, because P6-05's finding is that the
    policy's score is dominated by its seed — a mean would hide exactly the
    thing the reader needs to see.
    """
    # RandomNoise is not decoration. It is the null: on a frame whose two
    # classes get the same number of chances it must score the always-quiet
    # floor, and if it ever reports meaningfully more, the frame has
    # developed a length asymmetry again and no other row here means
    # anything until that is explained. Cheap to carry, and it is the one
    # row a sceptical reader can check without trusting any of the others.
    models: list = [AlwaysQuiet(cfg), RandomNoise(cfg), VolumeZScore(cfg),
                    CUSUM(cfg)]
    if not skip_gb:
        # The training frame does not depend on which t0 variant labels the
        # EVALUATION set, so a caller sweeping variants fits once and passes
        # the model back in rather than paying for it twice.
        gb = fitted_gb
        if gb is None:
            gb = GradientBoosting(cfg)
            print("  fitting gradient boosting on the train split...")
            gb.fit(build_training_frame(cfg, conn), conn=conn)
        models.append(gb)

    named: list[tuple[str, object]] = [(m.name, m) for m in models]
    for run in (policy_runs or []):
        from src.rl import load_policy
        label = f"rl_policy[{Path(run).name.split('-')[-1]}]"
        named.append((label, load_policy(cfg, run)))

    out = {}
    for label, model in named:
        # A finite placeholder: precision at the budget is rank-based, and
        # `evaluate` re-derives the operating point from the frame anyway.
        out[label] = model.predict(frame, threshold=float("inf"),
                                   conn=conn, context=f"compare/{label}")
    return out


def comparison_table(cfg: dict, predictions: dict[str, pd.DataFrame],
                     variant: str) -> pd.DataFrame:
    """One row per (baseline x slice), sharing one operating point per baseline.

    Slices come from `config.eval.split_by` — every number split scheduled vs
    unscheduled, never pooled, per the code standards' fourth rule.
    """
    from src.eval.report import slice_frames

    floor = None
    rows = []
    for name, frame in predictions.items():
        # One threshold per baseline, chosen on the whole frame, then applied
        # to every slice — so slices are comparable to each other rather than
        # each getting its own flattering cut.
        threshold = precision_at_alert_budget(frame).threshold
        for slice_name, sliced in slice_frames(frame).items():
            if sliced.empty:
                continue
            row = {"baseline": name, "slice": slice_name,
                   "t0_variant": variant,
                   **evaluate(sliced, threshold=threshold)}
            rows.append(row)
        if name == AlwaysQuiet(cfg).name:
            floor = next(r for r in rows
                         if r["baseline"] == name and r["slice"] == "all")

    table = pd.DataFrame(rows)
    # Lift is precision over the do-nothing floor for the SAME slice, which is
    # the only reading that survives slices with different base rates.
    if floor is not None:
        base = {(r["slice"]): r["precision"] for r in rows
                if r["baseline"] == floor["baseline"]}
        table["lift"] = [
            (p / base[s]) if base.get(s) else float("nan")
            for p, s in zip(table["precision"], table["slice"])]
    else:
        table["lift"] = float("nan")

    cols = [c for c in REPORT_COLUMNS if c in table.columns]
    return table[cols + [c for c in table.columns if c not in cols]]


def render(table: pd.DataFrame, slice_name: str = "all") -> str:
    """The headline view: one row per baseline, on one slice."""
    view = table[table["slice"] == slice_name].copy()
    view = view.sort_values("precision", ascending=False)
    cols = ["baseline", "t0_variant", "precision", "max_precision", "lift",
            "recall", "median_lead_trading_h", "n_alerts", "degenerate"]
    return view[cols].to_string(index=False,
                                float_format=lambda v: f"{v:.5f}")


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val",
                    help="train | val | test. Defaults to val; test is sealed.")
    ap.add_argument("--variant", default=None,
                    help=f"one of {sorted(T0_COLUMNS)}; default: both")
    ap.add_argument("--skip-gb", action="store_true",
                    help="omit gradient boosting (skips fitting a model)")
    ap.add_argument("--out", default=None, help="write the full table as CSV")
    ap.add_argument("--policy-run", action="append", default=None,
                    dest="policy_runs",
                    help="a P6-03 run directory; repeatable. Each seed gets "
                         "its own row — P6-05 found the policy's score is "
                         "dominated by the seed, and a mean would hide it.")
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    lo, hi = split_bounds(cfg, args.split, conn=conn)
    variants = [args.variant] if args.variant else list(T0_COLUMNS)

    started = time.time()
    tables, gb = [], None
    for variant in variants:
        print(f"\n=== t0 variant: {variant} ({args.split}) ===")
        frame = conform(build_eval_frame(cfg, conn, lo, hi, t0_variant=variant))
        if gb is None and not args.skip_gb:
            gb = GradientBoosting(cfg)
            print("  fitting gradient boosting on the train split...")
            gb.fit(build_training_frame(cfg, conn), conn=conn)
        predictions = run_baselines(cfg, conn, frame, skip_gb=args.skip_gb,
                                    fitted_gb=gb, policy_runs=args.policy_runs)
        table = comparison_table(cfg, predictions, variant)
        tables.append(table)
        print(render(table))

    full = pd.concat(tables, ignore_index=True)
    if args.out:
        full.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")

    if len(variants) > 1:
        print("\n=== what the news correction buys (slice: all) ===")
        head = full[full["slice"] == "all"]
        pivot = head.pivot(index="baseline", columns="t0_variant",
                           values=["precision", "median_lead_trading_h"])
        print(pivot.to_string(float_format=lambda v: f"{v:.5f}"))

    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
