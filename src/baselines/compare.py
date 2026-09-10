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
    # Plan §6: the action distribution beside the score, every time. These two
    # are the detector's OWN rule, not the budget's — `pct_windows_alerted`
    # above is derived at the alert budget and so describes the harness (it
    # came out 0.0189 for every row of the Phase 10 table). NaN for a detector
    # that has no rule of its own; see `Baseline.own_action_distribution`.
    "policy_flag_rate", "policy_pct_windows_alerted",
]

#: The slices every printed view walks, in reading order. AGENTS.md rule 7 —
#: "every number split scheduled vs unscheduled" — makes the last two
#: mandatory; `all` is kept first because it is the row a reader orients on.
SLICES = ("all", "scheduled", "unscheduled")

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


def build_training_frame(cfg: dict, conn,
                         ratio: int | None = None) -> pd.DataFrame:
    """Positives plus sampled quiet windows, from the TRAIN split only.

    Only gradient boosting needs this. `negatives_per_positive` is labelled
    "training only" in config for the reason P4-12 gives: fitted at the true
    0.46% base rate a model sees ~216 quiet rows per positive one and can reach
    99.5% accuracy by answering "quiet" forever.

    `ratio` overrides `sampling.negatives_per_positive` for one call, which is
    what makes plan §8's robustness check runnable: *"decide the sampling rule
    in week 1, write it in config, and report results at two ratios so nobody
    can accuse you of tuning it."* Until this argument existed,
    `sampling.robustness_ratios` was consulted in exactly one place — a print
    of how many negatives are AVAILABLE at each ratio — which is an
    availability census, not a robustness check, under a name that promises
    one. `draw` is seeded and nested (1:1 is a subset of 2:1 is a subset of
    3:1), so the three frames are genuinely the same experiment at three
    depths rather than three unrelated samples.
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

    ratio = cfg["sampling"]["negatives_per_positive"] if ratio is None else int(ratio)
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


def noise_seeds(cfg: dict) -> list[int]:
    """The seeds the null is drawn at — a spread, not a single point.

    One draw is not an error bar. On the Phase 10 shape a null draw has
    E[TP] ~ 19.1 with sd ~ 4.4, so sampling noise alone puts one row anywhere
    between roughly 0.55x and 1.45x lift; a lone row reading 1.4x cannot be
    told apart from a real residual asymmetry of that size. Falls back to the
    single configured `seed` so a config without the list still produces the
    null rather than dropping it.
    """
    noise = cfg.get("baselines", {}).get("random_noise", {})
    seeds = noise.get("seeds") or [noise.get("seed", 0)]
    return [int(s) for s in seeds]


def run_baselines(cfg: dict, conn, frame: pd.DataFrame,
                  skip_gb: bool = False,
                  fitted_gb: "GradientBoosting | None" = None,
                  policy_runs: list[str] | None = None,
                  sampling_ratio: int | None = None
                  ) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Every detector's prediction frame, from one evaluation frame.

    Each is scored through `Baseline.predict`, so all of them get the same
    contract validation, the same unscoreable handling and the same seal check.

    Returns the predictions **and the models that produced them**, keyed by the
    same label. `comparison_table` needs the models, not only their output, to
    ask each one what it would do on its own decision rule — plan §6's
    degeneracy check, which no column derived at the alert budget can answer.

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
    #
    # One row per seed, named the way the per-seed policy rows already are.
    # The null's own docstring says its seed "should not matter"; at this
    # sample size that is a claim with a measurable width, so the table shows
    # the width instead of asserting it. Each extra row is one score vector —
    # this baseline reads no features.
    models: list = [AlwaysQuiet(cfg)]
    models += [RandomNoise(cfg, seed=s) for s in noise_seeds(cfg)]
    models += [VolumeZScore(cfg), CUSUM(cfg)]
    if not skip_gb:
        # The training frame does not depend on which t0 variant labels the
        # EVALUATION set, so a caller sweeping variants fits once and passes
        # the model back in rather than paying for it twice.
        gb = fitted_gb
        if gb is None:
            gb = GradientBoosting(cfg)
            ratio = sampling_ratio or cfg["sampling"]["negatives_per_positive"]
            print(f"  fitting gradient boosting on the train split "
                  f"({ratio}:1 negatives)...")
            gb.fit(build_training_frame(cfg, conn, ratio=sampling_ratio),
                   conn=conn)
        models.append(gb)

    named: list[tuple[str, object]] = [(m.name, m) for m in models]
    for run in (policy_runs or []):
        from src.rl import load_policy
        label = f"rl_policy[{Path(run).name.split('-')[-1]}]"
        named.append((label, load_policy(cfg, run)))

    out, used = {}, {}
    for label, model in named:
        # A finite placeholder: precision at the budget is rank-based, and
        # `evaluate` re-derives the operating point from the frame anyway.
        out[label] = model.predict(frame, threshold=float("inf"),
                                   conn=conn, context=f"compare/{label}")
        used[label] = model
    return out, used


#: What a detector with no decision rule of its own answers for the plan §6
#: columns. Used when a table is built without its models — the tests do that,
#: and a table cannot invent a rule it was never handed.
_NO_OWN_RULE = {"flag_rate": float("nan"), "pct_windows_alerted": float("nan")}


def comparison_table(cfg: dict, predictions: dict[str, pd.DataFrame],
                     variant: str,
                     models: dict[str, object] | None = None) -> pd.DataFrame:
    """One row per (baseline x slice), sharing one operating point per baseline.

    Slices come from `config.eval.split_by` — every number split scheduled vs
    unscheduled, never pooled, per the code standards' fourth rule.

    `models` is what `run_baselines` returns beside the predictions. Given it,
    every row also carries what that detector would do on its OWN rule rather
    than at the alert budget, which is plan §6's degeneracy check and the one
    thing the budgeted columns structurally cannot answer.
    """
    from src.eval.report import slice_frames

    floor = None
    rows = []
    for name, frame in predictions.items():
        # One threshold per baseline, chosen on the whole frame, then applied
        # to every slice — so slices are comparable to each other rather than
        # each getting its own flattering cut.
        threshold = precision_at_alert_budget(frame).threshold
        model = (models or {}).get(name)
        for slice_name, sliced in slice_frames(frame).items():
            if sliced.empty:
                continue
            own = (model.own_action_distribution(sliced) if model is not None
                   else _NO_OWN_RULE)
            row = {"baseline": name, "slice": slice_name,
                   "t0_variant": variant,
                   **evaluate(sliced, threshold=threshold),
                   "policy_flag_rate": own["flag_rate"],
                   "policy_pct_windows_alerted": own["pct_windows_alerted"]}
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
    """The headline view: one row per baseline, on ONE slice.

    One slice, so callers have to say which. `main` prints all three in turn —
    AGENTS.md rule 7 wants every headline number split scheduled vs
    unscheduled, and for a long time the run log this reads out to carried the
    pooled rows only. Pooling is not neutral: gradient boosting scores 0.00475
    scheduled against 0.01642 unscheduled and `rl_policy[seed43]` inverts that
    ordering, so the pooled row hides the finding rather than summarising it.
    """
    view = table[table["slice"] == slice_name].copy()
    view = view.sort_values("precision", ascending=False)
    cols = ["baseline", "t0_variant", "precision", "max_precision", "lift",
            "recall", "median_lead_trading_h", "n_alerts", "degenerate",
            # Plan §6, beside the score rather than in a separate report.
            "policy_flag_rate"]
    cols = [c for c in cols if c in view.columns]
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
    ap.add_argument("--sampling-ratio", type=int, default=None,
                    help="negatives per positive for the gradient-boosting "
                         "TRAINING frame, overriding "
                         "sampling.negatives_per_positive. Plan §8 asks for "
                         "results at sampling.robustness_ratios so nobody can "
                         "accuse the ratio of being tuned; run this once per "
                         "ratio and compare. Affects gradient boosting only — "
                         "the other baselines never see a training frame.")
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
        predictions, models = run_baselines(
            cfg, conn, frame, skip_gb=args.skip_gb, fitted_gb=gb,
            policy_runs=args.policy_runs,
            sampling_ratio=args.sampling_ratio)
        table = comparison_table(cfg, predictions, variant, models=models)
        tables.append(table)
        # All three slices, not just the pooled one. AGENTS.md rule 7: every
        # headline number is split scheduled vs unscheduled. This print is what
        # the run log records and therefore what a report would be written
        # from, so printing only `all` here made the log itself violate the
        # rule however carefully `comparison_table` split the rows.
        for slice_name in SLICES:
            print(f"\n-- slice: {slice_name} --")
            if not (table["slice"] == slice_name).any():
                print("  (no rows in this slice)")
                continue
            print(render(table, slice_name))

    full = pd.concat(tables, ignore_index=True)
    if args.out:
        full.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")

    if len(variants) > 1:
        # Split here too, and for the same reason: the news correction is the
        # project's headline contribution, so "what it buys" is exactly the
        # kind of number rule 7 is about. The pivot is cheap — it reshapes rows
        # already computed — so there is no reason to report only the pooled
        # one.
        for slice_name in SLICES:
            head = full[full["slice"] == slice_name]
            if head.empty:
                continue
            print(f"\n=== what the news correction buys "
                  f"(slice: {slice_name}) ===")
            pivot = head.pivot(index="baseline", columns="t0_variant",
                               values=["precision", "median_lead_trading_h"])
            print(pivot.to_string(float_format=lambda v: f"{v:.5f}"))

    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
