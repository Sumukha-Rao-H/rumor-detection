"""The report table — P1-08 to P1-10 composed into what the report prints.

One row per (slice x t0 variant): precision and recall at the alert budget,
detection delay in trading hours, calibration, and the action distribution.

Two things here are easy to get wrong and both would produce flattering numbers,
so they are handled explicitly:

**Slicing must not drop the negatives.** Quiet windows have `is_scheduled = null`
because they are not events. Filtering a slice on `is_scheduled == True` deletes
every negative, leaving precision = TP/(TP+0) = 1.0 for every model, every time.
A slice therefore keeps ALL quiet windows and filters only the positives. A
false alarm is shared across slices rather than belonging to one.

**The threshold is chosen once, on the full frame.** If each slice picked its own
it would sit at its own operating point, the numbers would not be comparable,
and a slice where the model is weak would quietly get a looser threshold. One
live system has one alert budget.

**Alert counts do not sum across the scheduled/unscheduled split — positive
counts do.** Because a false alarm belongs to no event, `with_quiet` re-attaches
every quiet window to *both* the `scheduled` and `unscheduled` slices, so a
false alarm is counted once in each. `n_alerts`, `n_flag_hours`,
`pct_hours_flagged` and `pct_windows_alerted` for `scheduled` + `unscheduled`
will therefore generally NOT add up to the pooled `all` row whenever there are
false alarms. `n_positive` and recall are unaffected — positives are disjoint
by construction, so `scheduled.n_positive + unscheduled.n_positive ==
all.n_positive` always holds. Do not sum or weighted-average the alert-count
columns across this split; read each slice's own precision/recall instead.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd

from src.eval.contract import FLAG, WAIT, actions_from_scores, validate_predictions
from src.eval.metrics import (
    ScoresAreNotProbabilities,
    alert_budget,
    _calibration_summary,
    _delay_summary,
    precision_at_alert_budget,
    window_summary,
)
from src.utils.config import load_config

log = logging.getLogger(__name__)

#: Column order the report depends on. Pinned by a test.
COLUMNS = [
    "slice", "t0_variant", "n_windows", "n_positive", "base_rate",
    "threshold", "n_alerts", "precision", "max_precision", "recall",
    "median_lead_trading_h", "median_lead_wall_h", "n_missed",
    "n_wait_hours", "n_flag_hours", "pct_hours_flagged", "pct_windows_alerted",
    "brier", "brier_skill_score", "ece", "calibration_base_rate",
    "budget_exceeds_windows", "tie_spill_ratio", "degenerate",
]


def action_distribution(df: pd.DataFrame) -> dict:
    """WAIT/FLAG counts, at both the hour and the window level.

    The per-hour share is reported because the plan asks for it, but it is
    misleading alone: with at most one FLAG per 48-hour window it never exceeds
    ~2% even for a busy model. `pct_windows_alerted` is the interpretable one.
    """
    n_hours = len(df)
    n_flag = int((df["action"] == FLAG).sum())
    alerted = df.loc[df["action"] == FLAG, "window_id"].nunique()
    n_windows = df["window_id"].nunique()
    return {
        "n_wait_hours": int((df["action"] == WAIT).sum()),
        "n_flag_hours": n_flag,
        "pct_hours_flagged": (n_flag / n_hours) if n_hours else float("nan"),
        "pct_windows_alerted": (alerted / n_windows) if n_windows else float("nan"),
    }


def slice_frames(df: pd.DataFrame, split_by: list[str] | None = None
                 ) -> dict[str, pd.DataFrame]:
    """Split into named slices, **keeping every quiet window in each one**.

    See the module docstring: filtering on `is_scheduled` or `item_code`
    directly would delete all the negatives and force precision to 1.0.

    AGENTS.md rule 5 makes the scheduled/unscheduled split non-negotiable —
    "every number split scheduled vs unscheduled" — so this refuses to run
    with a `split_by` (from config or passed explicitly) that drops either
    one, rather than silently emitting a pooled-only report.
    """
    splits = split_by if split_by is not None else load_config()["eval"]["split_by"]
    missing = {"scheduled", "unscheduled"} - set(splits)
    if missing:
        raise ValueError(
            f"split_by is missing {sorted(missing)} — AGENTS.md rule 5 "
            "requires every number split scheduled vs unscheduled, so this "
            "split cannot be dropped. Check config.eval.split_by."
        )
    quiet = df[df["t0_utc"].isna()]
    positive = df[df["t0_utc"].notna()]

    out: dict[str, pd.DataFrame] = {"all": df}

    def with_quiet(pos: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([pos, quiet], ignore_index=True)

    if "scheduled" in splits:
        out["scheduled"] = with_quiet(positive[positive["is_scheduled"] == True])  # noqa: E712
    if "unscheduled" in splits:
        out["unscheduled"] = with_quiet(positive[positive["is_scheduled"] == False])  # noqa: E712
    if "item_code" in splits:
        # `item_code` carries a filing's WHOLE item list ("2.02,8.01"), because
        # an 8-K reports every item it covers. Matching it as one atomic string
        # gave a multi-item filing its own private bucket ("item 2.02,8.01")
        # and left it absent from both `item 2.02` and `item 8.01` — and 4,945
        # of 16,842 events carry more than one code, so that is roughly a
        # third of the population missing from the per-item breakdown.
        #
        # A multi-item filing now contributes to EVERY component item's slice.
        # Item slices therefore overlap and do not sum to the total, exactly as
        # the scheduled/unscheduled slices do not once quiet windows are shared
        # — a per-item breakdown of multi-item filings cannot be a partition.
        #
        # Codes in `config.items.exclude` never become a slice — see
        # `_item_codes`. Splitting the raw string would otherwise hand 9.01 and
        # 5.07 their own rows despite the pipeline having filtered them out.
        excluded = frozenset(load_config()["items"]["exclude"])
        codes = positive["item_code"].dropna().map(
            lambda raw: _item_codes(raw, excluded))
        for code in sorted({c for row in codes for c in row}):
            match = codes.map(lambda row, c=code: c in row)
            out[f"item {code}"] = with_quiet(positive[match.reindex(
                positive.index, fill_value=False)])

    return out


def _item_codes(raw: str, exclude: frozenset[str] | None = None) -> frozenset[str]:
    """The individual 8-K item codes inside one stored `item_code` value,
    minus the ones `config.items.exclude` drops.

    The exclusion has to be applied HERE and not only in `pipeline/events.py`,
    because splitting the stored string resurrects codes that filtering already
    removed. `item_code` holds a filing's whole item list, and the excluded
    codes ride along inside other filings' lists: 9.01 is an attachment marker,
    not an event type, which is precisely why the plan (§5) excludes it — "it
    would dominate the label distribution". It did. Before this, the per-item
    breakdown carried an `item 9.01` row holding 869 of the 1,011 test
    positives (86%) and an `item 5.07` row holding 29, and those two accounted
    for 36 of the table's 378 rows — slices for event types this study
    deliberately does not study.

    Reading the same `items.exclude` list `pipeline/events.py` reads, rather
    than naming the codes here, keeps the two from drifting apart.

    The more thorough fix is to store the already-filtered list in the frame
    instead of the raw items string. That is not done because the on-disk
    `data/processed/features.parquet` carries the raw string, so the code and
    the 31 MB shared artifact would disagree until it was regenerated — and
    regenerating it is not a side effect a report-formatting fix gets to have.
    Dropping the codes at the point of use gives the same table.

    No surviving cell moves: an excluded code only ever ADDED a slice, never
    changed the membership of another one, so the headline, scheduled and
    unscheduled rows are untouched.

    `exclude` is passed in by `slice_frames` so the config is read once for the
    whole frame rather than once per positive row — this is mapped over every
    positive, and on the real eval frame that is tens of thousands of calls.
    """
    excluded = (exclude if exclude is not None
                else frozenset(load_config()["items"]["exclude"]))
    return frozenset(part.strip() for part in str(raw).split(",")
                     if part.strip() and part.strip() not in excluded)


def evaluate(df: pd.DataFrame, threshold: float | None = None,
             max_alerts: int | None = None) -> dict:
    """Every metric for one frame, at one operating point.

    `threshold` is passed in so that all slices share the operating point
    chosen on the full frame. Omit it only when evaluating a frame on its own.
    """
    frame = validate_predictions(df)

    if threshold is None:
        threshold = precision_at_alert_budget(frame, max_alerts=max_alerts).threshold

    # actions_from_scores only rewrites `action`, and by construction sets at
    # most one FLAG per window, so the result is still contract-valid. From
    # here the private metric paths are used: validating once per slice rather
    # than three times (P1-Xb).
    decided = actions_from_scores(frame, threshold)

    windows = window_summary(decided)
    n_windows = len(windows)
    n_positive = int(windows["is_positive"].sum())
    alerted = windows[windows["peak_score"] >= threshold]
    tp = int(alerted["is_positive"].sum())

    delay = _delay_summary(decided)
    actions = action_distribution(decided)
    # Sized from the already-validated frame (see `alert_budget`), so the
    # ceiling costs no extra validation pass. Per-slice on purpose: a slice
    # with fewer positives has a lower ceiling, and its precision has to be
    # read against its own.
    budget_n = alert_budget(frame, max_alerts=max_alerts)

    try:
        cal = _calibration_summary(decided)
        brier, skill, ece = cal.brier, cal.brier_skill_score, cal.ece
        # Carried because the `base_rate` column beside it is a DIFFERENT
        # number and the row was silently mixing the two. `base_rate` is
        # per-WINDOW (n_positive / n_windows); calibration is computed per
        # (window, hour), which is the right unit for it — every hour of a
        # positive window is labelled 1. A positive is a 48-bar episode while a
        # quiet negative is a single bar, so on the real eval frame the two
        # rates differ by about 42x: the `all` row read base_rate = 0.003187
        # next to brier = 0.13306, and the skill score's baseline — the
        # forecast Brier is scored against — was the unshown 13.3% per-hour
        # rate. Both are now on the row, so neither has to be inferred.
        cal_base_rate = cal.base_rate
    except ScoresAreNotProbabilities:
        # A threshold baseline. nan means "not applicable" here, not "failed";
        # `_calibration_summary` still refuses loudly when called directly.
        # Any OTHER error propagates. This used to be a substring match on the
        # message text, an invisible contract that a reword would have broken
        # silently; the named type makes it explicit on both sides.
        brier = skill = ece = cal_base_rate = float("nan")

    return {
        "n_windows": n_windows,
        "n_positive": n_positive,
        "base_rate": (n_positive / n_windows) if n_windows else float("nan"),
        "threshold": threshold,
        "n_alerts": len(alerted),
        "precision": (tp / len(alerted)) if len(alerted) else float("nan"),
        # Issue 29, carried beside precision on purpose: the budget is SPENT,
        # not capped, so when it exceeds the positives available the ceiling
        # is below 1.0 and precision must be read against it. 15% next to a
        # 21.2% ceiling is 71% of achievable; 15% alone reads as failure.
        #
        # The denominator is `len(alerted)` — the SAME one precision uses —
        # and that is the whole point. It used to be `budget_n`, the allowance,
        # and a ceiling computed over a different denominator than the number
        # it caps is not a ceiling: 10 rows of the published Phase 10 table had
        # precision ABOVE their own `max_precision`. Two ways the two
        # denominators come apart, and the table showed both:
        #
        #   a slice        `budget_n` is sized from the slice's ticker-months,
        #                  and a slice keeps every quiet window (see the module
        #                  docstring) so its budget stays near the whole
        #                  frame's — while the alerts it actually receives fall
        #                  with its positives. `item 1.05`: 1 positive, 5,560
        #                  alerts, precision 0.000180, "ceiling" 0.000167.
        #
        #   tie spill      ties at the budget boundary are admitted together,
        #                  so `len(alerted)` can exceed the allowance.
        #
        # The honest question is "given that THIS many alerts went out, what is
        # the best precision anyone could have had?", and the answer is that a
        # flawless detector ranks every positive first and still has to spend
        # the rest of its alerts on negatives. On the headline `all` rows where
        # the alerts land exactly on the budget (cusum and volume_zscore, 5,996
        # of 5,996) the value is UNCHANGED at 0.16861 — the headline does not
        # move; only rows whose alert count already disagreed with the
        # allowance do.
        #
        # `BudgetResult.max_precision` in metrics.py still divides by the
        # allowance and so still has the incoherence. That is deliberate for
        # now, not an oversight: it is pinned by
        # `test_the_ceiling_is_incoherent_when_the_budget_exceeds_the_windows`,
        # which asserts `max_precision < precision` in the budget-exceeds-
        # windows regime — an assertion no coherent ceiling can satisfy. The
        # report table is what gets published, so it is fixed here; changing
        # the dataclass needs that test changed with it.
        "max_precision": (min(n_positive, len(alerted)) / len(alerted))
                         if len(alerted) else float("nan"),
        "recall": (tp / n_positive) if n_positive else float("nan"),
        "median_lead_trading_h": delay.median_trading_hours,
        "median_lead_wall_h": delay.median_wall_clock_hours,
        "n_missed": delay.n_missed,
        "brier": brier,
        "brier_skill_score": skill,
        "ece": ece,
        "calibration_base_rate": cal_base_rate,
        # `BudgetResult` has carried this since P1-08 and the report threw it
        # away. In this regime the allowance is larger than the number of
        # windows, so everything alerts, and precision collapses mechanically
        # to the base rate whatever the detector does —
        # `tests/test_baselines_volume_zscore.py` tells a reader this is the
        # flag to check before quoting a ceiling, and until now the report they
        # would be reading did not contain it. (No Phase 10 row entered it.)
        "budget_exceeds_windows": bool(budget_n >= n_windows),
        # How far the alerts actually issued overran the allowance. 1.0 means
        # the budget was spent exactly; anything much above it means ties at
        # the boundary were admitted in a block, so the alert set was settled
        # by tie-breaking rather than by ranking. `always_quiet` in the Phase
        # 10 run: 317,198 alerts against a 5,996 budget, a ratio of 53.
        #
        # This is a diagnostic column and NOT a widening of `degenerate`.
        # `degenerate` is a published verdict, and teaching it to fire on tie
        # spill could reclassify a legitimate detector that merely has a flat
        # patch at the boundary. A column a reader can see does the same job
        # with none of that risk: the signal was already being computed here
        # and thrown away.
        "tie_spill_ratio": (len(alerted) / budget_n) if budget_n
                           else float("nan"),
        # TRUE when this row's precision is not evidence of detection ability.
        # Two distinct ways that happens, and the column needs both:
        #
        #   no flags at all      the original meaning, and still the right
        #                        answer whenever a caller supplies a threshold
        #                        the scores never cross.
        #
        #   scores that cannot   `precision_at_alert_budget` SPENDS the budget:
        #   rank                 it ranks every window and takes the top k. A
        #                        detector emitting one constant therefore still
        #                        "alerts", but on whichever rows the sort
        #                        happened to leave on top — a tie-breaking
        #                        artifact, not a detection.
        #
        # The second case is why this column read `False` for `always_quiet`
        # while reporting 317,198 alerts against a detector that never flags by
        # construction: `evaluate` re-derives actions from scores at the chosen
        # operating point, which is right for comparability and discards the
        # WAIT the model actually emitted. Phase 6 needs this column to catch a
        # collapsed policy, and on scores alone it would not have. Found
        # 2026-09-04 (work-log 55), fixed 2026-09-08.
        "degenerate": bool(actions["n_flag_hours"] == 0
                           or frame["score"].nunique(dropna=True) <= 1),
        **actions,
    }


def report_table(frames: Mapping[str, pd.DataFrame] | pd.DataFrame,
                 max_alerts: int | None = None) -> pd.DataFrame:
    """The full table: one row per (slice x t0 variant).

    `frames` maps a t0 variant name to its prediction frame — the same model
    evaluated once per variant in `config.eval.t0_variants`, since the contract
    carries one t0 column. A bare DataFrame is treated as a single unnamed
    variant.

    A detector that cannot detect shows up as `degenerate=True` — either
    because it never flags, or because its scores are all one value and so
    cannot rank. See `evaluate` for why the second case is not optional: the
    alert budget is spent rather than capped, so a constant-scoring detector
    still collects alerts, and reading its precision as skill would be wrong.
    """
    if isinstance(frames, pd.DataFrame):
        frames = {"default": frames}

    rows = []
    for variant, df in frames.items():
        frame = validate_predictions(df)
        # One operating point per variant, chosen on everything.
        threshold = precision_at_alert_budget(frame, max_alerts=max_alerts).threshold
        for name, sliced in slice_frames(frame).items():
            if sliced.empty:
                log.warning(
                    "report_table: slice %r (t0_variant=%r) is completely "
                    "empty (no positives, no quiet windows) — omitting its "
                    "row rather than reporting on zero data.", name, variant,
                )
                continue
            # `max_alerts` has to travel with the threshold it produced.
            # Dropping it here let `evaluate` re-derive its own budget from
            # ticker-months x the config rate, so a caller asking for 400
            # alerts got a threshold sized for 400 and a ceiling sized for
            # thousands — at max_alerts=400 the table advertised a ceiling of
            # 1.0 beside a precision of 0.167. Phase 10 never passed it
            # (`compare.comparison_table` does not), so nothing published is
            # affected; forwarding it means the two cannot disagree.
            rows.append({"slice": name, "t0_variant": variant,
                         **evaluate(sliced, threshold=threshold,
                                    max_alerts=max_alerts)})

    return pd.DataFrame(rows, columns=COLUMNS)


def t0_variant_gap(table: pd.DataFrame, metric: str = "median_lead_trading_h",
                   slice_name: str = "all") -> float:
    """Difference in a metric between the two t0 variants: first minus second.

    The plan requires both variants and the gap between them to be reported —
    that comparison is itself a small contribution, since it measures how much
    of the apparent warning was really time the market already knew. `filing`
    t0 ignores any news that broke before the filing, so its lead time is the
    inflated one; `news_adjusted` uses `min(filing, news)` and is the honest
    figure. The gap is `filing - news_adjusted` so it reads as "how much lead
    time was fake."

    Which row is "first" and which is "second" is pinned by
    `config.eval.t0_variants` (its declared order), not by the order rows
    happen to appear in `table` — that order follows the iteration order of
    whatever mapping was passed to `report_table`, which is not something a
    caller should have to control to get a stable sign. If the two variant
    names in `table` don't match the configured pair, the two names are
    sorted so the result is still independent of caller/dict ordering — just
    without the "filing minus news_adjusted" meaning.
    """
    variants = table[table["slice"] == slice_name].set_index("t0_variant")[metric]
    if len(variants) != 2:
        return float("nan")

    configured = [v for v in load_config()["eval"]["t0_variants"] if v in variants.index]
    first, second = configured if len(configured) == 2 else sorted(variants.index)
    return float(variants[first] - variants[second])
