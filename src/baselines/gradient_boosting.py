"""The first baseline that learns — and the strongest classical opponent.

Volume z-score reads one feature; CUSUM accumulates that same feature. This one
is handed the whole feature matrix and finds whatever combination separates the
classes. If a boosted tree on 13 tabular columns beats Phase 6's sequential
policy, that is the project's finding and it goes in the report.

What "identical features" means
-------------------------------
The task says *"on identical features to everything else. No extra inputs."*
Read literally against the two baselines that use `volume_z` alone, that would
restrict this model to one column — which would not be gradient boosting in any
meaningful sense.

The reading taken: **the same feature matrix everyone has access to, and no data
from outside it.** The rule is about not smuggling in extra *inputs*, not about
crippling the model. z-score and CUSUM use one column because that is what those
methods are; this uses all of them because that is what this method is. What has
to be identical is the information available, and it is.

`is_scheduled` and `item_code` sit in the matrix and are deliberately excluded:
both are properties of the event being predicted, knowable only once it has
happened. Using them would be leakage of the plainest kind.

Trained on the sample, evaluated on the population
--------------------------------------------------
P4-12 drew this distinction and this is the task that needs both halves.
Training uses the 3:1 sampled frame (`negatives_per_positive`, whose config
comment reads "training only"): fitted on the true 0.46% base rate the model
would see 216 quiet rows per positive and could reach 99.5% accuracy by
answering "quiet" forever. Evaluation uses the full frame at its real base rate,
the identical population the other three are scored on.

A row is the unit of both. Training negatives come from 48-bar quiet windows
while evaluation negatives are single bars, and that is fine — the model scores
a feature vector, and a bar from a quiet window and a lone quiet bar are the
same object. Only labelling and aggregation care about window shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines.base import Baseline

#: The model's inputs, fixed and ordered. Everything numeric in P4-11's matrix.
#:
#: `is_scheduled` and `item_code` are absent on purpose — they describe the
#: event being predicted and are known only after it happens. `window_id`,
#: `ticker`, `ts_utc` and `t0_utc` are identifiers and the label.
FEATURES: tuple[str, ...] = (
    "ret_1h", "ret_4h", "ret_24h", "ret_120h",
    "volume_z", "volatility",
    "ret_rel_1h", "ret_rel_4h", "ret_rel_24h", "ret_rel_120h",
    "trading_hours_to_close", "days_since_last_8k", "days_since_last_earnings",
)


def news_features(cfg: dict) -> tuple[str, ...]:
    """The P8-01 news columns, or () when the news channel is off.

    Derived from `features.news_windows_h` rather than written out, so a change
    to the configured windows cannot leave this list naming columns the matrix
    does not have — or, worse, silently omitting ones it does, which would make
    the Phase 8 ablation quietly measure nothing.

    This baseline is the ONLY one that can use them. `cusum` and
    `volume_zscore` read `volume_z` alone and `always_quiet` reads nothing, so
    the with/without comparison lives or dies here.
    """
    if not cfg["features"].get("include_news_coverage", False):
        return ()
    cols = ["hours_since_news"]
    for w in cfg["features"]["news_windows_h"]:
        cols += [f"news_count_{w}h", f"news_breadth_{w}h"]
    return tuple(cols)


def label_rows(frame: pd.DataFrame) -> np.ndarray:
    """1 where the row belongs to a positive window, else 0.

    `t0_utc` IS the label, per the contract — there is deliberately no separate
    label column anywhere in this project, because a redundant column is one
    that can disagree with what it duplicates.
    """
    return frame["t0_utc"].notna().to_numpy().astype(int)


class GradientBoosting(Baseline):
    """Boosted trees over the whole feature matrix.

    Must be `fit()` before `predict()`. Scores are calibrated-ish
    probabilities in [0, 1], so unlike the other baselines this one is a
    legitimate target for Brier and ECE.
    """

    name = "gradient_boosting"

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__(cfg)
        self.params = dict(self.cfg.get("baselines", {}).get(
            "gradient_boosting", {}))
        self.model = None

    @property
    def features(self) -> tuple[str, ...]:
        """The columns actually used, after config exclusions.

        Excluded by default: `days_since_last_8k` and
        `days_since_last_earnings`. Not a hyperparameter choice — they are
        **contaminated by the negative-sampling design**, measured, not
        suspected. `sampling.quiet_gap_hours` is 168 h, so a training negative
        is by construction never within 7 days of a filing: 0.1% of train
        negatives have an 8-K in the last week against 18.9% of train
        positives. "Recent 8-K" therefore separates the training classes
        almost perfectly. At evaluation, where negatives are every quiet bar,
        31.5% of them have an 8-K in the last week — the learned rule is not
        merely useless there, it is inverted.

        The model was learning the sampler, not the market. Dropping the two
        nearly doubles validation precision (0.0257 -> 0.0493). Both figures
        are reported; see work-log entry 54.
        """
        excluded = set(self.params.get("exclude_features", ()))
        available = FEATURES + news_features(self.cfg)
        return tuple(f for f in available if f not in excluded)

    def _check_columns(self, frame: pd.DataFrame) -> None:
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{self.name}: frame is missing feature column(s) {missing}. "
                f"The model is fitted on a fixed, ordered column list so a "
                f"renamed or reordered matrix cannot be scored against the "
                f"wrong columns.")

    def fit(self, train: pd.DataFrame, conn=None) -> "GradientBoosting":
        """Fit on the training split. Never call this with validation data.

        `conn` is passed to `assert_not_test` so a frame straying into the
        sealed period is refused here as well as at predict time — fitting on
        test data is the one leak that no later check could catch.
        """
        from xgboost import XGBClassifier   # local: keeps import cost off the
                                            # path of baselines that never use it

        self._check_columns(train)
        if conn is not None:
            from src.pipeline import split
            split.assert_not_test(self.cfg, conn, train["ts_utc"],
                                  f"{self.name}.fit")

        y = label_rows(train)
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"{self.name}: training frame holds only one class "
                f"({'positives' if y.all() else 'negatives'}). A model that "
                f"has never seen a positive scores everything identically and "
                f"silently becomes always-quiet under another name.")

        self.model = XGBClassifier(
            n_estimators=self.params.get("n_estimators", 400),
            max_depth=self.params.get("max_depth", 4),
            learning_rate=self.params.get("learning_rate", 0.05),
            random_state=self.params.get("random_state", 42),
            n_jobs=self.params.get("n_jobs", 4),
            tree_method="hist",
            # NaN is routed by a learned default direction at each split, which
            # is strictly more information than imputation would supply — and
            # sidesteps the leakage risk of fitting an imputer on the full
            # dataset, which the code standards name directly.
            missing=np.nan,
            eval_metric="logloss",
        )
        self.model.fit(train.loc[:, list(self.features)], y)
        return self

    def score(self, frame: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError(
                f"{self.name}: not fitted. Call fit(train) first — scoring "
                f"with an untrained model would return a plausible-looking "
                f"number that means nothing.")
        self._check_columns(frame)
        proba = self.model.predict_proba(frame.loc[:, list(self.features)])[:, 1]
        return pd.Series(proba, index=frame.index, dtype="float64")
