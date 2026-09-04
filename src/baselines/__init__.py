"""Classical baselines — the number the learned policy has to beat.

Built before the agent on purpose: without a tuned baseline a null result in
Phase 6 means nothing, and a strawman makes the whole comparison worthless.
"""

from src.baselines.always_quiet import AlwaysQuiet
from src.baselines.base import Baseline, unscoreable_floor
from src.baselines.cusum import (CUSUM, CusumOperatingPoint, cusum_statistic,
                                 tune_cusum)
from src.baselines.gradient_boosting import (FEATURES, GradientBoosting,
                                             label_rows)
from src.baselines.volume_zscore import OperatingPoint, VolumeZScore, tune

__all__ = ["AlwaysQuiet", "Baseline", "CUSUM", "CusumOperatingPoint",
           "FEATURES", "GradientBoosting", "OperatingPoint", "VolumeZScore",
           "cusum_statistic", "label_rows", "tune", "tune_cusum",
           "unscoreable_floor"]
