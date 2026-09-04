"""The trained policy, wrapped so it reaches the metrics the same way as the rest.

P6-05 asks for evaluation "through the same `metrics.py` path as the baselines,
at the same budget". The cleanest way to guarantee that is not to write a
parallel evaluation — it is to make the policy *a* `Baseline`. Then it inherits
the whole P5-01 path: contract validation, the unscoreable rule, the seal
check, one FLAG per window, one alert budget. Any difference between its row in
P6-06's table and CUSUM's is then a difference between detectors rather than
between harnesses.

The score is P(FLAG)
--------------------
`precision_at_alert_budget` ranks windows by score, so the policy has to emit
something orderable rather than a bare action. PPO's policy head already
produces a distribution over actions, and its probability of FLAG is exactly
the right quantity: higher means "more inclined to alert here".

That also makes this the second baseline whose scores are genuine probabilities,
so Brier and ECE apply to it as they do to gradient boosting — unlike the
z-score and CUSUM statistics, which are unbounded by construction.

Scoring row by row is valid, and only because the policy is memoryless
------------------------------------------------------------------------
An MlpPolicy has no state between steps, so its action at hour *t* depends on
that hour's observation and nothing else. Scoring each row independently
therefore reproduces exactly what the policy would do inside the env.

This is the same property P6-02 relies on when it notes that episode length
leaks the label in the evaluation frame but cannot be exploited. Both rest on
memorylessness, so both would break together if a recurrent policy were
substituted — which is why the tracker carries that as a risk and why
`assert_memoryless` exists here rather than as a comment.

Observations are cleaned by `env.clean_observations`, the same function the
env uses, so the policy sees at evaluation exactly what it saw in training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines.base import Baseline
from src.rl.env import clean_observations, observation_features


def assert_memoryless(model) -> None:
    """Refuse a recurrent policy, loudly.

    Row-by-row scoring reproduces in-env behaviour only for a policy with no
    state between steps. A recurrent one would silently score differently here
    than it acts in the env — and, worse, could exploit the episode-length
    asymmetry P6-02 documents, where a positive is 48 bars and a negative is
    one. Better a refusal than a number that cannot be trusted.
    """
    name = type(model.policy).__name__
    if "Recurrent" in name or "Lstm" in name or "LSTM" in name:
        raise ValueError(
            f"{name} is recurrent. Row-by-row scoring assumes a memoryless "
            f"policy, and in the evaluation frame episode LENGTH separates the "
            f"classes perfectly (positives 48 bars, negatives 1). See P6-02 and "
            f"the tracker's risk list before enabling this.")


class PolicyBaseline(Baseline):
    """A trained stopping policy, scored like every other detector."""

    name = "rl_policy"

    def __init__(self, cfg: dict, model, features: tuple[str, ...] | None = None):
        super().__init__(cfg)
        assert_memoryless(model)
        self.model = model
        self.features = features or observation_features(cfg)
        self.obs_clip = float((cfg.get("rl") or {}).get("obs_clip", 10.0))

    def score(self, frame: pd.DataFrame) -> pd.Series:
        import torch

        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{self.name}: frame is missing feature column(s) {missing}")

        obs = clean_observations(frame.loc[:, list(self.features)].to_numpy(),
                                 self.obs_clip)
        policy = self.model.policy
        policy.set_training_mode(False)
        with torch.no_grad():
            tensor, _ = policy.obs_to_tensor(obs)
            probs = policy.get_distribution(tensor).distribution.probs
            flag_prob = probs[:, 1].cpu().numpy()
        return pd.Series(np.asarray(flag_prob, dtype="float64"),
                         index=frame.index)


def load_policy(cfg: dict, path) -> PolicyBaseline:
    """Load a saved policy from a run directory or a .zip produced by P6-03."""
    from pathlib import Path

    from stable_baselines3 import PPO

    path = Path(path)
    if path.is_dir():
        path = path / "policy"
    return PolicyBaseline(cfg, PPO.load(str(path), device=(cfg.get("rl") or {})
                                        .get("device", "cpu")))
