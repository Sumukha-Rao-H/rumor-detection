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

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.base import Baseline
from src.rl.env import clean_observations, observation_features
from src.rl.monitor import action_summary

#: The policy's own decision rule. `score` is P(FLAG) from the policy head and
#: the deterministic action SB3 takes is the argmax over two actions, so
#: P(FLAG) >= 0.5 is exactly "this policy would flag here". Not a tunable knob
#: — it is what argmax means with two actions — so it is a constant here rather
#: than a config entry that would invite someone to move it.
OWN_FLAG_RULE = 0.5


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

        # The width of the observation the policy was BUILT with, against the
        # width of the feature list it is about to be fed. A stored policy
        # whose feature list changed length is caught here, at construction,
        # instead of inside torch several minutes into an evaluation run. It
        # cannot catch a same-length reordering — `load_policy` compares the
        # manifest's list for that — but it is the check that still works when
        # there is no manifest beside the model.
        space = getattr(model, "observation_space", None)
        if space is not None and tuple(space.shape) != (len(self.features),):
            raise ValueError(
                f"{self.name}: the saved policy takes an observation of shape "
                f"{tuple(space.shape)} but the current config asks for "
                f"{len(self.features)} features {list(self.features)}. The "
                f"policy was trained against a different feature list, so it "
                f"cannot be re-scored against this one — it has to be "
                f"RETRAINED at the current config.")

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

    def own_action_distribution(self, frame: pd.DataFrame) -> dict:
        """What the policy would do *on its own terms* — plan §6's check.

        Three things in this repo claimed to answer "is the policy
        degenerate?" and none of them did. `monitor.action_summary` was the
        right function with no caller outside its tests. The training
        callback's `flag_rate` is the lifetime average over 300k steps of
        SAMPLED actions, so it is dominated by early exploration and is not
        the behaviour anyone evaluates. And the report's
        `pct_windows_alerted` is derived at the alert BUDGET, which is a
        property of the harness rather than of the detector — it came out
        0.0189 for every row of the Phase 10 table, baselines and all five
        policy seeds alike, which is what a number describing the harness
        looks like.

        This is the missing one: the policy's own rule, `P(FLAG) >= 0.5`,
        which is what its deterministic argmax does. A policy that has
        collapsed to always-WAIT reports `flag_rate` 0.0 here however
        respectable its budgeted precision looks, and that is the failure the
        plan says is the most likely one.

        Accepts either a prediction frame (its `score` column is already
        P(FLAG), so the distribution costs nothing extra) or a raw feature
        frame, which is scored on the spot.
        """
        probs = (frame["score"] if "score" in frame.columns
                 else self.score(frame))
        flags = probs >= OWN_FLAG_RULE
        counts = {0: int((~flags).sum()), 1: int(flags.sum())}
        summary = action_summary(counts)
        # The window-level view as well as the hourly one: with at most one
        # FLAG per window an hourly rate is small even for a busy policy, so
        # the share of WINDOWS the policy would alert on is the readable half.
        # Counted over windows the policy would flag at any hour, which is what
        # a stopping policy actually does.
        n_windows = frame["window_id"].nunique()
        alerted = frame.loc[flags.to_numpy(), "window_id"].nunique()
        summary["pct_windows_alerted"] = (
            (alerted / n_windows) if n_windows else float("nan"))
        return summary


def load_policy(cfg: dict, path) -> PolicyBaseline:
    """Load a saved policy from a run directory or a .zip produced by P6-03.

    Refuses a run whose recorded feature list disagrees with the one the
    current config produces. `train.write_manifest` has always stored
    `manifest["features"]` for exactly this purpose and nothing read it.

    The loud case is already loud: all twenty stored runs record an
    eleven-feature list including `trading_hours_to_close`, which
    `observation_features` no longer returns, so they fail — but only inside
    torch, after `compare.main` has spent ten minutes building an evaluation
    frame. The dangerous case is the quiet one: swap one excluded feature for
    another and the count is unchanged, the observation space still fits, and
    `PolicyBaseline` feeds differently-ordered columns to a trained network
    with no error anywhere. That produces a number, and the number is
    meaningless.

    A mismatch is never fixable by re-scoring, so the message says so.
    """
    from stable_baselines3 import PPO

    path = Path(path)
    run_dir = path if path.is_dir() else path.parent
    if path.is_dir():
        path = path / "policy"

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text()).get("features")
        wanted = list(observation_features(cfg))
        # `is not None` rather than a plain truth test: a manifest from before
        # the key existed carries nothing to check and is let through, but an
        # empty list is a real disagreement and must not be waved past.
        if stored is not None and list(stored) != wanted:
            raise ValueError(
                f"{run_dir}: this policy was trained on "
                f"{len(stored)} feature(s) {list(stored)} but the current "
                f"config asks for {len(wanted)} feature(s) {wanted}. The "
                f"network's weights are tied to that list and to its ORDER, "
                f"so the policy cannot be re-scored against the new one — it "
                f"must be RETRAINED at the current config. (Check "
                f"baselines.gradient_boosting.exclude_features and "
                f"features.include_news_coverage, which are what "
                f"observation_features reads.)")

    return PolicyBaseline(cfg, PPO.load(str(path), device=(cfg.get("rl") or {})
                                        .get("device", "cpu")))
