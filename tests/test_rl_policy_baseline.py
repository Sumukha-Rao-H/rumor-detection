"""P6-05 — the policy scored the same way as everything else.

The point of wrapping the policy as a `Baseline` rather than writing a parallel
evaluation is that it cannot then be measured differently by accident. These
tests pin the two properties that makes true: it inherits the whole P5-01 path,
and row-by-row scoring reproduces what the policy would do inside the env —
which holds only because the policy is memoryless, and is refused when it is
not.
"""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.eval import contract
from src.pipeline.split import boundaries, seal
from src.rl import (FLAG, WAIT, FootprintEnv, PolicyBaseline, clean_observations,
                    observation_features, train)
from src.rl.policy_baseline import assert_memoryless
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600
STEPS = 2048


@pytest.fixture
def cfg():
    base = load_config()
    return {**base, "rl": {**base["rl"], "n_envs": 2, "n_steps": 64,
                           "batch_size": 32, "net_arch": [16, 16]}}


@pytest.fixture
def features(cfg):
    return observation_features(cfg)


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "policy.db")


def make_frame(features, n_pos=10, n_neg=30, hours=6) -> pd.DataFrame:
    base = date_str_to_ts("2025-10-01")
    rows = []
    for i in range(n_pos + n_neg):
        positive = i < n_pos
        anchor = base + i * 200 * HOUR
        for h in range(hours):
            row = {"window_id": f"{'P' if positive else 'N'}{i}",
                   "ticker": f"T{i}", "ts_utc": anchor - (hours - h) * HOUR,
                   "t0_utc": anchor if positive else None,
                   "is_scheduled": True if positive else None,
                   "item_code": "8.01" if positive else None}
            for f in features:
                row[f] = (1.0 + h * 0.3) if positive else 0.05
            rows.append(row)
    return pd.DataFrame(rows).astype({
        "window_id": "string", "ticker": "string", "ts_utc": "Int64",
        "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string"})


@pytest.fixture
def frame(features):
    return make_frame(features)


@pytest.fixture(scope="module")
def trained(request):
    """One short training run shared by the module — training is the slow part."""
    base = load_config()
    cfg = {**base, "rl": {**base["rl"], "n_envs": 2, "n_steps": 64,
                          "batch_size": 32, "net_arch": [16, 16]}}
    feats = observation_features(cfg)
    frame = make_frame(feats)
    model, _, _ = train(cfg, frame, seed=1, timesteps=STEPS,
                        run_dir=request.config.rootpath / ".pytest_cache" / "p605")
    return cfg, model, frame


# --------------------------------------------------------------------------
# It reaches the metrics the same way as the baselines
# --------------------------------------------------------------------------
def test_output_passes_the_contract(trained):
    cfg, model, frame = trained
    out = PolicyBaseline(cfg, model).predict(frame, threshold=0.5)
    pd.testing.assert_frame_equal(out, contract.validate_predictions(out))


def test_it_is_a_baseline_so_it_inherits_the_whole_path(trained):
    """Not a parallel evaluation: the same class, so the same contract
    validation, unscoreable rule, one-FLAG-per-window and seal check."""
    from src.baselines import Baseline

    cfg, model, _ = trained
    assert isinstance(PolicyBaseline(cfg, model), Baseline)


def test_scores_are_probabilities(trained):
    """P(FLAG) from the policy head — orderable, so the budget metric can rank
    windows, and genuinely a probability, so Brier and ECE apply."""
    cfg, model, frame = trained
    out = PolicyBaseline(cfg, model).predict(frame, threshold=0.5)
    assert out["score"].between(0.0, 1.0).all()


def test_at_most_one_flag_per_window(trained):
    cfg, model, frame = trained
    out = PolicyBaseline(cfg, model).predict(frame, threshold=0.5)
    per_window = out[out["action"] == contract.FLAG].groupby("window_id").size()
    assert (per_window <= 1).all()


def test_the_seal_is_enforced(trained, conn):
    cfg, model, _ = trained
    seal(cfg, conn)
    _, val_end = boundaries(cfg)
    feats = observation_features(cfg)
    f = make_frame(feats, n_pos=1, n_neg=1, hours=2)
    f["ts_utc"] = pd.array([val_end + i * HOUR for i in range(len(f))],
                           dtype="Int64")
    f["t0_utc"] = pd.array([val_end + 99 * HOUR] * len(f), dtype="Int64")

    with pytest.raises(SystemExit, match="SEALED TEST SET"):
        PolicyBaseline(cfg, model).predict(f, threshold=0.5, conn=conn)


# --------------------------------------------------------------------------
# Row-by-row scoring must reproduce in-env behaviour
# --------------------------------------------------------------------------
def test_row_scoring_matches_what_the_policy_does_in_the_env(trained):
    """The property the whole approach rests on. If these diverged, the number
    reported would not be the number the policy would actually produce."""
    cfg, model, frame = trained
    baseline = PolicyBaseline(cfg, model)
    scores = baseline.score(frame)

    env = FootprintEnv(cfg, frame)
    obs, info = env.reset()
    for _ in range(15):
        action, _ = model.predict(obs, deterministic=True)
        row = frame[(frame["window_id"] == info["window_id"]) &
                    (frame["ts_utc"] == info["ts_utc"])]
        prob = float(scores.loc[row.index[0]])
        # The deterministic action is the argmax, so it agrees with P(FLAG).
        assert (int(action) == FLAG) == (prob > 0.5)
        obs, _, terminated, _, info = env.step(WAIT)
        if terminated:
            obs, info = env.reset()


def test_observations_are_cleaned_identically_to_training(trained):
    """One shared function, so the policy sees at evaluation exactly what it
    saw in training. Two copies that drifted would change the inputs silently."""
    cfg, model, frame = trained
    feats = observation_features(cfg)
    clip = cfg["rl"]["obs_clip"]

    raw = frame.loc[:, list(feats)].to_numpy()
    env = FootprintEnv(cfg, frame)
    np.testing.assert_array_equal(clean_observations(raw[0], clip),
                                  env._clean(raw[0]))


def test_nan_rows_are_still_scored(trained):
    """NaN becomes the neutral 0.0, so the policy has an opinion everywhere and
    no window falls to the unscoreable rule."""
    cfg, model, frame = trained
    feats = observation_features(cfg)
    dirty = frame.copy()
    dirty.loc[dirty.index[:6], feats[0]] = np.nan
    out = PolicyBaseline(cfg, model).predict(dirty, threshold=0.5)
    assert np.isfinite(out["score"]).all()


# --------------------------------------------------------------------------
# The guard that keeps the above true
# --------------------------------------------------------------------------
def test_a_recurrent_policy_is_refused():
    """Row-by-row scoring is valid only for a memoryless policy, and in the
    evaluation frame episode LENGTH separates the classes perfectly. A
    recurrent policy could exploit that; better a refusal than a number nobody
    can trust."""
    class FakeRecurrentPolicy:
        pass
    FakeRecurrentPolicy.__name__ = "RecurrentActorCriticPolicy"

    class FakeModel:
        policy = FakeRecurrentPolicy()

    with pytest.raises(ValueError, match="recurrent"):
        assert_memoryless(FakeModel())


def test_a_memoryless_policy_is_accepted(trained):
    _, model, _ = trained
    assert_memoryless(model)          # does not raise


def test_a_frame_missing_a_feature_is_refused(trained):
    cfg, model, frame = trained
    feats = observation_features(cfg)
    with pytest.raises(ValueError, match="missing feature column"):
        PolicyBaseline(cfg, model).predict(frame.drop(columns=[feats[0]]),
                                           threshold=0.5)


def test_name_is_stable_for_the_report(trained):
    cfg, model, _ = trained
    assert PolicyBaseline(cfg, model).name == "rl_policy"
