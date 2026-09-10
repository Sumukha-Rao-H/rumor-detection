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


# --------------------------------------------------------------------------
# The stored feature list has to be checked, and nothing read it
# --------------------------------------------------------------------------
def test_a_run_whose_stored_features_disagree_with_the_config_is_refused(
        trained, request, tmp_path):
    """`train.write_manifest` records `features` for exactly this purpose.

    Nothing read it. Today the loud case is loud anyway — all twenty stored
    runs record an eleven-feature list including `trading_hours_to_close`,
    which `observation_features` no longer returns, so they fail inside torch
    after `compare.main` has spent ten minutes building an evaluation frame.
    The dangerous case is quiet: swap one excluded feature for another, the
    count is unchanged, the observation space still fits, and the policy is fed
    differently-ordered columns with no error anywhere. That produces a number,
    and the number means nothing.

    The message has to say the fix is retraining, because it is: the weights
    are tied to the list and to its order, so no amount of re-scoring helps.
    """
    import json
    import shutil

    from src.rl import load_policy

    cfg, _, _ = trained
    source = request.config.rootpath / ".pytest_cache" / "p605"
    run = tmp_path / "stale-run"
    run.mkdir()
    shutil.copy(source / "policy.zip", run / "policy.zip")

    manifest = json.loads((source / "manifest.json").read_text())
    stale = list(manifest["features"]) + ["trading_hours_to_close"]
    manifest["features"] = stale
    (run / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError) as excinfo:
        load_policy(cfg, run)

    message = str(excinfo.value)
    assert "RETRAINED" in message
    assert "trading_hours_to_close" in message          # names the stored list
    assert observation_features(cfg)[0] in message      # and the wanted one


def test_a_run_whose_stored_features_match_still_loads(trained, request):
    """The guard has to let a good run through, or it proves nothing."""
    from src.rl import load_policy

    cfg, _, _ = trained
    source = request.config.rootpath / ".pytest_cache" / "p605"
    assert load_policy(cfg, source).name == "rl_policy"


def test_a_policy_built_for_a_different_observation_width_is_refused(trained):
    """The check that still works when there is no manifest beside the model.

    Caught at construction rather than several minutes into an evaluation run,
    and worded the same way: a width mismatch is never fixable by re-scoring.
    """
    cfg, model, _ = trained
    feats = observation_features(cfg)
    with pytest.raises(ValueError, match="RETRAINED"):
        PolicyBaseline(cfg, model, features=feats + ("volume_z",))


# --------------------------------------------------------------------------
# Plan §6 — the degeneracy check, on the path evaluation actually takes
# --------------------------------------------------------------------------
class _NeverFlagsPolicy:
    """P(FLAG) varies smoothly with the observation but never reaches 0.5.

    The realistic collapse. Nothing here is constant — Brier, ECE, the score
    column and the ranking all look alive — but the argmax is WAIT on every
    single row, so the policy never acts.
    """

    def set_training_mode(self, mode: bool) -> None:
        pass

    def obs_to_tensor(self, obs):
        import torch

        return torch.as_tensor(obs, dtype=torch.float32), None

    def get_distribution(self, tensor):
        import torch

        flag = 0.10 + 0.35 * torch.sigmoid(tensor[:, 0])
        probs = torch.stack([1.0 - flag, flag], dim=1)
        return type("D", (), {"distribution": type("P", (), {"probs": probs})()})()


class _NeverFlagsModel:
    def __init__(self, n_features: int):
        import types

        self.policy = _NeverFlagsPolicy()
        self.observation_space = types.SimpleNamespace(shape=(n_features,))


def test_a_policy_that_never_reaches_its_own_threshold_is_reported_as_degenerate(
        cfg, frame):
    """Plan §6: "report the learned agent's action distribution, not only its
    accuracy — the most likely failure is a degenerate policy."

    Three things claimed to do this and none would catch one.
    `monitor.action_summary` had no caller outside its own tests. The training
    callback's lifetime `flag_rate` averages 300k steps of sampled actions and
    is dominated by early exploration. And the report's `pct_windows_alerted`
    is derived at the alert BUDGET, so it describes the harness — it came out
    0.0189 for every row of the Phase 10 table, all baselines and all five
    seeds alike.

    This is the end-to-end path, comparison table included: a policy whose
    P(FLAG) never reaches 0.5 reports `policy_flag_rate` 0.0, while the
    budgeted column beside it still shows a healthy-looking alert share.
    """
    from src.baselines.compare import comparison_table

    feats = observation_features(cfg)
    model = _NeverFlagsModel(len(feats))
    policy = PolicyBaseline(cfg, model)

    out = policy.predict(frame, threshold=float("inf"))
    assert out["score"].nunique() > 1          # not a constant scorer
    assert out["score"].max() < 0.5            # and it never acts

    table = comparison_table(cfg, {"rl_policy": out}, "news_adjusted",
                             models={"rl_policy": policy})
    row = table[table["slice"] == "all"].iloc[0]

    assert row["policy_flag_rate"] == 0.0
    assert row["policy_pct_windows_alerted"] == 0.0
    # The column that could not see it. This is the contrast the fix exists
    # for: the budgeted share looks entirely ordinary on a dead policy.
    assert row["pct_windows_alerted"] > 0.0


def test_a_detector_with_no_rule_of_its_own_answers_nan(cfg, frame):
    """Every model answers the same call, and none of them invents a rule.

    A threshold baseline's operating point IS whatever cut the budget hands
    it, so there is no "own" distribution to report and NaN is the honest
    answer — not 0.0, which would read as a collapsed detector.
    """
    import math

    from src.baselines import AlwaysQuiet

    out = AlwaysQuiet(cfg).predict(frame)
    dist = AlwaysQuiet(cfg).own_action_distribution(out)
    assert math.isnan(dist["flag_rate"])
    assert math.isnan(dist["pct_windows_alerted"])


def test_the_own_distribution_counts_the_deterministic_argmax(trained):
    """It has to agree with what the policy would actually do in the env.

    `score` is P(FLAG) and SB3's deterministic action is the argmax over two
    actions, so `score >= 0.5` is that argmax exactly. If these diverged, the
    reported distribution would describe a policy nobody runs.
    """
    cfg, model, frame = trained
    policy = PolicyBaseline(cfg, model)

    scores = policy.score(frame)
    expected = int((scores >= 0.5).sum())
    dist = policy.own_action_distribution(frame)

    assert dist["n_flag"] == expected
    assert dist["n_steps"] == len(frame)
    assert dist["n_flag"] + dist["n_wait"] == len(frame)
    for _ in range(20):
        pass
    env = FootprintEnv(cfg, frame)
    obs, info = env.reset()
    for _ in range(15):
        action, _ = model.predict(obs, deterministic=True)
        row = frame[(frame["window_id"] == info["window_id"]) &
                    (frame["ts_utc"] == info["ts_utc"])]
        assert (int(action) == FLAG) == \
               (float(scores.loc[row.index[0]]) >= 0.5)
        obs, _, terminated, _, info = env.step(WAIT)
        if terminated:
            obs, info = env.reset()
