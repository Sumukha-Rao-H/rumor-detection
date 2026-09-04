"""P6-03 — the training loop, and the reproducibility that is its done-when.

`test_the_same_seed_reproduces_identical_parameters` is the task. A result
nobody can regenerate is a result nobody can check, and this project is graded
on whether its numbers can be defended — so the seed has to actually determine
the run, and the manifest has to record everything else that does.

Runs here use a few thousand timesteps. These test the machinery, not the
policy; whether the agent learns anything is P6-04's question and is measured
on real data, not asserted in a unit test.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import db
from src.rl import FLAG, WAIT, data_fingerprint, observation_features, train
from src.rl.train import write_manifest
from src.pipeline.split import boundaries, seal
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600
STEPS = 2048


@pytest.fixture
def cfg():
    """Tiny and fast: two envs, short rollouts, a small net."""
    base = load_config()
    return {**base, "rl": {**base["rl"], "n_envs": 2, "n_steps": 64,
                           "batch_size": 32, "net_arch": [16, 16],
                           "total_timesteps": STEPS}}


@pytest.fixture
def features(cfg):
    return observation_features(cfg)


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "train.db")


def make_frame(features, n_pos: int = 12, n_neg: int = 36,
               hours: int = 8) -> pd.DataFrame:
    """Positives carry a rising signal; quiet windows do not."""
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


def params(model) -> np.ndarray:
    """Flatten every policy parameter into one vector for comparison."""
    return np.concatenate([p.detach().cpu().numpy().ravel()
                           for p in model.policy.parameters()])


# --------------------------------------------------------------------------
# The done-when
# --------------------------------------------------------------------------
def test_training_runs_and_returns_a_model(cfg, frame, tmp_path):
    model, manifest, callback = train(cfg, frame, seed=1, timesteps=STEPS,
                                      run_dir=tmp_path / "run")
    assert model is not None
    assert manifest["seed"] == 1
    assert manifest["total_timesteps"] == STEPS
    assert (tmp_path / "run" / "policy.zip").exists()


def test_the_same_seed_reproduces_identical_parameters(cfg, frame, tmp_path):
    """THE done-when. If this fails the run cannot be defended, whatever it
    scores."""
    a, _, _ = train(cfg, frame, seed=7, timesteps=STEPS, run_dir=tmp_path / "a")
    b, _, _ = train(cfg, frame, seed=7, timesteps=STEPS, run_dir=tmp_path / "b")
    np.testing.assert_allclose(params(a), params(b), rtol=0, atol=0)


def test_different_seeds_diverge(cfg, frame, tmp_path):
    """Proves the check above is not vacuous — two runs that agreed no matter
    what would pass it while the seed did nothing."""
    a, _, _ = train(cfg, frame, seed=1, timesteps=STEPS, run_dir=tmp_path / "a")
    b, _, _ = train(cfg, frame, seed=2, timesteps=STEPS, run_dir=tmp_path / "b")
    assert not np.allclose(params(a), params(b))


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------
def test_the_manifest_records_what_is_needed_to_reproduce(cfg, frame, tmp_path):
    _, manifest, _ = train(cfg, frame, seed=3, timesteps=STEPS,
                           run_dir=tmp_path / "run")
    saved = json.loads((tmp_path / "run" / "manifest.json").read_text())

    assert saved["seed"] == 3
    assert saved["rl_config"]["gamma"] == 1.0
    assert saved["reward"] == cfg["reward"]
    assert saved["features"] == list(observation_features(cfg))
    for key in ("rows", "windows", "positives", "window_id_sha256"):
        assert key in saved["data"]
    for key in ("python", "stable_baselines3", "torch", "numpy"):
        assert saved["versions"][key]
    assert "commit" in saved["git"] and "dirty" in saved["git"]
    assert saved["elapsed_s"] >= 0


def test_the_manifest_is_written_before_training(cfg, frame, tmp_path):
    """A run that crashes half way must still be identifiable; an
    unidentifiable run in a results directory is worse than no run."""
    path = tmp_path / "early" / "manifest.json"
    write_manifest(path, cfg, seed=5, frame=frame)
    assert json.loads(path.read_text())["seed"] == 5


def test_the_data_fingerprint_changes_with_the_data(cfg, features):
    """A seed identifies a run only against the same data, so 'the same data'
    has to be checkable rather than assumed."""
    a = data_fingerprint(make_frame(features, n_pos=12))
    b = data_fingerprint(make_frame(features, n_pos=13))
    assert a["window_id_sha256"] != b["window_id_sha256"]
    assert a["positives"] != b["positives"]


def test_the_fingerprint_is_stable_for_the_same_data(cfg, frame):
    assert data_fingerprint(frame) == data_fingerprint(frame.iloc[::-1])


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
def test_training_refuses_the_sealed_test_split(cfg, features, conn):
    """Fitting on test data is the one leak no later check could catch."""
    seal(cfg, conn)
    _, val_end = boundaries(cfg)
    frame = make_frame(features, n_pos=2, n_neg=2, hours=4)
    frame["ts_utc"] = pd.array([val_end + i * HOUR for i in range(len(frame))],
                               dtype="Int64")

    with pytest.raises(SystemExit, match="SEALED TEST SET"):
        train(cfg, frame, conn=conn, seed=1, timesteps=64)


def test_an_unimplemented_algo_is_refused(cfg, frame, tmp_path):
    """Better a named refusal than silently training something else and
    reporting it under the configured name."""
    other = {**cfg, "rl": {**cfg["rl"], "algo": "DQN"}}
    with pytest.raises(SystemExit, match="not implemented"):
        train(other, frame, seed=1, timesteps=64, run_dir=tmp_path / "r")


def test_gamma_is_one_by_default(cfg):
    """Pinned because it is a deliberate departure from the usual 0.99:
    discounting would add a second, unstated preference for flagging early on
    top of r_early_bonus."""
    assert load_config()["rl"]["gamma"] == 1.0


# --------------------------------------------------------------------------
# Monitoring and logging
# --------------------------------------------------------------------------
def test_the_action_callback_counts_both_actions(cfg, frame, tmp_path):
    """P6-04 needs this: a policy collapsing to always-WAIT looks healthy on
    reward alone, because restraint on a quiet window pays 0 and quiet windows
    dominate."""
    _, _, callback = train(cfg, frame, seed=1, timesteps=STEPS,
                           run_dir=tmp_path / "run")
    assert sum(callback.counts.values()) > 0
    assert set(callback.counts) <= {WAIT, FLAG}
    assert 0.0 <= callback.flag_rate <= 1.0


def test_tensorboard_files_are_written(cfg, frame, tmp_path):
    train(cfg, frame, seed=1, timesteps=STEPS, run_dir=tmp_path / "run")
    events = list((tmp_path / "run" / "tb").rglob("events.out.tfevents.*"))
    assert events, "no tensorboard event files were written"


def test_the_policy_produces_valid_actions(cfg, frame, tmp_path):
    from src.rl import FootprintEnv

    model, _, _ = train(cfg, frame, seed=1, timesteps=STEPS,
                        run_dir=tmp_path / "run")
    env = FootprintEnv(cfg, frame)
    obs, _ = env.reset()
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        assert int(action) in (WAIT, FLAG)
        obs, _, terminated, _, _ = env.step(int(action))
        if terminated:
            obs, _ = env.reset()
