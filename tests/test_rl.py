"""Env and metric tests — §8 reward accounting and §11.2 scoring."""

import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import (Rollout, abstention_rate, accuracy_f1, brier, ece,
                              summarize, time_delta)
from src.rl.env import (COMMIT_FALSE, COMMIT_TRUE, WAIT, EventStore,
                        RumorVerificationEnv)

HOUR = 3600
T0 = 1_700_000_000
DIM = 12

CFG = {"reward": {"r_correct": 1.0, "r_early_bonus": 0.5, "r_wrong": -2.0,
                  "r_wait": -0.02, "r_timeout": -0.5, "T_max": 6},
       "paths": {"states": "unused"}}


def _store(tmp_path, labels=(1, 0), t_official=None):
    frame = pd.DataFrame([{
        "event_id": f"E-{i}", "label": lab, "t0_utc": T0,
        "t_official_utc": t_official if t_official is not None else T0 + 5 * HOUR,
        "split": "train"} for i, lab in enumerate(labels)])
    for i in range(len(labels)):
        np.savez(tmp_path / f"E-{i}.npz", X=np.zeros((6, DIM), dtype=np.float32))
    return EventStore(tmp_path, frame, "train")


def _env(tmp_path, **kw):
    return RumorVerificationEnv(_store(tmp_path, **kw), CFG, sequential=True)


# --- reward accounting ------------------------------------------------------

def test_a_correct_immediate_commit_earns_the_full_early_bonus(tmp_path):
    env = _env(tmp_path, labels=(1,))
    env.reset()
    _, reward, terminated, _, info = env.step(COMMIT_TRUE)
    assert terminated and info["correct"]
    assert reward == pytest.approx(1.0 + 0.5)      # (1 - 0/6) * 0.5


def test_the_bonus_decays_as_the_agent_waits(tmp_path):
    env = _env(tmp_path, labels=(1,))
    env.reset()
    env.step(WAIT)
    env.step(WAIT)
    _, reward, _, _, _ = env.step(COMMIT_TRUE)
    assert reward == pytest.approx(1.0 + 0.5 * (1 - 2 / 6))


def test_a_wrong_verdict_costs_more_than_waiting_to_the_horizon(tmp_path):
    """The asymmetry is what makes patience rational rather than hand-coded."""
    env = _env(tmp_path, labels=(0,))
    env.reset()
    _, reward, _, _, info = env.step(COMMIT_TRUE)
    assert reward == -2.0 and not info["correct"]
    assert abs(reward) > abs(CFG["reward"]["r_wait"]) * CFG["reward"]["T_max"]


def test_waiting_costs_a_little_every_hour(tmp_path):
    env = _env(tmp_path, labels=(1,))
    env.reset()
    _, reward, terminated, truncated, _ = env.step(WAIT)
    assert reward == pytest.approx(-0.02) and not (terminated or truncated)


def test_running_out_of_hours_truncates_with_the_timeout_penalty(tmp_path):
    env = _env(tmp_path, labels=(1,))
    env.reset()
    truncated = False
    for _ in range(CFG["reward"]["T_max"] + 2):
        obs, reward, terminated, truncated, info = env.step(WAIT)
        if truncated:
            break
    assert truncated and reward == -0.5
    assert info["timed_out"] and "pred" not in info


def test_delta_is_positive_when_the_agent_beats_the_news(tmp_path):
    env = _env(tmp_path, labels=(1,), t_official=T0 + 5 * HOUR)
    env.reset()
    env.step(WAIT)
    _, _, _, _, info = env.step(COMMIT_TRUE)
    assert info["delta_hours"] == pytest.approx(4.0)


def test_commit_false_predicts_the_negative_class(tmp_path):
    env = _env(tmp_path, labels=(0,))
    env.reset()
    _, reward, _, _, info = env.step(COMMIT_FALSE)
    assert info["pred"] == 0 and info["correct"] and reward > 0


# --- scoring ----------------------------------------------------------------

def _roll(label, pred, p=None, delta=None, committed=True):
    return Rollout(event_id="e", label=label, committed=committed, pred=pred,
                   p_true=p, delta_hours=delta, t_commit_hours=1)


def test_accuracy_counts_only_committed_episodes():
    rolls = [_roll(1, 1), _roll(0, 0), Rollout("e", 1, committed=False)]
    assert accuracy_f1(rolls) == pytest.approx(
        {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
         "committed": 2})
    assert abstention_rate(rolls) == pytest.approx(1 / 3)


def test_an_all_negative_policy_scores_zero_f1():
    """The failure mode a 9% minority class produces — it must be visible."""
    rolls = [_roll(0, 0) for _ in range(9)] + [_roll(1, 0)]
    stats = accuracy_f1(rolls)
    assert stats["accuracy"] == pytest.approx(0.9)
    assert stats["f1"] == 0.0


def test_brier_rewards_honest_probabilities():
    confident_right = [_roll(1, 1, p=1.0)]
    hedged = [_roll(1, 1, p=0.6)]
    assert brier(confident_right) < brier(hedged)


def test_ece_is_zero_for_a_perfectly_calibrated_set():
    rolls = [_roll(1, 1, p=1.0), _roll(0, 0, p=0.0)]
    assert ece(rolls) == pytest.approx(0.0, abs=1e-9)


def test_time_delta_ignores_verdicts_that_were_wrong():
    """Beating the news with the wrong answer is not an advantage."""
    rolls = [_roll(1, 1, delta=10.0), _roll(0, 1, delta=48.0)]
    stats = time_delta(rolls)
    assert stats["n"] == 1 and stats["median"] == pytest.approx(10.0)


def test_share_ge_24h_is_the_synopsis_target():
    rolls = [_roll(1, 1, delta=30.0), _roll(1, 1, delta=2.0)]
    assert time_delta(rolls)["share_ge_24h"] == pytest.approx(0.5)


def test_summarize_reports_every_axis():
    stats = summarize([_roll(1, 1, p=0.8, delta=12.0)])
    for key in ("accuracy", "f1", "abstention_rate", "brier", "ece",
                "mean_commit_hours", "confusion", "time_delta"):
        assert key in stats
