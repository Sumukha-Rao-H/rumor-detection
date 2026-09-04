"""P6-01 — the environment, and the reward arithmetic it must not improvise.

Every reward here is checked against `config.reward` rather than against a
literal, so a change to the table is caught by these tests instead of silently
retraining the agent on different incentives.

The other load-bearing test is `test_sampler_contaminated_features_are_excluded`.
The tracker carries it as a risk: the agent trains on the same 3:1 sampled
negatives that taught gradient boosting to read `days_since_last_8k` instead of
the market, and this env is where that has to be prevented.
"""

import numpy as np
import pandas as pd
import pytest

from src.rl import FLAG, WAIT, FootprintEnv, episodes_from_frame, observation_features
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def features(cfg):
    return observation_features(cfg)


def make_frame(specs, features, base_value: float = 0.5) -> pd.DataFrame:
    """specs: list of (window_id, positive?, n_hours)."""
    base = date_str_to_ts("2025-10-01")
    rows = []
    for i, (wid, positive, n) in enumerate(specs):
        anchor = base + i * 200 * HOUR
        for h in range(n):
            row = {"window_id": wid, "ticker": f"T{i}",
                   "ts_utc": anchor - (n - h) * HOUR,
                   "t0_utc": anchor if positive else None,
                   "is_scheduled": True if positive else None,
                   "item_code": "8.01" if positive else None}
            for f in features:
                row[f] = base_value + h * 0.1
            rows.append(row)
    return pd.DataFrame(rows).astype({
        "window_id": "string", "ticker": "string", "ts_utc": "Int64",
        "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string"})


@pytest.fixture
def positive_env(cfg, features):
    return FootprintEnv(cfg, make_frame([("P0", True, 6)], features))


@pytest.fixture
def quiet_env(cfg, features):
    return FootprintEnv(cfg, make_frame([("N0", False, 6)], features))


# --------------------------------------------------------------------------
# Episode termination
# --------------------------------------------------------------------------
def test_episode_ends_at_flag(positive_env):
    positive_env.reset()
    _, _, terminated, truncated, info = positive_env.step(FLAG)
    assert terminated is True
    assert truncated is False
    assert info["outcome"] == "correct_flag"


def test_episode_ends_at_t0_without_a_flag(positive_env):
    """Running out of window is termination, not truncation: the episode
    reached its natural end at t0."""
    positive_env.reset()
    for _ in range(5):
        _, _, terminated, _, _ = positive_env.step(WAIT)
        assert terminated is False
    _, _, terminated, truncated, info = positive_env.step(WAIT)
    assert terminated is True
    assert truncated is False
    assert info["outcome"] == "missed"


def test_a_one_bar_episode_terminates_cleanly(cfg, features):
    """The evaluation frame's negatives are single bars, so this is the shape
    the env will meet most often at scoring time."""
    env = FootprintEnv(cfg, make_frame([("N0", False, 1)], features))
    env.reset()
    _, reward, terminated, _, info = env.step(WAIT)
    assert terminated is True
    assert info["outcome"] == "correct_wait"
    assert reward == 0.0


def test_step_before_reset_raises(positive_env):
    with pytest.raises(RuntimeError, match="before reset"):
        positive_env.step(WAIT)


# --------------------------------------------------------------------------
# Reward arithmetic, always against the config table
# --------------------------------------------------------------------------
def test_a_correct_flag_pays_the_configured_reward(cfg, positive_env):
    positive_env.reset()
    _, reward, _, _, _ = positive_env.step(FLAG)
    r = cfg["reward"]
    # Flagged at step 0 of a 6-bar window: the whole bonus.
    assert reward == pytest.approx(r["r_correct_flag"] + r["r_early_bonus"])


def test_a_false_alarm_pays_the_configured_penalty(cfg, quiet_env):
    quiet_env.reset()
    _, reward, _, _, info = quiet_env.step(FLAG)
    assert reward == pytest.approx(cfg["reward"]["r_false_alarm"])
    assert info["outcome"] == "false_alarm"


def test_waiting_costs_the_configured_amount(cfg, positive_env):
    positive_env.reset()
    _, reward, _, _, _ = positive_env.step(WAIT)
    assert reward == pytest.approx(cfg["reward"]["r_wait"])


def test_missing_a_positive_pays_the_miss_penalty(cfg, positive_env):
    positive_env.reset()
    rewards = [positive_env.step(WAIT)[1] for _ in range(6)]
    assert rewards[-1] == pytest.approx(cfg["reward"]["r_missed"])


def test_restraint_on_a_quiet_window_pays_nothing(cfg, quiet_env):
    """Correct restraint is the baseline, not an achievement."""
    quiet_env.reset()
    rewards = [quiet_env.step(WAIT)[1] for _ in range(6)]
    assert rewards[-1] == 0.0


def test_a_false_alarm_costs_more_than_a_hit_pays(cfg):
    """The reward-shaping equivalent of the alert budget. If this inverts, the
    agent is being paid to flag everything."""
    r = cfg["reward"]
    assert abs(r["r_false_alarm"]) > r["r_correct_flag"] + r["r_early_bonus"]


# --------------------------------------------------------------------------
# The early bonus
# --------------------------------------------------------------------------
def test_the_early_bonus_decays_with_time(cfg, features):
    """An alert 40 hours before the announcement must be worth more than one
    an hour before — otherwise the reward ignores what the project measures."""
    rewards = []
    for wait_steps in (0, 2, 4):
        env = FootprintEnv(cfg, make_frame([("P0", True, 6)], features))
        env.reset()
        for _ in range(wait_steps):
            env.step(WAIT)
        rewards.append(env.step(FLAG)[1])
    assert rewards[0] > rewards[1] > rewards[2]


def test_flagging_on_the_last_bar_is_still_correct(cfg, features):
    """That hour is strictly before t0, so it is a legitimate hit with a bonus
    near zero — not a miss."""
    env = FootprintEnv(cfg, make_frame([("P0", True, 6)], features))
    env.reset()
    for _ in range(5):
        env.step(WAIT)
    _, reward, _, _, info = env.step(FLAG)
    assert info["outcome"] == "correct_flag"
    assert reward == pytest.approx(cfg["reward"]["r_correct_flag"])


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------
def test_observation_matches_the_feature_row(cfg, features):
    env = FootprintEnv(cfg, make_frame([("P0", True, 4)], features))
    obs, _ = env.reset()
    assert obs.shape == (len(features),)
    assert np.allclose(obs, 0.5, atol=1e-6)          # first row, base_value


def test_nan_becomes_zero_and_values_are_clipped(cfg, features):
    frame = make_frame([("P0", True, 3)], features)
    frame.loc[0, features[0]] = np.nan
    frame.loc[0, features[1]] = 1e6
    env = FootprintEnv(cfg, frame)
    obs, _ = env.reset()

    assert obs[0] == 0.0
    assert obs[1] == pytest.approx(cfg["rl"]["obs_clip"])
    assert np.isfinite(obs).all()


def test_the_observation_space_bounds_match_the_clip(cfg, positive_env):
    clip = cfg["rl"]["obs_clip"]
    assert positive_env.observation_space.low.min() == pytest.approx(-clip)
    assert positive_env.observation_space.high.max() == pytest.approx(clip)


# --------------------------------------------------------------------------
# The P5-05 risk, honoured here
# --------------------------------------------------------------------------
def test_sampler_contaminated_features_are_excluded(cfg):
    """The tracker's Phase 6 risk. Training negatives come from the same 3:1
    sample that taught gradient boosting to read the sampler: quiet_gap_hours
    is 168 h, so 0.1% of training negatives have a recent 8-K against 31.5% of
    evaluation negatives."""
    feats = observation_features(cfg)
    assert "days_since_last_8k" not in feats
    assert "days_since_last_earnings" not in feats
    assert "volume_z" in feats


def test_the_exclusion_follows_the_same_config_knob_as_the_tree(cfg):
    """One knob, not two lists that can drift apart and disagree about what is
    safe to learn from."""
    permissive = {**cfg, "baselines": {
        **cfg["baselines"],
        "gradient_boosting": {**cfg["baselines"]["gradient_boosting"],
                              "exclude_features": []}}}
    assert len(observation_features(permissive)) > len(observation_features(cfg))


def test_a_frame_missing_a_feature_is_refused(cfg, features):
    frame = make_frame([("P0", True, 3)], features).drop(columns=[features[0]])
    with pytest.raises(ValueError, match="missing feature column"):
        FootprintEnv(cfg, frame)


def test_an_empty_frame_is_refused(cfg, features):
    frame = make_frame([("P0", True, 3)], features).iloc[:0]
    with pytest.raises(ValueError, match="no episodes"):
        FootprintEnv(cfg, frame)


# --------------------------------------------------------------------------
# Ordering and reproducibility
# --------------------------------------------------------------------------
def test_rows_are_sorted_on_ingest(cfg, features):
    """The env serves rows positionally, so the whole no-lookahead property
    rests on this order being right rather than assumed."""
    frame = make_frame([("P0", True, 5)], features)
    episodes = episodes_from_frame(frame.iloc[::-1], features)
    assert list(episodes[0]["ts_utc"]) == sorted(episodes[0]["ts_utc"])


def test_the_seed_reproduces_the_episode_order(cfg, features):
    frame = make_frame([(f"W{i}", i % 2 == 0, 3) for i in range(8)], features)
    a = FootprintEnv(cfg, frame, seed=7)
    b = FootprintEnv(cfg, frame, seed=7)
    order_a = [a.reset()[1]["window_id"] for _ in range(8)]
    order_b = [b.reset()[1]["window_id"] for _ in range(8)]
    assert order_a == order_b


def test_different_seeds_give_different_orders(cfg, features):
    frame = make_frame([(f"W{i}", i % 2 == 0, 3) for i in range(12)], features)
    a = [FootprintEnv(cfg, frame, seed=1).reset()[1]["window_id"]
         for _ in range(1)]
    b = [FootprintEnv(cfg, frame, seed=99).reset()[1]["window_id"]
         for _ in range(1)]
    # Not a guarantee for any single draw, but over 12 windows a fixed pair of
    # seeds landing identically would signal the seed is being ignored.
    env_a = FootprintEnv(cfg, frame, seed=1)
    env_b = FootprintEnv(cfg, frame, seed=99)
    seq_a = [env_a.reset()[1]["window_id"] for _ in range(12)]
    seq_b = [env_b.reset()[1]["window_id"] for _ in range(12)]
    assert seq_a != seq_b


def test_reset_wraps_past_the_last_episode(cfg, features):
    """SB3 calls reset on a schedule of its own; running out must reshuffle
    rather than raise."""
    frame = make_frame([("A", True, 2), ("B", False, 2)], features)
    env = FootprintEnv(cfg, frame)
    seen = [env.reset()[1]["window_id"] for _ in range(6)]
    assert len(seen) == 6


# --------------------------------------------------------------------------
# Gymnasium compliance
# --------------------------------------------------------------------------
def test_it_passes_the_gymnasium_api_checker(cfg, features):
    """SB3 will refuse a non-compliant env; better to find out here."""
    from gymnasium.utils.env_checker import check_env

    frame = make_frame([(f"W{i}", i % 2 == 0, 4) for i in range(4)], features)
    check_env(FootprintEnv(cfg, frame), skip_render_check=True)
