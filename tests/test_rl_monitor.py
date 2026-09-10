"""P6-04 — the arithmetic that should be checked before a GPU is warmed up.

`reward_landscape` answers a question no reward curve can: what is the agent
able to gain at all? If always-FLAG and always-WAIT pay the same, no amount of
training teaches a policy *when* to flag, because there is nothing to climb.
That is arithmetic, not a hypothesis, and these tests pin it against the real
config so a reward-table edit that flattens the landscape fails here rather
than after a week of tuning.
"""

import pytest

from src.rl.monitor import (Payoffs, action_summary, describe,
                            reward_landscape, suggest_r_wait)
from src.utils.config import load_config


@pytest.fixture
def cfg():
    return load_config()


TRAIN_BASE_RATE = 0.25          # the 3:1 sample: one window in four is positive


# --------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------
def test_always_flag_matches_the_hand_calculation(cfg):
    p = reward_landscape(cfg, base_rate=0.25)
    r = cfg["reward"]
    expected = 0.25 * (r["r_correct_flag"] + r["r_early_bonus"]) + \
        0.75 * r["r_false_alarm"]
    assert p.always_flag == pytest.approx(expected)


def test_always_wait_matches_the_hand_calculation(cfg):
    """(h - 1) step costs, not h.

    `FootprintEnv.step` pays `r_wait` only on a WAIT that leaves the episode
    running; the WAIT that runs an h-bar window out returns `r_missed` on a
    positive and 0.0 on a quiet one, with no step cost beside it. So waiting
    out a window costs h-1 steps. The env's reading is the coherent one and
    `reward_landscape` was the one that was off by a step, which overstated
    always-WAIT by 0.005 at the configured table — no verdict moves, but the
    derivation recorded in `config.reward` has to be reproducible from this
    function.
    """
    p = reward_landscape(cfg, base_rate=0.25)
    r = cfg["reward"]
    cost = (cfg["decision"]["horizon_hours"] - 1) * r["r_wait"]
    expected = 0.25 * (cost + r["r_missed"]) + 0.75 * cost
    assert p.always_wait == pytest.approx(expected)


def test_the_wait_cost_is_the_one_the_env_actually_pays(cfg):
    """Tied to the env rather than to a second copy of the arithmetic.

    A quiet window is walked to its end with WAIT and the rewards summed. That
    total is what always-WAIT earns on a quiet window, and it is what
    `reward_landscape` must report at base rate 0 — otherwise the landscape
    describes a reward table nobody is training against.
    """
    import pandas as pd

    from src.rl.env import WAIT, FootprintEnv, observation_features
    from src.utils.timeutils import date_str_to_ts

    bars = 5
    feats = observation_features(cfg)
    base = date_str_to_ts("2025-10-01")
    rows = []
    for h in range(bars):
        row = {"window_id": "Q0", "ticker": "T0", "ts_utc": base + h * 3600,
               "t0_utc": None, "is_scheduled": None, "item_code": None}
        row.update({f: 0.0 for f in feats})
        rows.append(row)
    frame = pd.DataFrame(rows).astype({
        "window_id": "string", "ticker": "string", "ts_utc": "Int64",
        "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string"})

    env = FootprintEnv(cfg, frame)
    env.reset()
    total, terminated = 0.0, False
    while not terminated:
        _, reward, terminated, _, _ = env.step(WAIT)
        total += reward

    landscape = reward_landscape(cfg, base_rate=0.0, horizon=bars)
    assert landscape.always_wait == pytest.approx(total)


def test_the_oracle_beats_both_degenerate_strategies(cfg):
    """If perfect selectivity did not pay more than doing nothing useful, the
    reward would be incapable of rewarding the behaviour wanted."""
    p = reward_landscape(cfg, base_rate=TRAIN_BASE_RATE)
    assert p.oracle > p.always_flag
    assert p.oracle > p.always_wait
    assert p.headroom > 0


def test_a_higher_base_rate_favours_flagging(cfg):
    """A sanity check on the model: if most windows were positive, flagging
    everything would be a reasonable strategy."""
    low = reward_landscape(cfg, base_rate=0.05)
    high = reward_landscape(cfg, base_rate=0.90)
    assert high.always_flag > low.always_flag


# --------------------------------------------------------------------------
# The property that caught the reward defect
# --------------------------------------------------------------------------
def test_the_degenerate_strategies_are_distinguishable(cfg):
    """THE check. At r_wait -0.02 this gap was 0.040 — smaller than PPO's
    batch-to-batch noise — and the first real run duly flagged on 94% of steps.

    A tie here means the reward cannot tell 'flag everything' from 'flag
    nothing', so an agent that cannot yet discriminate drifts to whichever
    extreme its parameters happen to favour. At a 0.46% evaluation base rate
    that default must be WAIT, not FLAG.
    """
    p = reward_landscape(cfg, base_rate=TRAIN_BASE_RATE)
    assert p.degenerate_gap > 0.1, (
        f"always-FLAG {p.always_flag:+.4f} and always-WAIT {p.always_wait:+.4f} "
        f"are within {p.degenerate_gap:.4f} — the reward table cannot "
        f"distinguish them")


def test_waiting_is_the_better_default_when_uncertain(cfg):
    """The safe default has to be the one the reward prefers. Waiting is right
    on ~99.5% of evaluation decision points."""
    p = reward_landscape(cfg, base_rate=TRAIN_BASE_RATE)
    assert p.always_wait > p.always_flag


def test_waiting_is_not_free(cfg):
    """The other side: r_wait exists so an agent cannot sit out every window at
    no cost. Zero would make always-WAIT and the oracle differ only on
    positives."""
    assert cfg["reward"]["r_wait"] < 0


def test_a_flat_landscape_is_reported_as_such(cfg):
    """The diagnostic has to say so in words, not leave a reader to compare
    four numbers."""
    text = describe(reward_landscape(cfg, base_rate=0.25, r_wait=-0.02))
    assert "effectively tied" in text


def test_the_healthy_landscape_reports_no_warning(cfg):
    text = describe(reward_landscape(cfg, base_rate=0.25))
    assert "effectively tied" not in text
    assert "pays nothing" not in text


# --------------------------------------------------------------------------
# The suggestion
# --------------------------------------------------------------------------
def test_the_suggested_step_cost_is_a_stated_share_of_a_false_alarm(cfg):
    """Derived from a criterion that can be argued with, not guessed."""
    horizon = cfg["decision"]["horizon_hours"]
    penalty = abs(cfg["reward"]["r_false_alarm"])
    suggested = suggest_r_wait(cfg, 0.25, share_of_false_alarm=0.125)
    assert abs(suggested) * horizon == pytest.approx(0.125 * penalty, rel=0.01)


def test_the_configured_step_cost_follows_that_criterion(cfg):
    """The value actually in config should sit near the derived one, or the
    criterion is decoration."""
    horizon = cfg["decision"]["horizon_hours"]
    penalty = abs(cfg["reward"]["r_false_alarm"])
    share = abs(cfg["reward"]["r_wait"]) * horizon / penalty
    assert 0.05 <= share <= 0.25, (
        f"waiting out a window costs {share:.1%} of a false alarm; the stated "
        f"criterion is about an eighth")


# --------------------------------------------------------------------------
# The action distribution
# --------------------------------------------------------------------------
def test_action_summary_reports_the_flag_rate():
    s = action_summary({0: 900, 1: 100})
    assert s["n_steps"] == 1000
    assert s["flag_rate"] == pytest.approx(0.1)


def test_action_summary_survives_an_empty_run():
    s = action_summary({})
    assert s["n_steps"] == 0
    assert s["flag_rate"] == 0.0


def test_a_collapsed_policy_is_visible_in_the_summary():
    """The failure mean reward hides: restraint on a quiet window pays 0 and
    quiet windows dominate, so an agent that stopped acting looks healthy."""
    assert action_summary({0: 1000, 1: 0})["flag_rate"] == 0.0
    assert action_summary({0: 0, 1: 1000})["flag_rate"] == 1.0
