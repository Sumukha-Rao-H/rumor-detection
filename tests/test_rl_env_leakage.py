"""P6-02 — the observation at step t must contain nothing from t+1.

The done-when of this task, and the reason it gets its own file rather than
more cases appended to `test_rl_env.py`: this is a leakage suite, and the
project's leakage suites follow one shape — tamper with the future, demand
byte-identity, then prove the check can fail by tampering with the past.

The env is the last place a lookahead could enter Phase 6. The feature matrix
is already covered by P2-08 and P4-13, but those test the *builders*. This
tests the *server*: `FootprintEnv` hands out rows positionally, and a single
off-by-one in the step index would serve row t+1 at step t while every
timestamp remained perfectly ordered and every earlier test stayed green.
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


def make_frame(specs, features) -> pd.DataFrame:
    """Each hour gets a distinct, recognisable value: (window+1) + hour/100.

    Distinct per row on purpose — if the env ever served the wrong row, an
    equality check on a frame of constants would not notice.

    The +1 matters: with a bare window index the first window's first row would
    be exactly 0.0, and multiplying it by the tamper factor would leave it at
    0.0. The guard-on-the-guard test caught precisely that and was right to —
    a fixture whose tamper is a no-op makes a leakage suite look clean.
    """
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
                row[f] = float(i + 1) + h / 100.0
            rows.append(row)
    return pd.DataFrame(rows).astype({
        "window_id": "string", "ticker": "string", "ts_utc": "Int64",
        "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string"})


def observations(cfg, frame, window_id: str, n_steps: int) -> list[np.ndarray]:
    """Replay one episode by window_id, collecting the observations."""
    env = FootprintEnv(cfg, frame)
    index = next(i for i, e in enumerate(env.episodes)
                 if e["window_id"] == window_id)
    env._order = np.array([index])
    env._cursor = 0
    obs, _ = env.reset()
    seen = [obs.copy()]
    for _ in range(n_steps - 1):
        obs, _, terminated, _, _ = env.step(WAIT)
        seen.append(obs.copy())
        if terminated:
            break
    return seen


def tamper(frame: pd.DataFrame, features, window_id: str,
           from_hour: int, factor: float = 50.0) -> pd.DataFrame:
    """Multiply every feature at or after `from_hour` within one window."""
    out = frame.copy()
    block = out.index[out["window_id"] == window_id]
    target = block[from_hour:]
    for f in features:
        out.loc[target, f] = out.loc[target, f] * factor
    return out


# --------------------------------------------------------------------------
# THE test
# --------------------------------------------------------------------------
def test_observations_are_unchanged_by_tampering_with_later_rows(cfg, features):
    """The done-when. If the observation at step t drew on row t+1 in any way,
    multiplying every later row by 50 would move it."""
    specs = [("P0", True, 8)]
    clean = make_frame(specs, features)
    dirty = tamper(clean, features, "P0", from_hour=4)

    before = observations(cfg, clean, "P0", 4)
    after = observations(cfg, dirty, "P0", 4)

    assert len(before) == len(after) == 4
    for i, (a, b) in enumerate(zip(before, after)):
        np.testing.assert_array_equal(a, b, err_msg=f"observation {i} moved")


def test_tampering_with_the_current_row_does_change_the_observation(cfg, features):
    """Guard on the guard. Without this an inert check — one that compared
    nothing, or compared a constant — would pass as clean."""
    specs = [("P0", True, 8)]
    clean = make_frame(specs, features)
    dirty = tamper(clean, features, "P0", from_hour=0)

    before = observations(cfg, clean, "P0", 4)
    after = observations(cfg, dirty, "P0", 4)

    assert not np.array_equal(before[0], after[0])


def test_only_rows_at_or_after_the_tamper_point_move(cfg, features):
    """Sharper than the two above together: the boundary itself is checked, so
    an off-by-one in the slice is caught rather than averaging out."""
    specs = [("P0", True, 8)]
    clean = make_frame(specs, features)
    dirty = tamper(clean, features, "P0", from_hour=5)

    before = observations(cfg, clean, "P0", 8)
    after = observations(cfg, dirty, "P0", 8)

    for i in range(5):
        np.testing.assert_array_equal(before[i], after[i],
                                      err_msg=f"step {i} should be untouched")
    for i in range(5, 8):
        assert not np.array_equal(before[i], after[i]), \
            f"step {i} should have moved"


def test_tampering_with_another_episode_changes_nothing(cfg, features):
    """Cross-episode contamination. Windows are served independently, and one
    window's rows must never reach another's observations."""
    specs = [("P0", True, 6), ("P1", True, 6)]
    clean = make_frame(specs, features)
    dirty = tamper(clean, features, "P1", from_hour=0)

    before = observations(cfg, clean, "P0", 6)
    after = observations(cfg, dirty, "P0", 6)

    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_the_observation_is_exactly_its_own_row(cfg, features):
    """Positional serving, checked directly rather than only through tampering:
    step i must be row i of that window, in ts_utc order."""
    specs = [("P0", True, 5)]
    frame = make_frame(specs, features)
    episodes = episodes_from_frame(frame, features)
    expected = episodes[0]["obs"]

    for i, obs in enumerate(observations(cfg, frame, "P0", 5)):
        np.testing.assert_allclose(obs, expected[i], rtol=1e-6)


# --------------------------------------------------------------------------
# What the observation must NOT carry
# --------------------------------------------------------------------------
def test_the_observation_carries_no_label(cfg, features):
    """A positive and a quiet window with identical features must produce
    identical observations. If they differ, the label is reachable."""
    frame = make_frame([("P0", True, 4), ("N0", False, 4)], features)
    # Give both windows the same feature values.
    for f in features:
        frame.loc[frame["window_id"] == "N0", f] = \
            frame.loc[frame["window_id"] == "P0", f].to_numpy()

    p = observations(cfg, frame, "P0", 4)
    n = observations(cfg, frame, "N0", 4)
    for a, b in zip(p, n):
        np.testing.assert_array_equal(a, b)


def test_the_observation_carries_no_step_index(cfg, features):
    """The agent gets features and nothing else — no counter, no progress bar.

    This matters more than it looks. In the EVALUATION frame a positive is 48
    bars and a negative is 1, so episode LENGTH reveals the label perfectly. A
    memoryless policy cannot exploit that, because it never sees how many steps
    have passed. A recurrent policy could. See the caveat test below.
    """
    frame = make_frame([("P0", True, 6)], features)
    seen = observations(cfg, frame, "P0", 6)
    env = FootprintEnv(cfg, frame)
    assert env.observation_space.shape == (len(features),)
    # No element is a step counter: nothing equals its own index.
    for i, obs in enumerate(seen[1:], start=1):
        assert not np.any(obs == float(i)) or i == 0


def test_the_observation_carries_no_identifiers(cfg, features):
    """window_id, ticker, ts_utc and t0_utc are not inputs. An agent that could
    read a ticker id could memorise which companies leak."""
    env = FootprintEnv(cfg, make_frame([("P0", True, 4)], features))
    assert set(env.features) == set(features)
    for banned in ("window_id", "ticker", "ts_utc", "t0_utc", "is_scheduled",
                   "item_code"):
        assert banned not in env.features


# --------------------------------------------------------------------------
# A caveat worth pinning rather than discovering later
# --------------------------------------------------------------------------
def test_episode_length_reveals_the_label_in_the_evaluation_frame(cfg, features):
    """Documented, not fixed — and safe only because the policy is memoryless.

    P5-03's evaluation population makes a positive a 48-bar episode and a
    negative a single bar. Episode length therefore separates the classes
    perfectly *in that frame*. Training is unaffected: there both classes are
    48-bar windows drawn by `build_quiet_matrix`, so lengths match.

    It is safe today for one reason only — P6-03 specifies a small MLP, which
    has no memory and cannot count steps. Swap in a recurrent policy and this
    becomes a live leak at evaluation time. Pinned here so that swap cannot
    happen quietly.
    """
    eval_shaped = make_frame([("P0", True, 48), ("N0", False, 1)], features)
    episodes = {e["window_id"]: e for e in
                episodes_from_frame(eval_shaped, features)}

    assert len(episodes["P0"]["obs"]) == 48
    assert len(episodes["N0"]["obs"]) == 1
    assert episodes["P0"]["is_positive"] and not episodes["N0"]["is_positive"]

    # Training-shaped frames do not have the asymmetry.
    train_shaped = make_frame([("P0", True, 48), ("N0", False, 48)], features)
    lengths = {len(e["obs"]) for e in
               episodes_from_frame(train_shaped, features)}
    assert lengths == {48}


# --------------------------------------------------------------------------
# Rewards, against the config table, consolidated
# --------------------------------------------------------------------------
def test_every_reward_path_matches_the_config_table(cfg, features):
    """All five outcomes in one place, each read from `config.reward` rather
    than from a literal — so editing the table fails a test instead of quietly
    retraining the agent on different incentives."""
    r = cfg["reward"]

    env = FootprintEnv(cfg, make_frame([("P0", True, 4)], features))
    env.reset()
    assert env.step(FLAG)[1] == pytest.approx(
        r["r_correct_flag"] + r["r_early_bonus"])

    env = FootprintEnv(cfg, make_frame([("N0", False, 4)], features))
    env.reset()
    assert env.step(FLAG)[1] == pytest.approx(r["r_false_alarm"])

    env = FootprintEnv(cfg, make_frame([("P0", True, 4)], features))
    env.reset()
    assert env.step(WAIT)[1] == pytest.approx(r["r_wait"])
    rewards = [env.step(WAIT)[1] for _ in range(3)]
    assert rewards[-1] == pytest.approx(r["r_missed"])

    env = FootprintEnv(cfg, make_frame([("N0", False, 4)], features))
    env.reset()
    quiet = [env.step(WAIT)[1] for _ in range(4)]
    assert quiet[-1] == 0.0


def test_the_reward_before_a_flag_does_not_depend_on_later_rows(cfg, features):
    """WAIT pays a flat step cost. If it ever came to depend on what the window
    does later, the training signal itself would carry the future."""
    clean = make_frame([("P0", True, 8)], features)
    dirty = tamper(clean, features, "P0", from_hour=3)

    def wait_rewards(frame):
        env = FootprintEnv(cfg, frame)
        env._order, env._cursor = np.array([0]), 0
        env.reset()
        return [env.step(WAIT)[1] for _ in range(3)]

    assert wait_rewards(clean) == pytest.approx(wait_rewards(dirty))


def test_termination_is_never_truncation(cfg, features):
    """Every episode here ends naturally — at a FLAG or at t0. Gymnasium's
    `truncated` flag means a time limit cut it short, which never applies, and
    SB3 bootstraps differently on the two."""
    frame = make_frame([("P0", True, 4), ("N0", False, 4)], features)
    env = FootprintEnv(cfg, frame)
    for _ in range(4):
        env.reset()
        terminated = False
        while not terminated:
            _, _, terminated, truncated, _ = env.step(WAIT)
            assert truncated is False
        assert terminated is True
