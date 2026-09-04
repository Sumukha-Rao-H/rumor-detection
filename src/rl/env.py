"""The decision problem as a Gymnasium environment.

One **episode** is one decision window. One **step** is one trading hour. The
agent sees the feature vector for that hour and chooses WAIT or FLAG; the
episode ends the moment it flags, or when the window runs out at t0.

This is the first module that treats the task as *sequential*. Every Phase 5
baseline scored each hour independently and let the harness take the first
crossing. Here the stopping decision is the decision, which is the whole
premise of Phase 6 — and if it turns out not to help, P5-06's table is what
makes that a finding rather than a disappointment.

Steps are bars, and bars are trading hours
------------------------------------------
The reward pays a bonus for flagging early, and the code standards require lead
time in trading hours, never wall-clock. Normally that means calendar
arithmetic. Here it does not, and the reason is worth stating because it looks
like a shortcut.

`build_matrix` slices windows by BARS, and `features.py` explains why: bars
exist only while the market is open, so a positional offset is automatically a
trading-time offset. Within a window, **step index is trading hours** by
construction. The early bonus can scale with steps remaining without ever
touching a calendar, and rule 3 still holds.

It inherits P5-05's feature exclusion
--------------------------------------
Training episodes come from the same 3:1 sampled negatives that taught gradient
boosting to read `days_since_last_8k` instead of the market. `quiet_gap_hours`
is 168 h, so a training negative is by construction never within seven days of
a filing — 0.1% of them, against 18.9% of training positives — while 31.5% of
*evaluation* negatives are. An agent handed those columns would learn the
sampling procedure exactly as the tree did, and its result would be worthless.

So the observation uses the same feature list, honouring the same
`baselines.gradient_boosting.exclude_features` knob. This is the module the
tracker's risk entry was written for.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.baselines.gradient_boosting import FEATURES

WAIT, FLAG = 0, 1


def observation_features(cfg: dict) -> tuple[str, ...]:
    """The columns the agent sees, after the P5-05 exclusion.

    Deliberately reads the *same* config knob gradient boosting reads rather
    than keeping a second list, so the two cannot drift apart and quietly
    disagree about what is safe to learn from.
    """
    excluded = set(cfg.get("baselines", {}).get("gradient_boosting", {})
                   .get("exclude_features", ()))
    return tuple(f for f in FEATURES if f not in excluded)


def episodes_from_frame(frame: pd.DataFrame,
                        features: tuple[str, ...]) -> list[dict]:
    """Split a contract-shaped feature frame into per-window episodes.

    Each episode carries its observation matrix, whether it is positive, and
    its ticker and t0 for reporting. Rows are sorted by `ts_utc` on ingest: the
    env serves them positionally, so the entire no-lookahead property rests on
    that order being right rather than being assumed.
    """
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing feature column(s) {missing}")

    ordered = frame.sort_values(["window_id", "ts_utc"], kind="stable")
    episodes = []
    for window_id, block in ordered.groupby("window_id", sort=False):
        episodes.append({
            "window_id": str(window_id),
            "ticker": str(block["ticker"].iloc[0]),
            "t0_utc": (None if pd.isna(block["t0_utc"].iloc[0])
                       else int(block["t0_utc"].iloc[0])),
            "is_positive": bool(block["t0_utc"].notna().iloc[0]),
            "ts_utc": block["ts_utc"].to_numpy(dtype="int64"),
            "obs": block.loc[:, list(features)].to_numpy(dtype="float64"),
        })
    return episodes


class FootprintEnv(gym.Env):
    """Stop-or-wait over one decision window at a time.

    Actions: 0 = WAIT, 1 = FLAG. Flagging ends the episode.

    Rewards come from `config.reward` and nowhere else:

    ============================  ==========================================
    outcome                       value
    ============================  ==========================================
    FLAG on a positive window     `r_correct_flag` + `r_early_bonus` * share
                                  of the window still remaining
    FLAG on a quiet window        `r_false_alarm`
    WAIT                          `r_wait`, each hour
    reached t0 without flagging   `r_missed`
    reached the end of a quiet
    window without flagging       0.0
    ============================  ==========================================

    `r_false_alarm` (-2.0) is twice the magnitude of `r_correct_flag` (+1.0) on
    purpose: false alarms are exactly what an alert budget rations, so the
    reward has to price them above hits. The opposite failure — a policy that
    collapses to never flagging — is what `r_wait` and `r_missed` push against,
    and P6-04 watches the action distribution for it because reward alone will
    not reveal it.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, frame: pd.DataFrame,
                 features: tuple[str, ...] | None = None,
                 seed: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.reward_table = cfg["reward"]
        rl = cfg.get("rl") or {}
        self.obs_clip = float(rl.get("obs_clip", 10.0))
        self._seed = rl.get("seed", 42) if seed is None else seed

        self.features = features or observation_features(cfg)
        self.episodes = episodes_from_frame(frame, self.features)
        if not self.episodes:
            raise ValueError("no episodes — the frame holds no windows")

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-self.obs_clip, high=self.obs_clip,
            shape=(len(self.features),), dtype=np.float32)

        self._rng = np.random.default_rng(self._seed)
        self._order = self._rng.permutation(len(self.episodes))
        self._cursor = 0
        self._episode: dict | None = None
        self._step = 0

    # -- observations ----------------------------------------------------
    def _clean(self, row: np.ndarray) -> np.ndarray:
        """NaN to 0.0, then clip.

        Every surviving feature is a centred quantity — z-scores and returns
        sit near zero — so 0.0 is the neutral reading rather than a fabricated
        observation. No imputer is fitted, which also avoids the leakage route
        the standards name: an imputer fitted across splits carries validation
        statistics into training.

        Clipping comes after, so the zero sentinel is never clipped away.
        """
        cleaned = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(cleaned, -self.obs_clip,
                       self.obs_clip).astype(np.float32)

    def _observe(self) -> np.ndarray:
        return self._clean(self._episode["obs"][self._step])

    def _info(self) -> dict:
        return {
            "window_id": self._episode["window_id"],
            "ticker": self._episode["ticker"],
            "is_positive": self._episode["is_positive"],
            "step": self._step,
            "episode_length": len(self._episode["obs"]),
            "ts_utc": int(self._episode["ts_utc"][min(
                self._step, len(self._episode["ts_utc"]) - 1)]),
        }

    # -- gym API ---------------------------------------------------------
    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Start the next episode.

        Windows are visited in a seeded shuffle. Timestamp order would let the
        agent watch a market regime evolve in exactly the sequence it occurred,
        which is a quiet way for training order to carry information the agent
        will not have in deployment. Wrapping past the last episode reshuffles
        rather than raising, because SB3 calls `reset` on a schedule of its own.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._order = self._rng.permutation(len(self.episodes))
            self._cursor = 0

        if self._cursor >= len(self._order):
            self._order = self._rng.permutation(len(self.episodes))
            self._cursor = 0

        self._episode = self.episodes[self._order[self._cursor]]
        self._cursor += 1
        self._step = 0
        return self._observe(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._episode is None:
            raise RuntimeError("step() before reset()")

        rt = self.reward_table
        length = len(self._episode["obs"])
        positive = self._episode["is_positive"]
        info = self._info()

        if int(action) == FLAG:
            if positive:
                # Share of the window still ahead. Step index IS trading hours
                # here (see the module docstring), so this is a trading-hours
                # bonus without any calendar arithmetic. A one-bar window gives
                # the full bonus rather than dividing by zero.
                remaining = (length - 1 - self._step) / max(length - 1, 1)
                reward = float(rt["r_correct_flag"]) + \
                    float(rt["r_early_bonus"]) * remaining
            else:
                reward = float(rt["r_false_alarm"])
            info["outcome"] = "correct_flag" if positive else "false_alarm"
            return self._observe(), reward, True, False, info

        # WAIT
        self._step += 1
        if self._step >= length:
            # The window ran out. A positive was missed; restraint on a quiet
            # window is the baseline, not an achievement, so it pays nothing.
            reward = float(rt["r_missed"]) if positive else 0.0
            info["outcome"] = "missed" if positive else "correct_wait"
            info["step"] = length
            last = self._clean(self._episode["obs"][length - 1])
            return last, reward, True, False, info

        info["outcome"] = "wait"
        info["step"] = self._step
        return self._observe(), float(rt["r_wait"]), False, False, info
