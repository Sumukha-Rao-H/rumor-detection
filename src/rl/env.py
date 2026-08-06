"""RumorVerificationEnv — plan §8. One episode is one rumor event.

The MDP the whole project exists to study:

    state    the precomputed row for this hour (418 features, §7)
    actions  WAIT (0), COMMIT_TRUE (1), COMMIT_FALSE (2)
    reward   +r_correct on a right verdict, scaled up the earlier it comes;
             r_wrong on a wrong one; a small r_wait charge per hour; r_timeout
             if the agent never commits.

Three design points are load-bearing and should not be "simplified" away:

  three actions, not two   Committing has to carry a verdict. A two-action
        stop/continue env would need a separate classifier to say what was
        decided, and then earliness and correctness would be tuned separately —
        which is exactly the split this project argues against.
  asymmetric costs         r_wrong is worse than the accumulated r_wait of
        waiting to the horizon. That asymmetry is what makes patience rational
        and is the reason abstention emerges rather than being hand-coded.
  the early bonus          scaled by (1 - t/T_max), so Time Delta Advantage is
        something the agent optimises rather than something we measure
        afterwards and hope for.

States are read from disk (§7), so stepping is an array index — no NLP, no SQL.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:                                            # pragma: no cover - import shim
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:                      # pragma: no cover
    raise SystemExit("gymnasium is required: pip install -r requirements.txt") from exc

log = logging.getLogger(__name__)

WAIT, COMMIT_TRUE, COMMIT_FALSE = 0, 1, 2
ACTION_NAMES = {WAIT: "WAIT", COMMIT_TRUE: "COMMIT_TRUE", COMMIT_FALSE: "COMMIT_FALSE"}
HOUR = 3600


@dataclass
class Episode:
    """One event's precomputed states and its answer."""
    event_id: str
    X: np.ndarray
    label: int
    t0_utc: int
    t_official_utc: int

    @property
    def T(self) -> int:
        return int(self.X.shape[0])


class EventStore:
    """The .npz files for one split, standardised with the train-fit scaler."""

    def __init__(self, states_dir: str | Path, frame, split: str | None = None):
        self.dir = Path(states_dir)
        self.split = split
        rows = frame if split is None else frame[frame["split"] == split]
        self.events: list[Episode] = []
        scaler = self._load_scaler()
        for _, row in rows.iterrows():
            path = self.dir / f"{row['event_id']}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            X = data["X"].astype(np.float32)
            if scaler is not None:
                mean, std, offset = scaler
                X[:, offset:] = (X[:, offset:] - mean) / std
            self.events.append(Episode(
                event_id=row["event_id"], X=X, label=int(row["label"]),
                t0_utc=int(row["t0_utc"]),
                t_official_utc=int(row["t_official_utc"] or 0)))
        if not self.events:
            raise SystemExit(
                f"no state files for split={split!r} in {self.dir}. "
                f"Run: python -m src.pipeline.features")

    def _load_scaler(self):
        path = self.dir / "_scaler.npz"
        if not path.exists():
            return None
        data = np.load(path)
        return data["mean"], data["std"], int(data["offset"])

    @property
    def dim(self) -> int:
        return int(self.events[0].X.shape[1])

    def __len__(self) -> int:
        return len(self.events)


class RumorVerificationEnv(gym.Env):
    """Hourly WAIT/COMMIT decisions over one rumor event."""

    metadata = {"render_modes": []}

    def __init__(self, store: EventStore, cfg: dict, sequential: bool = False):
        super().__init__()
        self.store = store
        self.rcfg = cfg["reward"]
        self.t_max = int(self.rcfg["T_max"])
        self.sequential = sequential          # eval walks the split in order
        self._cursor = 0
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(store.dim,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.episode: Episode | None = None
        self.t = 0

    # -- gym API ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.sequential:
            self.episode = self.store.events[self._cursor % len(self.store)]
            self._cursor += 1
        else:
            idx = self.np_random.integers(len(self.store))
            self.episode = self.store.events[int(idx)]
        self.t = 0
        return self.episode.X[0].copy(), self._info()

    def step(self, action: int):
        action = int(action)
        limit = min(self.episode.T, self.t_max)

        if action == WAIT:
            self.t += 1
            if self.t >= limit:
                # Ran out of hours without deciding: mild, but never free.
                self.t = limit - 1
                return (self.episode.X[self.t].copy(),
                        float(self.rcfg["r_timeout"]), False, True,
                        self._info(timed_out=True))
            return (self.episode.X[self.t].copy(), float(self.rcfg["r_wait"]),
                    False, False, self._info())

        pred = 1 if action == COMMIT_TRUE else 0
        correct = pred == self.episode.label
        if correct:
            earliness = 1.0 - self.t / self.t_max
            reward = float(self.rcfg["r_correct"]) + \
                float(self.rcfg["r_early_bonus"]) * earliness
        else:
            reward = float(self.rcfg["r_wrong"])
        return (self.episode.X[self.t].copy(), reward, True, False,
                self._info(pred=pred, correct=correct))

    # -- bookkeeping -----------------------------------------------------

    def _info(self, pred: int | None = None, correct: bool | None = None,
              timed_out: bool = False) -> dict:
        info = {"event_id": self.episode.event_id, "t": self.t,
                "label": self.episode.label, "timed_out": timed_out}
        if pred is not None:
            info["pred"] = pred
            info["correct"] = bool(correct)
            info["t_commit_utc"] = self.episode.t0_utc + self.t * HOUR
            # Delta: hours between our verdict and the official record. Positive
            # means we got there first, which is the entire point.
            if self.episode.t_official_utc:
                info["delta_hours"] = (
                    self.episode.t_official_utc - info["t_commit_utc"]) / HOUR
        return info


def make_env(cfg: dict, frame, split: str, sequential: bool = False):
    store = EventStore(cfg["paths"]["states"], frame, split)
    return RumorVerificationEnv(store, cfg, sequential=sequential)


def manifest(cfg: dict) -> dict:
    path = Path(cfg["paths"]["states"]) / "_manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}
