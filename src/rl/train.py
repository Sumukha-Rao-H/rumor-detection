"""Train the stopping policy, and record enough to reproduce the run.

Reproducibility is the deliverable here, not a nicety: P6-03's done-when is
"the run is reproducible from the recorded seed". A result nobody can regenerate
is a result nobody can check, and this project is graded on whether its numbers
can be defended.

So every stochastic component is seeded from one place, and a manifest lands
beside the model recording the seed, the config actually used, the git commit,
a fingerprint of the training data, and library versions. A seed reproduces a
run only against the same data and the same libraries — "same data" has to be
checkable rather than assumed, and SB3 and torch move their defaults between
releases.

Two choices worth reading before the code
------------------------------------------
**`gamma = 1.0`, not the usual 0.99.** Over a 48-step episode, 0.99 multiplies
a terminal reward by 0.62, which makes flagging early worth more than flagging
late — a second, unstated preference for early flagging stacked on top of
`r_early_bonus`, which already says exactly that and is visible in config where
it can be argued with. Episodes here are short and finite, so undiscounted
returns are well defined and there is no convergence reason to discount.
Leaving gamma at 1.0 keeps the reward table the only thing deciding timing.

**CPU by default.** The observation is 11 floats and the network is two small
hidden layers; at that size GPU kernel-launch overhead dominates and CPU is
usually faster, which SB3 warns about directly. `rl.device` selects, so a
machine that can see a card can try it.

Usage:
  python -m src.rl.train                       # full run, config defaults
  python -m src.rl.train --timesteps 20000     # a short smoke run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.rl.env import FootprintEnv, observation_features
from src.utils.config import load_config
from src.utils.timeutils import utc_now_ts


def _git_state() -> dict:
    """The commit a run was trained at, and whether the tree was dirty.

    A dirty tree means the recorded commit does not fully describe the code
    that ran, so it is recorded rather than silently ignored.
    """
    def run(*args: str) -> str:
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:                      # git absent, or not a repo
            return ""

    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def data_fingerprint(frame: pd.DataFrame) -> dict:
    """Identify the training data a seed was used against.

    A seed reproduces a run only for the same data. Hashing the sorted window
    ids plus the shape is enough to tell two training sets apart without
    storing the set itself.
    """
    ids = ",".join(sorted(frame["window_id"].astype(str).unique()))
    return {
        "rows": int(len(frame)),
        "windows": int(frame["window_id"].nunique()),
        "positives": int(frame.loc[frame["t0_utc"].notna(), "window_id"].nunique()),
        "window_id_sha256": hashlib.sha256(ids.encode()).hexdigest()[:16],
    }


def write_manifest(path: Path, cfg: dict, seed: int, frame: pd.DataFrame,
                   extra: dict | None = None) -> dict:
    """Everything needed to regenerate this run, written before training.

    Written first on purpose: a run that crashes half way is still identifiable,
    and an unidentifiable run in a results directory is worse than no run.
    """
    import stable_baselines3
    import torch

    manifest = {
        "seed": seed,
        "created_utc": utc_now_ts(),
        "rl_config": dict(cfg.get("rl") or {}),
        "reward": dict(cfg["reward"]),
        "features": list(observation_features(cfg)),
        "data": data_fingerprint(frame),
        "git": _git_state(),
        "versions": {
            "python": platform.python_version(),
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _callback_class():
    """Imported lazily so the module loads without SB3 present."""
    from stable_baselines3.common.callbacks import BaseCallback

    class ActionDistributionCallback(BaseCallback):
        """Log how often the policy flags, not just what it earns.

        The failure this exists to reveal is a policy collapsing to always-WAIT.
        It looks *fine* on reward — restraint on a quiet window pays 0, and
        quiet windows are the overwhelming majority — so mean reward can look
        healthy while the agent has stopped acting entirely. P6-04's whole job
        is watching this, and the tracker already carries a risk that the
        report's `degenerate` column cannot detect it either.
        """

        def __init__(self) -> None:
            super().__init__()
            self.counts = {0: 0, 1: 0}

        def _on_step(self) -> bool:
            for action in np.atleast_1d(self.locals.get("actions", [])):
                self.counts[int(action)] = self.counts.get(int(action), 0) + 1
            total = sum(self.counts.values())
            if total and self.n_calls % 100 == 0:
                flag_rate = self.counts.get(1, 0) / total
                self.logger.record("policy/flag_rate", flag_rate)
                self.logger.record("policy/n_flag", self.counts.get(1, 0))
                self.logger.record("policy/n_wait", self.counts.get(0, 0))
            return True

        @property
        def flag_rate(self) -> float:
            total = sum(self.counts.values())
            return (self.counts.get(1, 0) / total) if total else float("nan")

    return ActionDistributionCallback


def build_training_frame(cfg: dict, conn) -> pd.DataFrame:
    """The 3:1 sampled TRAIN frame — the same one gradient boosting was fitted on.

    Deliberately shared, so the two learned models see identical data and their
    comparison in P6-06 is about method rather than about diet.
    """
    from src.baselines.compare import build_training_frame as shared
    return shared(cfg, conn)


def train(cfg: dict, frame: pd.DataFrame, conn=None,
          seed: int | None = None, timesteps: int | None = None,
          run_dir: Path | None = None, verbose: int = 0):
    """Train a policy on `frame`. Returns (model, manifest, callback)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv

    rl = cfg.get("rl") or {}
    seed = int(rl.get("seed", 42) if seed is None else seed)
    timesteps = int(rl.get("total_timesteps", 300_000)
                    if timesteps is None else timesteps)
    algo = rl.get("algo", "PPO")
    if algo != "PPO":
        raise SystemExit(
            f"rl.algo={algo!r} is not implemented. Only PPO is wired up; see "
            f"impl/P6-03 for why DQN was rejected and what changing it entails.")

    if conn is not None:
        from src.pipeline import split
        split.assert_not_test(cfg, conn, frame["ts_utc"], "rl.train")

    run_dir = Path(run_dir or Path(rl.get("runs_dir", "data/runs")) /
                   f"ppo-seed{seed}-{utc_now_ts()}")
    manifest = write_manifest(run_dir / "manifest.json", cfg, seed, frame,
                              extra={"total_timesteps": timesteps})

    # One seeding call before anything stochastic is constructed, so the
    # env's shuffle, the network init and the rollout sampling all descend
    # from the recorded number.
    set_random_seed(seed)

    n_envs = int(rl.get("n_envs", 4))

    def make(rank: int):
        def _init():
            # Distinct per worker: identical seeds would make every worker
            # visit the same episodes in the same order, quietly reducing the
            # effective batch to one env's worth of experience.
            return Monitor(FootprintEnv(cfg, frame, seed=seed + rank))
        return _init

    venv = DummyVecEnv([make(i) for i in range(n_envs)])
    venv.seed(seed)

    model = PPO(
        "MlpPolicy", venv,
        learning_rate=float(rl.get("learning_rate", 3e-4)),
        n_steps=int(rl.get("n_steps", 512)),
        batch_size=int(rl.get("batch_size", 256)),
        # 1.0, not 0.99 — see the module docstring. Discounting would add a
        # second, unstated preference for flagging early on top of the reward
        # table's explicit r_early_bonus.
        gamma=float(rl.get("gamma", 1.0)),
        policy_kwargs={"net_arch": list(rl.get("net_arch", [64, 64]))},
        tensorboard_log=str(run_dir / "tb"),
        device=rl.get("device", "cpu"),
        seed=seed,
        verbose=verbose,
    )

    callback = _callback_class()()
    started = time.time()
    model.learn(total_timesteps=timesteps, callback=callback,
                progress_bar=False)
    elapsed = time.time() - started

    model.save(run_dir / "policy")
    manifest["elapsed_s"] = round(elapsed, 1)
    manifest["flag_rate"] = callback.flag_rate
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    return model, manifest, callback


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    print("building the training frame (3:1 sample, TRAIN split only)...")
    frame = build_training_frame(cfg, conn)
    print(f"  {len(frame):,} rows / {frame.window_id.nunique():,} windows")

    model, manifest, callback = train(
        cfg, frame, conn=conn, seed=args.seed, timesteps=args.timesteps,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        verbose=args.verbose)

    print(f"\ntrained in {manifest['elapsed_s']}s on {manifest['rl_config'].get('device')}")
    print(f"flag rate during training: {callback.flag_rate:.4f}")
    print(f"seed {manifest['seed']}; manifest and policy under the run directory")


if __name__ == "__main__":
    main()
