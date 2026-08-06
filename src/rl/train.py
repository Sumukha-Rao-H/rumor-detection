"""Train and evaluate the verification policy — plan §9.

PPO is the primary algorithm, DQN the comparison. The network is deliberately
tiny (§9: 418 -> 256 -> 128 -> 3, ~140k params) because the expensive parts —
the sentence embedding and the sentiment model — were already spent offline in
§7. What trains here is only the decision rule, which is why this runs on CPU
in minutes and does not need the 3050 at all.

Rollouts record p(TRUE) from the policy's own action distribution, renormalised
over the two COMMIT actions, so the calibration metrics in §11.2 have something
to score. A random or always-commit policy can be rolled out through the same
path, which is what makes the baseline comparison honest: identical env,
identical scoring, only the policy differs.

Usage:
  python -m src.rl.train --algo ppo --timesteps 50000
  python -m src.rl.train --algo dqn --timesteps 25000
  python -m src.rl.train --algo random --eval-only     # baseline, no training
  python -m src.rl.train --eval-only --model models/ppo.zip
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import Rollout, format_summary, summarize
from src.rl.env import COMMIT_FALSE, COMMIT_TRUE, WAIT, EventStore, RumorVerificationEnv
from src.utils.config import load_config

log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("models")


# ------------------------------------------------------------------ policies

class RandomPolicy:
    """Uniform over the three actions — the floor any agent must clear."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs) -> tuple[int, float | None]:
        return int(self.rng.integers(3)), None


class ImmediatePolicy:
    """Commits at t=0 on a coin flip — the 'static classifier' strawman (§11.1).

    It exists to make the point that earliness without accuracy is worthless:
    its Δ is maximal and its accuracy is chance.
    """

    name = "immediate"

    def __init__(self, seed: int = 0, p_true: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.p_true = p_true

    def act(self, obs) -> tuple[int, float | None]:
        draw = float(self.rng.random())
        return (COMMIT_TRUE if draw < self.p_true else COMMIT_FALSE), self.p_true


class SB3Policy:
    """A trained stable-baselines3 model, with p(TRUE) where available."""

    def __init__(self, model, name: str):
        self.model = model
        self.name = name

    def act(self, obs) -> tuple[int, float | None]:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action), self._p_true(obs)

    def _p_true(self, obs) -> float | None:
        """Renormalise the policy's commit mass into a probability of TRUE.

        PPO exposes a categorical distribution; DQN does not, so it reports
        None and the calibration metrics simply skip it rather than inventing
        a confidence that was never computed.
        """
        try:
            import torch
            tensor, _ = self.model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(tensor)
                probs = dist.distribution.probs[0].cpu().numpy()
            commit = probs[COMMIT_TRUE] + probs[COMMIT_FALSE]
            return float(probs[COMMIT_TRUE] / commit) if commit > 0 else None
        except (AttributeError, ImportError, IndexError):
            return None


# ------------------------------------------------------------------ rollouts

def rollout(env: RumorVerificationEnv, policy, episodes: int | None = None
            ) -> list[Rollout]:
    """Walk a split once, recording what the policy did on every event."""
    episodes = episodes or len(env.store)
    out: list[Rollout] = []
    for _ in range(episodes):
        obs, info = env.reset()
        total, actions, p_true = 0.0, [], None
        terminated = truncated = False
        while not (terminated or truncated):
            action, p = policy.act(obs)
            if action != WAIT:
                p_true = p
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
        out.append(Rollout(
            event_id=info["event_id"], label=info["label"],
            committed="pred" in info, pred=info.get("pred"), p_true=p_true,
            t_commit_hours=info.get("t") if "pred" in info else None,
            delta_hours=info.get("delta_hours"), reward=total, actions=actions))
    return out


# ------------------------------------------------------------------ training

def build_model(algo: str, env, cfg: dict, seed: int, tensorboard: str | None):
    from stable_baselines3 import DQN, PPO
    net = dict(net_arch=[256, 128])
    if algo == "ppo":
        return PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
                   batch_size=256, gamma=0.99, ent_coef=0.01, seed=seed,
                   policy_kwargs=net, tensorboard_log=tensorboard, verbose=0)
    if algo == "dqn":
        return DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=100_000,
                   batch_size=128, gamma=0.99, exploration_fraction=0.2,
                   exploration_final_eps=0.05, seed=seed, policy_kwargs=net,
                   tensorboard_log=tensorboard, verbose=0)
    raise SystemExit(f"unknown algo {algo!r}")


def load_policy(algo: str, path: Path | None, env, seed: int):
    if algo == "random":
        return RandomPolicy(seed)
    if algo == "immediate":
        return ImmediatePolicy(seed)
    from stable_baselines3 import DQN, PPO
    cls = PPO if algo == "ppo" else DQN
    if path is None or not path.exists():
        raise SystemExit(f"no model at {path} — train one first")
    return SB3Policy(cls.load(path, env=env), algo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", default="ppo",
                        choices=["ppo", "dqn", "random", "immediate"])
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model", help="path to load/save (default: models/<algo>.zip)")
    parser.add_argument("--events", help="parquet (default: paths.events)")
    parser.add_argument("--tensorboard", help="log dir for tensorboard")
    parser.add_argument("--out", help="write the eval summary as JSON")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config()
    frame = pd.read_parquet(args.events or cfg["paths"]["events"])
    states = cfg["paths"]["states"]

    train_env = RumorVerificationEnv(EventStore(states, frame, "train"), cfg)
    eval_store = EventStore(states, frame, args.eval_split)
    eval_env = RumorVerificationEnv(eval_store, cfg, sequential=True)

    model_path = Path(args.model) if args.model else DEFAULT_MODEL_DIR / f"{args.algo}.zip"

    if args.algo in ("random", "immediate"):
        policy = load_policy(args.algo, None, eval_env, args.seed)
    elif args.eval_only:
        policy = load_policy(args.algo, model_path, eval_env, args.seed)
    else:
        log.info("training %s for %d timesteps on %d train events",
                 args.algo.upper(), args.timesteps, len(train_env.store))
        model = build_model(args.algo, train_env, cfg, args.seed, args.tensorboard)
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        log.info("saved %s", model_path)
        policy = SB3Policy(model, args.algo)

    rollouts = rollout(eval_env, policy)
    stats = summarize(rollouts)
    stats["policy"] = args.algo
    stats["split"] = args.eval_split
    log.info("\n--- %s on %s (%d events) ---\n%s", args.algo.upper(),
             args.eval_split, len(eval_store), format_summary(stats))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(stats, indent=2, default=float))
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
