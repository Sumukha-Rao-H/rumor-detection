"""What the reward table actually pays, and what the policy actually does.

Two questions reward curves cannot answer, and both matter more than the curve.

**What can the agent possibly gain?** Before training anything, the payoff of
each reference strategy can be computed in closed form from `config.reward`. If
always-FLAG and always-WAIT score within noise of each other, no amount of
training will teach a policy *when* to flag — there is nothing to climb. That
is not a hypothesis; it is arithmetic, and it should be checked before a GPU is
warmed up rather than after a week of tuning.

**What is the policy doing?** Mean reward hides a collapsed policy. Restraint on
a quiet window pays 0 and quiet windows are the overwhelming majority, so an
agent that has stopped acting entirely looks healthy on reward alone. The plan
says it in §6 — *report the action distribution alongside accuracy every time* —
and this module is how.

Usage:
  python -m src.rl.monitor                    # the landscape at the configured table
  python -m src.rl.monitor --r-wait -0.005    # what a different step cost would pay
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.utils.config import load_config


@dataclass(frozen=True)
class Payoffs:
    """Expected episode reward for each reference strategy.

    `oracle` flags every positive at the first hour and never flags a quiet
    window — unreachable in practice, but it is the ceiling the gradient has to
    point towards.
    """

    always_flag: float
    always_wait: float
    oracle: float
    base_rate: float
    horizon: int

    @property
    def degenerate_gap(self) -> float:
        """How far apart the two do-nothing-useful strategies sit.

        Near zero means the reward table cannot distinguish them, so a policy
        has almost nothing to gain by learning when to flag.
        """
        return abs(self.always_flag - self.always_wait)

    @property
    def headroom(self) -> float:
        """What selectivity is worth over the better degenerate strategy."""
        return self.oracle - max(self.always_flag, self.always_wait)

    def as_dict(self) -> dict:
        return {"always_flag": self.always_flag, "always_wait": self.always_wait,
                "oracle": self.oracle, "degenerate_gap": self.degenerate_gap,
                "headroom": self.headroom, "base_rate": self.base_rate,
                "horizon": self.horizon}


def reward_landscape(cfg: dict, base_rate: float, horizon: int | None = None,
                     r_wait: float | None = None) -> Payoffs:
    """Closed-form expected reward per episode for the reference strategies.

    `base_rate` is the share of episodes that are positive — the TRAINING
    frame's rate, not the evaluation population's, because this describes what
    the agent experiences while learning.
    """
    r = dict(cfg["reward"])
    if r_wait is not None:
        r["r_wait"] = r_wait
    h = int(horizon if horizon is not None else cfg["decision"]["horizon_hours"])
    p = float(base_rate)

    # Flag at the first hour: the whole early bonus, and no waiting cost.
    always_flag = p * (r["r_correct_flag"] + r["r_early_bonus"]) + \
        (1 - p) * r["r_false_alarm"]
    # Wait through the window: the step cost, plus a miss on positives.
    #
    # (h - 1), not h. The env pays `r_wait` only on a WAIT that leaves the
    # episode running. The WAIT that runs the window out returns `r_missed` on
    # a positive and 0.0 on a quiet one, and no step cost alongside it — see
    # `FootprintEnv.step`. So sitting out an h-bar window costs h-1 steps, not
    # h. The env's reading is the coherent one and this function was the one
    # that was off. The arithmetic difference is 0.005 at the configured table
    # and changes no verdict, but the derivation recorded in `config.reward`
    # has to be reproducible from here or it is not a derivation.
    wait_cost = (h - 1) * r["r_wait"]
    always_wait = p * (wait_cost + r["r_missed"]) + (1 - p) * wait_cost
    # Perfect selectivity: flag positives immediately, wait out the quiet ones.
    oracle = p * (r["r_correct_flag"] + r["r_early_bonus"]) + (1 - p) * wait_cost

    return Payoffs(always_flag=always_flag, always_wait=always_wait,
                   oracle=oracle, base_rate=p, horizon=h)


def suggest_r_wait(cfg: dict, base_rate: float, horizon: int | None = None,
                   share_of_false_alarm: float = 0.125) -> float:
    """A step cost that leaves room to learn, derived rather than guessed.

    The criterion, stated so it can be argued with: **waiting through a whole
    window should cost a stated fraction of one false alarm** — enough that
    waiting is not free, little enough that it is not nearly as bad as being
    wrong.

    At the default 0.125, sitting out a full 48-bar window costs an eighth of a
    false alarm. `r_wait` was originally -0.02, which over 48 bars is -0.96
    against a -2.0 false alarm: **48%**, which is what flattened the landscape.
    """
    h = int(horizon if horizon is not None else cfg["decision"]["horizon_hours"])
    penalty = abs(float(cfg["reward"]["r_false_alarm"]))
    return -round(share_of_false_alarm * penalty / h, 5)


def action_summary(counts: dict) -> dict:
    """Turn raw action counts into the line the plan asks for in §6."""
    total = sum(counts.values()) or 1
    flags = counts.get(1, 0)
    return {"n_steps": sum(counts.values()), "n_flag": flags,
            "n_wait": counts.get(0, 0), "flag_rate": flags / total}


def describe(payoffs: Payoffs) -> str:
    """A readable landscape, with the verdict spelled out."""
    lines = [
        f"base rate {payoffs.base_rate:.1%}, horizon {payoffs.horizon} bars",
        f"  always-FLAG   {payoffs.always_flag:+.4f}",
        f"  always-WAIT   {payoffs.always_wait:+.4f}",
        f"  oracle        {payoffs.oracle:+.4f}",
        f"  degenerate gap {payoffs.degenerate_gap:.4f}"
        f"   headroom {payoffs.headroom:+.4f}",
    ]
    if payoffs.degenerate_gap < 0.1:
        lines.append("  ⛔ the two degenerate strategies are effectively tied — "
                     "a policy has almost nothing to gain by learning WHEN to "
                     "flag")
    if payoffs.headroom <= 0:
        lines.append("  ⛔ selectivity pays nothing over the better degenerate "
                     "strategy — the reward cannot reward the behaviour wanted")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-rate", type=float, default=0.25,
                    help="share of TRAINING episodes that are positive "
                         "(3:1 sampling gives 0.25)")
    ap.add_argument("--r-wait", type=float, default=None,
                    help="override the configured step cost")
    ap.add_argument("--sweep", action="store_true",
                    help="show the landscape across a range of step costs")
    args = ap.parse_args()

    cfg = load_config()
    if args.sweep:
        print(f"{'r_wait':>9} {'wait/window':>12} {'aFLAG':>9} {'aWAIT':>9} "
              f"{'oracle':>9} {'gap':>8} {'headroom':>9}")
        for rw in (-0.02, -0.015, -0.01, -0.0075, -0.005, -0.0025, -0.001):
            p = reward_landscape(cfg, args.base_rate, r_wait=rw)
            print(f"{rw:9.4f} {rw*p.horizon:12.3f} {p.always_flag:9.4f} "
                  f"{p.always_wait:9.4f} {p.oracle:9.4f} "
                  f"{p.degenerate_gap:8.4f} {p.headroom:9.4f}")
        print(f"\nsuggested r_wait at 12.5% of a false alarm: "
              f"{suggest_r_wait(cfg, args.base_rate)}")
        return

    print(describe(reward_landscape(cfg, args.base_rate, r_wait=args.r_wait)))


if __name__ == "__main__":
    main()
