"""Scoring a sequential policy — plan §11.2.

Accuracy alone cannot describe this task: a policy that commits instantly and a
policy that waits 40 hours can post the same accuracy while being completely
different systems. So every rollout is scored on three axes:

    correctness   accuracy / F1 over committed verdicts, plus the abstention
                  rate. Abstentions are reported, never quietly dropped —
                  a policy that only answers the easy 10% must look different
                  from one that answers everything.
    calibration   Brier score and ECE. The verdict is meant to be actionable,
                  and an uncalibrated 0.9 is worse than an honest 0.6.
    earliness     Time Delta Advantage, Δ = t_official - t_commit, in hours.
                  Positive means the agent resolved the rumor before the
                  official record. This is the headline number.

Δ is measured against aggregator-visible timestamps, which lag the wire by
minutes — §11.3 requires saying so wherever it is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Rollout:
    """One episode's outcome."""
    event_id: str
    label: int
    committed: bool
    pred: int | None = None
    p_true: float | None = None
    t_commit_hours: int | None = None
    delta_hours: float | None = None
    reward: float = 0.0
    actions: list[int] = field(default_factory=list)


def _committed(rollouts: list[Rollout]) -> list[Rollout]:
    return [r for r in rollouts if r.committed and r.pred is not None]


def confusion(rollouts: list[Rollout]) -> dict:
    done = _committed(rollouts)
    tp = sum(1 for r in done if r.pred == 1 and r.label == 1)
    fp = sum(1 for r in done if r.pred == 1 and r.label == 0)
    tn = sum(1 for r in done if r.pred == 0 and r.label == 0)
    fn = sum(1 for r in done if r.pred == 0 and r.label == 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def accuracy_f1(rollouts: list[Rollout]) -> dict:
    done = _committed(rollouts)
    if not done:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "committed": 0}
    c = confusion(rollouts)
    correct = c["tp"] + c["tn"]
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"accuracy": correct / len(done), "precision": precision,
            "recall": recall, "f1": f1, "committed": len(done)}


def abstention_rate(rollouts: list[Rollout]) -> float:
    return 1.0 - len(_committed(rollouts)) / len(rollouts) if rollouts else 0.0


def brier(rollouts: list[Rollout]) -> float | None:
    """Mean squared error of p(TRUE) against the outcome."""
    scored = [r for r in _committed(rollouts) if r.p_true is not None]
    if not scored:
        return None
    return float(np.mean([(r.p_true - r.label) ** 2 for r in scored]))


def ece(rollouts: list[Rollout], bins: int = 15) -> float | None:
    """Expected calibration error over `bins` equal-width confidence bins."""
    scored = [r for r in _committed(rollouts) if r.p_true is not None]
    if not scored:
        return None
    conf = np.array([r.p_true if r.pred == 1 else 1 - r.p_true for r in scored])
    hit = np.array([float(r.pred == r.label) for r in scored])
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum():
            total += mask.mean() * abs(hit[mask].mean() - conf[mask].mean())
    return float(total)


def time_delta(rollouts: list[Rollout], correct_only: bool = True) -> dict:
    """Δ statistics in hours. Only correct verdicts count by default —
    beating the news with the wrong answer is not an advantage."""
    pool = [r for r in _committed(rollouts) if r.delta_hours is not None
            and (not correct_only or r.pred == r.label)]
    if not pool:
        return {"n": 0, "median": None, "mean": None, "share_ge_24h": None}
    deltas = np.array([r.delta_hours for r in pool], dtype=float)
    return {
        "n": len(pool),
        "median": float(np.median(deltas)),
        "mean": float(np.mean(deltas)),
        "p25": float(np.percentile(deltas, 25)),
        "p75": float(np.percentile(deltas, 75)),
        # The synopsis's stated target.
        "share_ge_24h": float((deltas >= 24).mean()),
        "share_positive": float((deltas > 0).mean()),
    }


def mean_commit_time(rollouts: list[Rollout]) -> float | None:
    done = [r for r in _committed(rollouts) if r.t_commit_hours is not None]
    return float(np.mean([r.t_commit_hours for r in done])) if done else None


def summarize(rollouts: list[Rollout]) -> dict:
    """Everything §11.2 asks for, in one dict."""
    return {
        "episodes": len(rollouts),
        **accuracy_f1(rollouts),
        "abstention_rate": abstention_rate(rollouts),
        "brier": brier(rollouts),
        "ece": ece(rollouts),
        "mean_commit_hours": mean_commit_time(rollouts),
        "mean_reward": float(np.mean([r.reward for r in rollouts])) if rollouts else 0.0,
        "confusion": confusion(rollouts),
        "time_delta": time_delta(rollouts),
    }


def format_summary(stats: dict) -> str:
    """A compact block for logs and the dashboard."""
    td = stats["time_delta"]
    fmt = lambda v, p=3: "n/a" if v is None else f"{v:.{p}f}"   # noqa: E731
    return "\n".join([
        f"episodes         {stats['episodes']}",
        f"committed        {stats['committed']}  "
        f"(abstained {stats['abstention_rate']:.1%})",
        f"accuracy         {fmt(stats['accuracy'])}",
        f"f1               {fmt(stats['f1'])}",
        f"brier            {fmt(stats['brier'])}",
        f"ece              {fmt(stats['ece'])}",
        f"mean commit      {fmt(stats['mean_commit_hours'], 1)} h",
        f"Δ median         {fmt(td['median'], 1)} h  (n={td['n']})",
        f"Δ ≥ 24h          {fmt(td['share_ge_24h'])}",
        f"mean reward      {fmt(stats['mean_reward'])}",
    ])
