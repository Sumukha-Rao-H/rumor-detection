"""Classical baselines — the number the learned policy has to beat.

Built before the agent on purpose: without a tuned baseline a null result in
Phase 6 means nothing, and a strawman makes the whole comparison worthless.
"""

from src.baselines.base import Baseline, unscoreable_floor

__all__ = ["Baseline", "unscoreable_floor"]
