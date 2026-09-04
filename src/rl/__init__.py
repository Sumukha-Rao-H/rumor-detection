"""The learned stopping policy — Phase 6.

Built after the baselines on purpose. P5-06 fixed the number to beat, so a
policy scoring 0.09 here is a comparison rather than a bare figure, and a loss
to the z-score threshold is a finding rather than a disappointment.
"""

from src.rl.env import FLAG, WAIT, FootprintEnv, episodes_from_frame, observation_features

__all__ = ["FLAG", "WAIT", "FootprintEnv", "episodes_from_frame",
           "observation_features"]
