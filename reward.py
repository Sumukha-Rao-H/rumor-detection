from typing import Optional

class RewardCalculator:
    """Calculates rewards for RL agent predictions."""
    
    def calculate_reward(self, predicted_action: int, true_label: Optional[int] = None) -> float:
        """
        Calculates the reward for a given action and true label.
        
        Args:
            predicted_action: The action predicted by the agent
            true_label: The true label (if available)
        
        Returns:
            Reward value (float)
        """
        if true_label is None:
            # No ground truth available, return neutral reward
            return 0.0
        
        # Return positive reward for correct predictions, negative for incorrect
        if predicted_action == true_label:
            return 1.0
        else:
            return -1.0

