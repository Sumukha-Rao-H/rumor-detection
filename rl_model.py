import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from reddit_data_fetcher import RedditDataFetcher
from reward import RewardCalculator


class RedditPostState:
    """Represents the state derived from a Reddit post and associated stock data."""
    def __init__(self_obj, post_id: str, text: str, sentiment_score: float, 
                 keywords: List[str], stock_price: float, stock_volume: float,
                 price_change_24h: float, volume_change_24h: float):
        self_obj.post_id = post_id
        self_obj.text = text
        self_obj.sentiment_score = sentiment_score
        self_obj.keywords = keywords
        self_obj.stock_price = stock_price
        self_obj.stock_volume = stock_volume
        self_obj.price_change_24h = price_change_24h
        self_obj.volume_change_24h = volume_change_24h

    def to_features(self_obj) -> np.ndarray:
        """Converts the state into a numerical feature vector for the RL agent."""
        # This is a very basic feature vector. In a real scenario, 
        # text would be embedded, keywords one-hot encoded, etc.
        features = [
            self_obj.sentiment_score,
            len(self_obj.keywords), # Simple count for now
            self_obj.stock_price,
            self_obj.stock_volume,
            self_obj.price_change_24h,
            self_obj.volume_change_24h
        ]
        return np.array(features, dtype=np.float32)

    def __repr__(self_obj):
        return (f"RedditPostState(id='{self_obj.post_id}', sentiment={self_obj.sentiment_score:.2f}, "
                f"stock_price={self_obj.stock_price:.2f}, price_change_24h={self_obj.price_change_24h:.2f})")


class Action:
    """Defines the possible actions the RL agent can take."""
    PREDICT_LEAK = 0
    PREDICT_RUMOUR = 1

    @staticmethod
    def to_string(action_id: int) -> str:
        if action_id == Action.PREDICT_LEAK:
            return "LEAK"
        elif action_id == Action.PREDICT_RUMOUR:
            return "RUMOUR"
        else:
            return "UNKNOWN"


class RuleBasedRLModel:
    """A basic rule-based RL model for ingesting Reddit posts and stock prices
    to predict true leaks vs. rumours.

    This model initializes with a simple rule-based policy and placeholders
    for more sophisticated RL components (e.g., Q-table updates)."""

    def __init__(self_obj, fetcher: RedditDataFetcher, reward_calculator: RewardCalculator,
                 initial_q_value: float = 0.0, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self_obj.fetcher = fetcher
        self_obj.reward_calculator = reward_calculator
        self_obj.learning_rate = learning_rate
        self_obj.discount_factor = discount_factor

        # Initialize a very simple Q-table. For a rule-based model, this is more
        # of a conceptual placeholder or for simple state aggregations.
        # In a real system, states would be too complex for a direct table.
        # Here, we'll imagine a simplified state space for Q-learning example.
        self_obj.q_table: Dict[str, Dict[int, float]] = {}

        # Define some keywords and thresholds for the initial rule-based policy
        self_obj.leak_keywords = ["DD", "due diligence", "insider", "source close to", "confidential"]
        self_obj.rumour_keywords = ["troll", "shill", "moon", "yolo", "ape"]
        self_obj.sentiment_threshold_leak = 0.5  # High positive sentiment
        self_obj.price_change_threshold_leak = 0.02 # 2% price increase
        self_obj.volume_change_threshold_leak = 0.5 # 50% volume increase

    def _get_q_value(self_obj, state_key: str, action: int) -> float:
        """Retrieve Q-value for a given state-action pair."""
        return self_obj.q_table.get(state_key, {}).get(action, 0.0)

    def _set_q_value(self_obj, state_key: str, action: int, value: float):
        """Set Q-value for a given state-action pair."""
        if state_key not in self_obj.q_table:
            self_obj.q_table[state_key] = {}
        self_obj.q_table[state_key][action] = value

    def get_state_from_data(self_obj, post_data: Dict[str, Any], 
                            stock_data: Dict[str, Any]) -> RedditPostState:
        """Creates a RedditPostState object from raw post and stock data."""
        # Basic keyword extraction (can be improved with NLP)
        text_lower = post_data.get('content', '').lower()
        post_keywords = [kw for kw in self_obj.leak_keywords + self_obj.rumour_keywords if kw in text_lower]

        # Placeholder for sentiment analysis (can be integrated with NLTK, spaCy, etc.)
        # For now, we'll simulate a simple sentiment based on keywords
        sentiment_score = 0.0
        for kw in self_obj.leak_keywords: # Simulating higher sentiment for leak keywords
            if kw in text_lower: sentiment_score += 0.1
        for kw in self_obj.rumour_keywords: # Simulating lower sentiment for rumour keywords
            if kw in text_lower: sentiment_score -= 0.1
        sentiment_score = np.clip(sentiment_score, -1.0, 1.0) # Ensure within -1 to 1 range

        return RedditPostState(
            post_id=post_data.get('id', 'unknown'),
            text=post_data.get('content', ''),
            sentiment_score=sentiment_score,
            keywords=post_keywords,
            stock_price=stock_data.get('current_price', 0.0),
            stock_volume=stock_data.get('current_volume', 0.0),
            price_change_24h=stock_data.get('price_change_24h', 0.0),
            volume_change_24h=stock_data.get('volume_change_24h', 0.0)
        )

    def choose_action(self_obj, state: RedditPostState) -> int:
        """Determines the action based on predefined rules (policy)."""
        # Rule 1: Strong positive sentiment and significant price/volume increase
        if state.sentiment_score > self_obj.sentiment_threshold_leak and \
           state.price_change_24h > self_obj.price_change_threshold_leak and \
           state.volume_change_24h > self_obj.volume_change_threshold_leak:
            return Action.PREDICT_LEAK

        # Rule 2: Presence of specific leak-indicating keywords
        if any(kw in state.text.lower() for kw in self_obj.leak_keywords):
            return Action.PREDICT_LEAK

        # Default action: predict rumour
        return Action.PREDICT_RUMOUR

    def update(self_obj, current_state: RedditPostState, action: int, 
               reward: float, next_state: Optional[RedditPostState] = None):
        """Placeholder for updating the model based on reward (e.g., Q-learning)."""
        # For a truly rule-based model, this might not involve 'learning' in the RL sense.
        # However, for an *initialization* of an RL model that *starts* rule-based,
        # we include the structure for future learning.
        
        # Simplistic state key for the Q-table (e.g., discretize features)
        # In a real system, you'd use a more sophisticated state representation
        # or feature extraction to index a Q-table or feed into a neural network.
        state_key = f"{int(current_state.sentiment_score > 0)}.{int(current_state.price_change_24h > 0)}"

        current_q = self_obj._get_q_value(state_key, action)

        # If there's no next state (e.g., terminal state or end of episode),
        # the maximum Q-value for the next state is 0.
        if next_state:
            next_state_key = f"{int(next_state.sentiment_score > 0)}.{int(next_state.price_change_24h > 0)}"
            max_next_q = max([self_obj._get_q_value(next_state_key, a) for a in [Action.PREDICT_LEAK, Action.PREDICT_RUMOUR]])
        else:
            max_next_q = 0.0

        # Q-learning update rule
        new_q = current_q + self_obj.learning_rate * (reward + self_obj.discount_factor * max_next_q - current_q)
        self_obj._set_q_value(state_key, action, new_q)

        print(f"Updated Q-value for state '{state_key}', action {Action.to_string(action)}: {current_q:.2f} -> {new_q:.2f} (Reward: {reward:.2f})")


    def predict(self_obj, post_data: Dict[str, Any], stock_data: Dict[str, Any]) -> str:
        """Makes a prediction (LEAK or RUMOUR) for a given post and stock data."""
        state = self_obj.get_state_from_data(post_data, stock_data)
        action = self_obj.choose_action(state)
        return Action.to_string(action)


# Example Usage (for demonstration)
if __name__ == "__main__":
    # Mock data fetcher and reward calculator
    class MockRedditFetcher(RedditDataFetcher):
        def fetch_recent_posts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
            return [
                {"id": "post1", "title": "AMC DD - huge potential!", "content": "My extensive due diligence shows...", "timestamp": "2023-10-26T10:00:00Z", "ticker": "AMC"},
                {"id": "post2", "title": "Some random stock rumour", "content": "Heard from a guy, might be true lol", "timestamp": "2023-10-26T10:30:00Z", "ticker": "GME"},
                {"id": "post3", "title": "Insider info on TSLA", "content": "My source says Tesla will announce splits next week", "timestamp": "2023-10-26T11:00:00Z", "ticker": "TSLA"},
                {"id": "post4", "title": "YOLO on BBBY", "content": "BBBY to the moon!", "timestamp": "2023-10-26T11:30:00Z", "ticker": "BBBY"}
            ]

        def fetch_stock_prices(self, ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
            # Simplified mock for a single point in time
            if ticker == "AMC":
                return {"current_price": 40.0, "current_volume": 100_000_000, "price_change_24h": 0.03, "volume_change_24h": 0.6}
            elif ticker == "GME":
                return {"current_price": 20.0, "current_volume": 50_000_000, "price_change_24h": 0.005, "volume_change_24h": 0.1}
            elif ticker == "TSLA":
                return {"current_price": 250.0, "current_volume": 80_000_000, "price_change_24h": 0.025, "volume_change_24h": 0.7}
            elif ticker == "BBBY":
                return {"current_price": 1.5, "current_volume": 200_000_000, "price_change_24h": -0.05, "volume_change_24h": 0.3}
            return {"current_price": 0.0, "current_volume": 0.0, "price_change_24h": 0.0, "volume_change_24h": 0.0}

    class MockRewardCalculator(RewardCalculator):
        def calculate_reward(self, predicted_action: int, true_label: Optional[int] = None) -> float:
            if true_label is None:
                # In a real scenario, we'd need to observe future stock movement
                # or external validation to determine the 'true_label'.
                # For this demo, we'll assign a neutral reward.
                return 0.0
            if predicted_action == true_label:
                return 1.0 # Correct prediction
            else:
                return -1.0 # Incorrect prediction

    print("Initializing RuleBasedRLModel...")
    fetcher = MockRedditFetcher()
    reward_calc = MockRewardCalculator()
    rl_model = RuleBasedRLModel(fetcher, reward_calc)

    # Simulate processing a few posts
    posts = fetcher.fetch_recent_posts(query="stocks")

    print("\n--- Making Predictions ---")
    for post in posts:
        ticker = post.get('ticker', 'UNKNOWN')
        stock_data = fetcher.fetch_stock_prices(ticker, "", "") # Simplified for mock
        
        prediction = rl_model.predict(post, stock_data)
        print(f"Post ID: {post['id']}, Ticker: {ticker}, Prediction: {prediction}")

        # Simulate an update step after prediction (requires a true label for meaningful reward)
        # For demo, let's assume post1 and post3 were true leaks, post2 and post4 were rumours
        true_label = None
        if post['id'] == 'post1': true_label = Action.PREDICT_LEAK
        if post['id'] == 'post2': true_label = Action.PREDICT_RUMOUR
        if post['id'] == 'post3': true_label = Action.PREDICT_LEAK
        if post['id'] == 'post4': true_label = Action.PREDICT_RUMOUR

        if true_label is not None:
            # Convert prediction string back to action ID for reward calculation
            predicted_action_id = Action.PREDICT_LEAK if prediction == "LEAK" else Action.PREDICT_RUMOUR
            reward = reward_calc.calculate_reward(predicted_action_id, true_label)
            current_state = rl_model.get_state_from_data(post, stock_data)
            rl_model.update(current_state, predicted_action_id, reward)

    print("\n--- Q-table after updates (simplified view) ---")
    for state_key, actions_q_values in rl_model.q_table.items():
        print(f"State Key: {state_key}")
        for action_id, q_value in actions_q_values.items():
            print(f"  Action {Action.to_string(action_id)}: {q_value:.4f}")
