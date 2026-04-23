import argparse
from db_fetcher import DatabaseFetcher
from rl_model import RuleBasedRLModel, Action
from reward import RewardCalculator
from database import get_session
from models import PostLabel, StockPrice
from sqlalchemy import desc, asc
from datetime import datetime, timedelta

class ProxyRewardCalculator(RewardCalculator):
    def calculate_reward(self, predicted_action: int, true_label: int = None) -> float:
        if true_label is None:
            return 0.0
        if predicted_action == true_label:
            return 1.0
        else:
            return -1.0

def get_proxy_true_label(ticker: str, post_timestamp_str: str) -> int:
    """
    Given a ticker and a post timestamp, compute a proxy true label.
    If the stock goes up > 2% within the next 3 days, it was a LEAK.
    Else, RUMOUR.
    """
    if ticker == "UNKNOWN" or not post_timestamp_str:
        return Action.PREDICT_RUMOUR
        
    try:
        post_date = datetime.fromisoformat(post_timestamp_str).date()
    except Exception:
        return Action.PREDICT_RUMOUR

    target_date = post_date + timedelta(days=3)

    with get_session() as session:
        # Start price on or immediately after post date
        start_price_row = session.query(StockPrice).filter(
            StockPrice.ticker == ticker,
            StockPrice.price_date >= post_date
        ).order_by(asc(StockPrice.price_date)).first()

        # End price typically 3 trading days later
        end_price_row = session.query(StockPrice).filter(
            StockPrice.ticker == ticker,
            StockPrice.price_date <= target_date,
            StockPrice.price_date >= post_date
        ).order_by(desc(StockPrice.price_date)).first()

        if start_price_row and end_price_row and start_price_row.close and end_price_row.close and start_price_row.close > 0:
            change = (end_price_row.close - start_price_row.close) / start_price_row.close
            if change > 0.02:
                return Action.PREDICT_LEAK
            
    return Action.PREDICT_RUMOUR

def main():
    print("Initializing Database-backed Offline RL Model...")
    fetcher = DatabaseFetcher()
    reward_calc = ProxyRewardCalculator()
    # High learning rate and strict discount factor for basic RL rule-based Q updates
    rl_model = RuleBasedRLModel(fetcher, reward_calc, learning_rate=0.2)

    print("Fetching historical posts from database for offline training...")
    posts = fetcher.fetch_recent_posts(limit=1000)
    print(f"Loaded {len(posts)} posts for training.")

    with get_session() as session:
        for post in posts:
            ticker = post.get("ticker", "UNKNOWN")
            timestamp = post.get("timestamp")
            
            # Fetch state context synchronously from DB relative to post date
            stock_data = fetcher.fetch_stock_prices(ticker, timestamp, "")
            
            prediction_str = rl_model.predict(post, stock_data)
            predicted_action_id = Action.PREDICT_LEAK if prediction_str == "LEAK" else Action.PREDICT_RUMOUR
            
            # Compute proxy true label based on subsequent 3-day stock performance
            true_label = get_proxy_true_label(ticker, timestamp)
            reward = reward_calc.calculate_reward(predicted_action_id, true_label)
            
            # Update the RL model Q-values
            current_state = rl_model.get_state_from_data(post, stock_data)
            rl_model.update(current_state, predicted_action_id, reward)

            # Store the prediction output directly back as a model_v1 label
            existing_label = session.query(PostLabel).filter_by(post_id=post["id"]).first()
            if not existing_label:
                new_label = PostLabel(
                    post_id=post["id"],
                    label="leak" if prediction_str == "LEAK" else "rumor",
                    labeled_by="model_v1",
                    confidence=rl_model._get_q_value(
                        f"{int(current_state.sentiment_score > 0)}.{int(current_state.price_change_24h > 0)}",
                        predicted_action_id
                    )
                )
                session.add(new_label)
            else:
                existing_label.label = "leak" if prediction_str == "LEAK" else "rumor"
                existing_label.labeled_by = "model_v1"
                existing_label.confidence = rl_model._get_q_value(
                        f"{int(current_state.sentiment_score > 0)}.{int(current_state.price_change_24h > 0)}",
                        predicted_action_id
                )
            
            # Print minimal output indicating tracking flow
            print(f"Trained on '{post['id']}' ({ticker}) | Pred: {prediction_str} | True: {Action.to_string(true_label)} | Reward: {reward}")

    print("\n--- Q-table after historical DB training ---")
    for state_key, actions_q_values in rl_model.q_table.items():
        print(f"State Key: {state_key}")
        for action_id, q_value in actions_q_values.items():
            print(f"  Action {Action.to_string(action_id)}: {q_value:.4f}")

if __name__ == "__main__":
    main()
