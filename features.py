"""
Feature Engineering Queries
============================
Sample SQLAlchemy queries that join reddit_posts + stock_prices via
post_ticker_links, producing training-ready data for the RL model.

Usage:
    from features import build_training_features
    rows = build_training_features(limit=500)
"""

import logging
from datetime import timedelta

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from database import get_session
from models import RedditPost, StockPrice, PostTickerLink, PostLabel

log = logging.getLogger(__name__)


def build_training_features(limit: int = 1000) -> list[dict]:
    """
    Build a flat list of feature dicts by joining:
        reddit_posts  ->  post_ticker_links  ->  stock_prices

    For each (post, ticker) pair the query also grabs the *closest*
    stock price row on or after the post date so you can compute the
    price reaction to the post.

    Columns returned per row:
      - post_id, subreddit, title, body, score, upvote_ratio,
        num_comments, flair, created_utc
      - ticker
      - price_date, open, high, low, close, volume
      - label, confidence  (NULL when no label exists yet)

    Ordered by created_utc ascending (oldest first) so the RL agent
    sees posts in chronological order.
    """
    with get_session() as session:
        # Build the core join:
        #   reddit_posts JOIN post_ticker_links JOIN stock_prices
        # Also LEFT JOIN post_labels to include any existing labels.
        stmt = (
            select(
                RedditPost.id.label("post_id"),
                RedditPost.subreddit,
                RedditPost.title,
                RedditPost.body,
                RedditPost.score,
                RedditPost.upvote_ratio,
                RedditPost.num_comments,
                RedditPost.flair,
                RedditPost.created_utc,
                PostTickerLink.ticker,
                StockPrice.price_date,
                StockPrice.open,
                StockPrice.high,
                StockPrice.low,
                StockPrice.close,
                StockPrice.volume,
                PostLabel.label,
                PostLabel.confidence,
            )
            .join(PostTickerLink, RedditPost.id == PostTickerLink.post_id)
            .join(
                StockPrice,
                (PostTickerLink.ticker == StockPrice.ticker)
                & (StockPrice.price_date >= func.date(RedditPost.created_utc))
                & (
                    StockPrice.price_date
                    <= func.date(RedditPost.created_utc) + 7
                ),
            )
            .outerjoin(PostLabel, RedditPost.id == PostLabel.post_id)
            .order_by(RedditPost.created_utc.asc())
            .limit(limit)
        )

        results = session.execute(stmt).all()

        features = [
            {
                "post_id": r.post_id,
                "subreddit": r.subreddit,
                "title": r.title,
                "body": r.body,
                "score": r.score,
                "upvote_ratio": r.upvote_ratio,
                "num_comments": r.num_comments,
                "flair": r.flair,
                "created_utc": r.created_utc,
                "ticker": r.ticker,
                "price_date": r.price_date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "label": r.label,
                "confidence": r.confidence,
            }
            for r in results
        ]

    log.info(f"Built {len(features)} training feature rows.")
    return features


# ─────────────────────────────────────────────
# Quick test: run this file directly to preview features
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rows = build_training_features(limit=10)
    for row in rows:
        print(
            f"{row['created_utc']}  {row['ticker']:5s}  "
            f"${row['close']:.2f}  [{row['label'] or 'unlabeled'}]  "
            f"{row['title'][:60]}"
        )
