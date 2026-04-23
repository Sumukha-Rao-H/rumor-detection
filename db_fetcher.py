from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy import desc

from reddit_data_fetcher import RedditDataFetcher
from database import get_session
from models import RedditPost, StockPrice

class DatabaseFetcher(RedditDataFetcher):
    def fetch_recent_posts(self, query: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        with get_session() as session:
            q = session.query(RedditPost)
            if query:
                q = q.filter((RedditPost.title.ilike(f"%{query}%")) | (RedditPost.body.ilike(f"%{query}%")))
            posts = q.order_by(RedditPost.created_utc.asc()).limit(limit).all()
            
            results = []
            for p in posts:
                ticker = p.tickers_found[0] if p.tickers_found else "UNKNOWN"
                results.append({
                    "id": p.id,
                    "title": p.title,
                    "content": p.body or "",
                    "timestamp": p.created_utc.isoformat() if p.created_utc else "",
                    "ticker": ticker
                })
            return results

    def fetch_stock_prices(self, ticker: str, start_date: str, end_date: str = "") -> Dict[str, Any]:
        """
        Uses `start_date` to find the stock price prior to or on the date of the post.
        Computes 24h change by comparing it with the preceding trading day.
        """
        if not start_date or ticker == "UNKNOWN":
            return {"current_price": 0.0, "current_volume": 0.0, "price_change_24h": 0.0, "volume_change_24h": 0.0}

        target_date = datetime.fromisoformat(start_date).date()
        
        with get_session() as session:
            # Get the exact or prior date for the post
            current_price_row = session.query(StockPrice).filter(
                StockPrice.ticker == ticker,
                StockPrice.price_date <= target_date
            ).order_by(desc(StockPrice.price_date)).first()

            if not current_price_row:
                return {"current_price": 0.0, "current_volume": 0.0, "price_change_24h": 0.0, "volume_change_24h": 0.0}
            
            # Get the previous day's price to calculate change
            prev_price_row = session.query(StockPrice).filter(
                StockPrice.ticker == ticker,
                StockPrice.price_date < current_price_row.price_date
            ).order_by(desc(StockPrice.price_date)).first()

            current_price = current_price_row.close or 0.0
            current_volume = current_price_row.volume or 0.0
            
            price_change = 0.0
            volume_change = 0.0

            if prev_price_row and prev_price_row.close:
                prev_price = prev_price_row.close or 0.0
                prev_volume = prev_price_row.volume or 0.0
                if prev_price > 0:
                    price_change = (current_price - prev_price) / prev_price
                if prev_volume > 0:
                    volume_change = (current_volume - prev_volume) / prev_volume

            return {
                "current_price": current_price,
                "current_volume": current_volume,
                "price_change_24h": price_change,
                "volume_change_24h": volume_change
            }
