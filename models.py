"""
SQLAlchemy ORM Models
======================
Mirrors the four tables defined in Schema.sql so that queries, label
writes, and feature-engineering code can use the ORM instead of raw SQL.

Tables:
  - reddit_posts      -> RedditPost
  - stock_prices      -> StockPrice
  - post_ticker_links -> PostTickerLink
  - post_labels       -> PostLabel

NOTE: The collector (reddit_data_fetcher.py) still uses psycopg2 +
execute_values for bulk inserts -- that is intentional for performance.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Text,
    Integer,
    Float,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    CheckConstraint,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship


# ---------- Base class ----------
class Base(DeclarativeBase):
    """Shared declarative base for all models."""
    pass


# =================================================================
# reddit_posts
# =================================================================
class RedditPost(Base):
    __tablename__ = "reddit_posts"

    id            = Column(Text, primary_key=True)
    subreddit     = Column(Text, nullable=False)
    title         = Column(Text, nullable=False)
    body          = Column(Text)
    author        = Column(Text)
    score         = Column(Integer)
    upvote_ratio  = Column(Float)
    num_comments  = Column(Integer)
    url           = Column(Text)
    permalink     = Column(Text)
    flair         = Column(Text)
    is_self       = Column(Boolean)
    created_utc   = Column(DateTime(timezone=True), nullable=False)
    fetched_at    = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    tickers_found = Column(ARRAY(Text), default=[])

    # -- Relationships --
    ticker_links = relationship(
        "PostTickerLink", back_populates="post", cascade="all, delete-orphan"
    )
    label = relationship(
        "PostLabel", back_populates="post", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_posts_created", "created_utc"),
        Index("idx_posts_subreddit", "subreddit"),
        Index("idx_posts_tickers", "tickers_found", postgresql_using="gin"),
    )

    def __repr__(self):
        return f"<RedditPost id={self.id!r} sub={self.subreddit!r}>"


# =================================================================
# stock_prices
# =================================================================
class StockPrice(Base):
    __tablename__ = "stock_prices"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    ticker     = Column(Text, nullable=False)
    price_date = Column(Date, nullable=False)
    open       = Column(Float)
    high       = Column(Float)
    low        = Column(Float)
    close      = Column(Float)
    volume     = Column(BigInteger)
    fetched_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint("ticker", "price_date", name="uq_ticker_price_date"),
        Index("idx_prices_ticker", "ticker", "price_date"),
    )

    def __repr__(self):
        return f"<StockPrice {self.ticker} {self.price_date}>"


# =================================================================
# post_ticker_links  (many-to-many join table)
# =================================================================
class PostTickerLink(Base):
    __tablename__ = "post_ticker_links"

    post_id = Column(
        Text, ForeignKey("reddit_posts.id", ondelete="CASCADE"), primary_key=True
    )
    ticker = Column(Text, nullable=False, primary_key=True)

    # -- Relationships --
    post = relationship("RedditPost", back_populates="ticker_links")

    def __repr__(self):
        return f"<PostTickerLink post={self.post_id!r} ticker={self.ticker!r}>"


# =================================================================
# post_labels  (RL training labels)
# =================================================================
class PostLabel(Base):
    __tablename__ = "post_labels"

    post_id = Column(
        Text,
        ForeignKey("reddit_posts.id", ondelete="CASCADE"),
        primary_key=True,
    )
    label = Column(Text)
    labeled_by = Column(Text)       # 'human', 'model_v1', etc.
    confidence = Column(Float)
    labeled_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # -- Relationships --
    post = relationship("RedditPost", back_populates="label")

    __table_args__ = (
        CheckConstraint(
            "label IN ('rumor', 'leak', 'confirmed', 'false', 'unknown')",
            name="ck_post_labels_label",
        ),
    )

    def __repr__(self):
        return f"<PostLabel post={self.post_id!r} label={self.label!r}>"
