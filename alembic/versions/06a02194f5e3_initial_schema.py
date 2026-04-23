"""initial_schema

Revision ID: 06a02194f5e3
Revises: 
Create Date: 2026-04-23 10:36:16.670032

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY


# revision identifiers, used by Alembic.
revision: str = '06a02194f5e3'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- reddit_posts --
    op.create_table(
        "reddit_posts",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("subreddit", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("body", sa.Text()),
        sa.Column("author", sa.Text()),
        sa.Column("score", sa.Integer()),
        sa.Column("upvote_ratio", sa.Float()),
        sa.Column("num_comments", sa.Integer()),
        sa.Column("url", sa.Text()),
        sa.Column("permalink", sa.Text()),
        sa.Column("flair", sa.Text()),
        sa.Column("is_self", sa.Boolean()),
        sa.Column("created_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("tickers_found", ARRAY(sa.Text()), server_default="{}"),
    )
    op.create_index("idx_posts_created", "reddit_posts", ["created_utc"])
    op.create_index("idx_posts_subreddit", "reddit_posts", ["subreddit"])
    op.create_index(
        "idx_posts_tickers",
        "reddit_posts",
        ["tickers_found"],
        postgresql_using="gin",
    )

    # -- stock_prices --
    op.create_table(
        "stock_prices",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("price_date", sa.Date(), nullable=False),
        sa.Column("open", sa.Float()),
        sa.Column("high", sa.Float()),
        sa.Column("low", sa.Float()),
        sa.Column("close", sa.Float()),
        sa.Column("volume", sa.BigInteger()),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint("ticker", "price_date", name="uq_ticker_price_date"),
    )
    op.create_index("idx_prices_ticker", "stock_prices", ["ticker", "price_date"])

    # -- post_ticker_links --
    op.create_table(
        "post_ticker_links",
        sa.Column(
            "post_id",
            sa.Text(),
            sa.ForeignKey("reddit_posts.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("ticker", sa.Text(), nullable=False, primary_key=True),
    )

    # -- post_labels --
    op.create_table(
        "post_labels",
        sa.Column(
            "post_id",
            sa.Text(),
            sa.ForeignKey("reddit_posts.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("label", sa.Text()),
        sa.Column("labeled_by", sa.Text()),
        sa.Column("confidence", sa.Float()),
        sa.Column(
            "labeled_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
        sa.CheckConstraint(
            "label IN ('rumor', 'leak', 'confirmed', 'false', 'unknown')",
            name="ck_post_labels_label",
        ),
    )


def downgrade() -> None:
    op.drop_table("post_labels")
    op.drop_table("post_ticker_links")
    op.drop_index("idx_prices_ticker", table_name="stock_prices")
    op.drop_table("stock_prices")
    op.drop_index("idx_posts_tickers", table_name="reddit_posts")
    op.drop_index("idx_posts_subreddit", table_name="reddit_posts")
    op.drop_index("idx_posts_created", table_name="reddit_posts")
    op.drop_table("reddit_posts")
