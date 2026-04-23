-- ============================================================
-- Stock Rumors DB Schema
-- Run this manually or let collector.py init_db() handle it
-- ============================================================

CREATE TABLE IF NOT EXISTS reddit_posts (
    id              TEXT PRIMARY KEY,
    subreddit       TEXT        NOT NULL,
    title           TEXT        NOT NULL,
    body            TEXT,
    author          TEXT,
    score           INTEGER,
    upvote_ratio    FLOAT,
    num_comments    INTEGER,
    url             TEXT,
    permalink       TEXT,
    flair           TEXT,
    is_self         BOOLEAN,
    created_utc     TIMESTAMPTZ NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tickers_found   TEXT[]      DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS stock_prices (
    id              SERIAL PRIMARY KEY,
    ticker          TEXT        NOT NULL,
    price_date      DATE        NOT NULL,
    open            FLOAT,
    high            FLOAT,
    low             FLOAT,
    close           FLOAT,
    volume          BIGINT,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (ticker, price_date)
);

-- Links posts to tickers they mention (many-to-many)
CREATE TABLE IF NOT EXISTS post_ticker_links (
    post_id         TEXT        REFERENCES reddit_posts(id) ON DELETE CASCADE,
    ticker          TEXT        NOT NULL,
    PRIMARY KEY (post_id, ticker)
);

-- Useful for your RL model later: manual or model-assigned labels
CREATE TABLE IF NOT EXISTS post_labels (
    post_id         TEXT        REFERENCES reddit_posts(id) ON DELETE CASCADE PRIMARY KEY,
    label           TEXT        CHECK (label IN ('rumor', 'leak', 'confirmed', 'false', 'unknown')),
    labeled_by      TEXT,       -- 'human' | 'model_v1' etc.
    confidence      FLOAT,
    labeled_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_posts_created   ON reddit_posts (created_utc);
CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON reddit_posts (subreddit);
CREATE INDEX IF NOT EXISTS idx_posts_tickers   ON reddit_posts USING GIN (tickers_found);
CREATE INDEX IF NOT EXISTS idx_prices_ticker   ON stock_prices (ticker, price_date);