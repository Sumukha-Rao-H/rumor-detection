"""Storage-layer tests: schema, idempotent upserts, snapshots."""

import pytest

from src import db


@pytest.fixture()
def conn(tmp_path):
    c = db.get_conn(tmp_path / "test.db")
    yield c
    c.close()


def make_post(pid="abc123", **over):
    p = {
        "id": pid, "subreddit": "wallstreetbets", "title": "TSLA merger rumor",
        "selftext": "body", "author": "u1", "created_utc": 1_750_000_000,
        "score": 10, "upvote_ratio": 0.9, "num_comments": 5, "flair": None,
        "url": "https://reddit.com/x", "source": "live",
        "fetched_utc": 1_750_000_100, "score_6h": None, "score_24h": None,
    }
    p.update(over)
    return p


def test_upsert_posts_idempotent(conn):
    assert db.upsert_posts(conn, [make_post()]) == 1
    # Re-insert with a fresher score: no new row, mutable fields refreshed
    assert db.upsert_posts(conn, [make_post(score=99)]) == 0
    row = conn.execute("SELECT score FROM posts WHERE id='abc123'").fetchone()
    assert row["score"] == 99


def test_score_snapshots(conn):
    db.upsert_posts(conn, [make_post()])
    db.set_post_score_snapshot(conn, "abc123", "score_6h", 42)
    row = conn.execute("SELECT score_6h FROM posts WHERE id='abc123'").fetchone()
    assert row["score_6h"] == 42
    with pytest.raises(ValueError):
        db.set_post_score_snapshot(conn, "abc123", "score", 1)


def test_bars_upsert_and_latest(conn):
    rows = [("TSLA", 1_750_000_000, 1, 2, 0.5, 1.5, 1000, "60m"),
            ("TSLA", 1_750_003_600, 1.5, 2, 1, 1.8, 900, "60m")]
    db.upsert_bars(conn, rows)
    db.upsert_bars(conn, rows)  # idempotent
    n = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    assert n == 2
    assert db.latest_bar_ts(conn, "TSLA", "60m") == 1_750_003_600
    assert db.latest_bar_ts(conn, "AAPL", "60m") is None


def test_news_dedupe_by_url(conn):
    row = ("https://reuters.com/a", "TSLA", "Tesla acquires X", "reuters.com",
           1_750_000_000, "gdelt")
    assert db.upsert_news(conn, [row]) == 1
    assert db.upsert_news(conn, [row]) == 0


def test_ticker_links_and_dedupe(conn):
    db.upsert_posts(conn, [make_post()])
    db.link_post_tickers(conn, [("abc123", "TSLA"), ("abc123", "TSLA")])
    assert db.distinct_post_tickers(conn) == ["TSLA"]
    assert db.known_post_ids(conn, ["abc123", "zzz"]) == {"abc123"}
