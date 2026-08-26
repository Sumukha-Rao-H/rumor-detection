"""Event clustering tests (plan §6.3)."""

from src import db
from src.pipeline.events import build_events, cluster_ticker_events, is_rumor_candidate
from src.utils.config import load_config


def make_post(pid, title, created, score=10):
    return {"id": pid, "subreddit": "stocks", "title": title, "selftext": "",
            "author": "u1", "created_utc": created, "score": score,
            "upvote_ratio": 0.9, "num_comments": 1, "flair": None,
            "url": "https://reddit.com/x", "source": "arctic", "fetched_utc": created,
            "score_6h": None, "score_24h": None}


def test_is_rumor_candidate_matches_keyword():
    post = make_post("p1", "$TSLA merger talks heating up", 1_000_000)
    assert is_rumor_candidate(post, ["merger", "acquisition"]) == "merger"


def test_is_rumor_candidate_no_match():
    post = make_post("p2", "$TSLA to the moon!!!", 1_000_000)
    assert is_rumor_candidate(post, ["merger", "acquisition"]) is None


def test_cluster_splits_on_gap_and_drops_small_far_cluster():
    posts = [
        make_post("p1", "$TSLA merger rumor", 0, score=10),
        make_post("p2", "$TSLA merger update", 3600, score=10),          # 1h later — same cluster
        make_post("p3", "$TSLA merger confirmed?", 50 * 3600, score=10),  # 50h later — new, too small
    ]
    for p in posts:
        p["claim_type"] = "merger"
    clusters = cluster_ticker_events(posts, gap_hours=12, min_posts=2, min_solo_score=50)
    assert len(clusters) == 1
    assert [p["id"] for p in clusters[0]] == ["p1", "p2"]


def test_cluster_keeps_high_score_solo_post():
    posts = [make_post("p1", "$TSLA merger rumor", 0, score=200)]
    posts[0]["claim_type"] = "merger"
    clusters = cluster_ticker_events(posts, gap_hours=12, min_posts=2, min_solo_score=50)
    assert len(clusters) == 1 and len(clusters[0]) == 1


def test_build_events_writes_events_table(tmp_path):
    cfg = load_config()
    conn = db.get_conn(tmp_path / "t.db")
    posts = [
        make_post("p1", "$TSLA merger rumor huge", 0, score=100),
        make_post("p2", "$TSLA no news here just yolo", 100, score=5),
    ]
    db.upsert_posts(conn, posts)
    db.link_post_tickers(conn, [("p1", "TSLA"), ("p2", "TSLA")])

    n_new = build_events(conn, cfg)
    assert n_new == 1
    row = conn.execute("SELECT * FROM events").fetchone()
    assert row["ticker"] == "TSLA"
    assert row["t0_utc"] == 0
    assert row["claim_type"] == "merger"
    assert row["post_ids"] == "p1"
    assert row["label"] is None
    assert row["human_reviewed"] == 0

    assert build_events(conn, cfg) == 0  # idempotent re-run
