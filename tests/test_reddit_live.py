"""Live poller tests — network mocked via _get injection."""

import pytest

from src import db
from src.collectors.reddit_live import LivePoller, listing_to_posts
from src.utils.config import load_config


def make_listing(children):
    return {"data": {"children": [{"data": c} for c in children]}}


def make_child(pid, title, created=1_750_000_000, score=10):
    return {"id": pid, "subreddit": "stocks", "title": title, "selftext": "",
            "author": "u1", "created_utc": created, "score": score,
            "upvote_ratio": 0.9, "num_comments": 2, "link_flair_text": None,
            "url": "https://reddit.com/x"}


@pytest.fixture()
def poller(tmp_path):
    cfg = load_config()
    cfg["paths"]["live_json"] = str(tmp_path / "snapshots")
    cfg["reddit"]["live_min_interval_s"] = 0  # no pacing in tests
    conn = db.get_conn(tmp_path / "t.db")
    p = LivePoller(cfg, conn)
    yield p
    conn.close()


def test_listing_to_posts_normalization():
    posts = listing_to_posts(make_listing([make_child("p1", "hi")]), 123)
    assert posts[0]["source"] == "live"
    assert posts[0]["fetched_utc"] == 123


def test_poll_once_filters_dedupes_and_snapshots(poller):
    listing = make_listing([
        make_child("p1", "$TSLA buyout rumor"),
        make_child("p2", "no tickers here"),
    ])
    poller._get = lambda url, params: listing if "new.json" in url else None
    assert poller.poll_once() == 1          # only the ticker post inserted
    assert poller.poll_once() == 0          # second cycle: deduped by id
    ids = {r[0] for r in poller.conn.execute("SELECT id FROM posts")}
    assert ids == {"p1"}
    # raw snapshot archived once per cycle, before filtering
    assert len(list(poller.snapshot_dir.glob("*_new.json"))) == 2


def test_refetch_due_fills_score_snapshots(poller):
    old_created = 1_000_000  # far in the past — both windows due
    listing = make_listing([make_child("p1", "$TSLA merger", created=old_created)])
    calls = []

    def fake_get(url, params):
        if "new.json" in url:
            return listing
        calls.append(params["id"])
        return make_listing([make_child("p1", "$TSLA merger",
                                        created=old_created, score=77)])

    poller._get = fake_get
    poller.poll_once()
    row = poller.conn.execute(
        "SELECT score_6h, score_24h FROM posts WHERE id='p1'").fetchone()
    assert row["score_6h"] == 77 and row["score_24h"] == 77
    assert calls == ["t3_p1", "t3_p1"]
    # next cycle: nothing due anymore
    calls.clear()
    poller.poll_once()
    assert calls == []


def test_persistent_403_rotates_base_host(poller):
    from src.utils.ratelimit import Backoff

    class Resp:
        status_code = 403

    poller.backoff = Backoff(base_s=0)  # no real sleeping in tests
    poller.session.get = lambda url, params=None, timeout=None: Resp()
    assert poller.base_url == "https://www.reddit.com"
    assert poller._get(f"{poller.base_url}/r/stocks/new.json", {}) is None
    assert poller.base_url == "https://old.reddit.com"


def test_refetch_marks_deleted_posts_zero(poller):
    listing = make_listing([make_child("p1", "$TSLA merger", created=1_000_000)])

    def fake_get(url, params):
        if "new.json" in url:
            return listing
        return make_listing([])  # post deleted — info returns nothing

    poller._get = fake_get
    poller.poll_once()
    row = poller.conn.execute(
        "SELECT score_6h, score_24h FROM posts WHERE id='p1'").fetchone()
    assert row["score_6h"] == 0 and row["score_24h"] == 0
