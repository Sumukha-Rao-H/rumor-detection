"""Live poller tests (Arctic Shift-backed) — network mocked via _search injection."""

import pytest

from src import db
from src.collectors.reddit_live import LivePoller
from src.utils.config import load_config
from src.utils.timeutils import utc_now_ts


def make_record(pid, title, subreddit="stocks", created=1_750_000_000, score=10):
    return {"id": pid, "subreddit": subreddit, "title": title, "selftext": "",
            "author": "u1", "created_utc": created, "score": score,
            "upvote_ratio": 0.9, "num_comments": 2, "link_flair_text": None,
            "url": "https://reddit.com/x"}


@pytest.fixture()
def poller(tmp_path):
    cfg = load_config()
    cfg["paths"]["live_json"] = str(tmp_path / "snapshots")
    cfg["arctic_shift"]["min_interval_s"] = 0  # no pacing in tests
    cfg["pullpush"]["min_interval_s"] = 0
    conn = db.get_conn(tmp_path / "t.db")
    p = LivePoller(cfg, conn)
    yield p
    conn.close()


def test_poll_once_filters_dedupes_and_snapshots(poller):
    records = [
        make_record("p1", "$TSLA buyout rumor"),
        make_record("p2", "no tickers here"),
    ]
    poller._search = lambda base_url, limiter, subreddit, after, before=None: (
        records if subreddit == "stocks" else []
    )
    assert poller.poll_once() == 1          # only the ticker post inserted
    assert poller.poll_once() == 0          # second cycle: deduped by id
    ids = {r[0] for r in poller.conn.execute("SELECT id FROM posts")}
    assert ids == {"p1"}
    # one snapshot per subreddit per cycle that returned records
    assert len(list(poller.snapshot_dir.glob("*_stocks.json"))) == 2


def test_high_water_mark_advances_between_cycles(poller):
    calls = []
    newest_created = utc_now_ts() - 100  # newer than the initial lookback mark

    def fake_search(base_url, limiter, subreddit, after, before=None):
        calls.append((subreddit, after))
        if subreddit == "stocks" and sum(1 for s, _ in calls if s == "stocks") == 1:
            return [make_record("p1", "$TSLA merger", created=newest_created)]
        return []

    poller._search = fake_search
    poller.poll_once()
    poller.poll_once()
    stocks_afters = [a for sub, a in calls if sub == "stocks"]
    assert stocks_afters[1] == newest_created  # advanced to the last post's created_utc


def test_arctic_shift_failure_falls_back_to_pullpush(poller):
    poller.fallback_url = "https://api.pullpush.io/reddit/search/submission"
    calls = []

    def fake_search(base_url, limiter, subreddit, after, before=None):
        calls.append(base_url)
        if base_url == poller.primary_url:
            return None  # Arctic Shift exhausted its retries
        return [make_record("p1", "$TSLA merger")]

    poller._search = fake_search
    fresh = poller._fetch("stocks", 0)
    assert len(fresh) == 1
    assert calls == [poller.primary_url, poller.fallback_url]


def test_refetch_due_fills_score_snapshots(poller):
    old_created = utc_now_ts()  # "just created" — not due yet when poll_once runs its own refetch
    poller._search = lambda base_url, limiter, subreddit, after, before=None: (
        [make_record("p1", "$TSLA merger", created=old_created)] if subreddit == "stocks" else []
    )
    poller.poll_once()

    poller._search = lambda base_url, limiter, subreddit, after, before=None: (
        [make_record("p1", "$TSLA merger", created=old_created, score=77)]
    )
    poller.refetch_due(poller.cfg["reddit"]["refetch_hours"][1] * 3600 + old_created + 10)
    row = poller.conn.execute(
        "SELECT score_6h, score_24h FROM posts WHERE id='p1'").fetchone()
    assert row["score_6h"] == 77 and row["score_24h"] == 77


def test_refetch_marks_deleted_posts_zero(poller):
    old_created = utc_now_ts()
    poller._search = lambda base_url, limiter, subreddit, after, before=None: (
        [make_record("p1", "$TSLA merger", created=old_created)] if subreddit == "stocks" else []
    )
    poller.poll_once()

    poller._search = lambda base_url, limiter, subreddit, after, before=None: []  # post deleted
    poller.refetch_due(poller.cfg["reddit"]["refetch_hours"][1] * 3600 + old_created + 10)
    row = poller.conn.execute(
        "SELECT score_6h, score_24h FROM posts WHERE id='p1'").fetchone()
    assert row["score_6h"] == 0 and row["score_24h"] == 0
