"""Per-event news retrieval tests (plan §6.4 stage 1). No network is touched."""

import json

from src import db
from src.pipeline.event_news import Window, event_windows, finnhub_covers
from src.utils.timeutils import utc_now_ts

HOUR = 3600
DAY = 86400


def _cfg():
    return {"news": {"event_pre_hours": 24, "finnhub_lookback_days": 340},
            "event": {"label_horizon_hours": 72}}


def _event(conn, event_id, ticker, t0, n_rumor_posts=1, label=None):
    db.upsert_events(conn, [{
        "event_id": event_id, "ticker": ticker, "t0_utc": t0,
        "claim_summary": "c", "claim_type": "merger", "post_ids": json.dumps(["p"]),
        "n_posts": 1, "subreddits": json.dumps(["stocks"]),
    }])
    conn.execute("UPDATE events SET n_rumor_posts = ?, label = ? WHERE event_id = ?",
                 (n_rumor_posts, label, event_id))
    conn.commit()


def test_window_spans_the_plans_labeling_horizon(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA", 1_000_000)
    (w,) = event_windows(conn, _cfg())
    assert w.start_utc == 1_000_000 - 24 * HOUR
    assert w.end_utc == 1_000_000 + 72 * HOUR


def test_overlapping_windows_on_one_ticker_merge(tmp_path):
    """Two rumors a day apart are one GDELT call, not two."""
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA", 1_000_000)
    _event(conn, "A-2", "AAA", 1_000_000 + DAY)
    (w,) = event_windows(conn, _cfg())
    assert w.n_events == 2
    assert w.start_utc == 1_000_000 - 24 * HOUR
    assert w.end_utc == 1_000_000 + DAY + 72 * HOUR


def test_distant_windows_stay_separate(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA", 1_000_000)
    _event(conn, "A-2", "AAA", 1_000_000 + 30 * DAY)
    windows = event_windows(conn, _cfg())
    assert len(windows) == 2 and all(w.n_events == 1 for w in windows)


def test_windows_never_merge_across_tickers(tmp_path):
    """A BBB headline cannot confirm an AAA claim."""
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA", 1_000_000)
    _event(conn, "B-1", "BBB", 1_000_000)
    assert sorted(w.ticker for w in event_windows(conn, _cfg())) == ["AAA", "BBB"]


def test_only_rumor_events_are_fetched(tmp_path):
    """Events triage rejected have no claim to check."""
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA", 1_000_000, n_rumor_posts=0)
    _event(conn, "B-1", "BBB", 1_000_000, n_rumor_posts=2)
    assert [w.ticker for w in event_windows(conn, _cfg())] == ["BBB"]


def test_already_labeled_events_are_skipped_by_default(tmp_path):
    """A resumed run must not re-fetch news for events already decided."""
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA", 1_000_000, label=1)
    _event(conn, "B-1", "BBB", 1_000_000)
    assert [w.ticker for w in event_windows(conn, _cfg())] == ["BBB"]
    assert len(event_windows(conn, _cfg(), unlabeled_only=False)) == 2


def test_finnhub_coverage_is_decided_by_date(tmp_path):
    """Out-of-range Finnhub requests return 200 + empty, so we must pre-check."""
    now = 1_000_000_000
    recent = Window("AAA", now - 10 * DAY, now, 1)
    ancient = Window("AAA", now - 400 * DAY, now - 397 * DAY, 1)
    assert finnhub_covers(recent, _cfg(), now=now)
    assert not finnhub_covers(ancient, _cfg(), now=now)


def test_span_bookkeeping_is_per_api(tmp_path):
    """Fetching a window from GDELT must not mark it done for Finnhub."""
    conn = db.get_conn(tmp_path / "t.db")
    db.mark_news_span(conn, "AAA", 100, 200, "gdelt", 7, 1_000)
    assert db.fetched_news_spans(conn, "gdelt") == {("AAA", 100, 200)}
    assert db.fetched_news_spans(conn, "finnhub") == set()


def test_refetching_a_span_updates_rather_than_duplicates(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    db.mark_news_span(conn, "AAA", 100, 200, "gdelt", 7, 1_000)
    db.mark_news_span(conn, "AAA", 100, 200, "gdelt", 9, 2_000)
    rows = conn.execute("SELECT n_rows, fetched_utc FROM news_spans").fetchall()
    assert [tuple(r) for r in rows] == [(9, 2_000)]


def _collect_cfg(tmp_path):
    cfg = _cfg()
    cfg["news"].update({"gdelt_min_interval_s": 0, "finnhub_min_interval_s": 0,
                        "max_records": 10, "gdelt_base": "x"})
    cfg["reddit"] = {"user_agent": "test"}
    cfg["tickers"] = {"universe_csv": str(tmp_path / "u.csv")}
    return cfg


def test_gap_scope_asks_gdelt_only_where_finnhub_cannot_reach(tmp_path, monkeypatch):
    """GDELT answers ~1 call in 3; spending it on windows Finnhub covers wastes hours."""
    from src.pipeline import event_news

    conn = db.get_conn(tmp_path / "t.db")
    now = utc_now_ts()
    _event(conn, "OLD-1", "OLD", now - 500 * DAY)
    _event(conn, "NEW-1", "NEW", now - 10 * DAY)
    (tmp_path / "u.csv").write_text("ticker,name,name_match\n")

    asked = []
    monkeypatch.setattr(event_news, "_fetch",
                        lambda api, cfg, s, w, u, k: asked.append((api, w.ticker)) or [])
    event_news.collect(conn, _collect_cfg(tmp_path), ["gdelt"], gdelt_scope="gap")
    assert asked == [("gdelt", "OLD")]

    asked.clear()
    conn.execute("DELETE FROM news_spans")
    event_news.collect(conn, _collect_cfg(tmp_path), ["gdelt"], gdelt_scope="all")
    assert sorted(t for _, t in asked) == ["NEW", "OLD"]


def test_a_failed_query_stays_outstanding(tmp_path, monkeypatch):
    """Marking a throttled query 'done' would make §6.4 read silence as FALSE."""
    from src.collectors.news import NewsUnavailable
    from src.pipeline import event_news

    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA", 1_000_000)
    (tmp_path / "u.csv").write_text("ticker,name,name_match\n")

    def always_throttled(*a, **k):
        raise NewsUnavailable("GDELT gave up")

    monkeypatch.setattr(event_news, "_fetch", always_throttled)
    stats = event_news.collect(conn, _collect_cfg(tmp_path), ["gdelt"],
                               gdelt_scope="all")
    assert stats["gdelt_failed"] == 1
    assert db.fetched_news_spans(conn, "gdelt") == set()   # retried next run


def test_an_empty_but_successful_query_is_marked_done(tmp_path, monkeypatch):
    """'We asked and there was nothing' is an answer worth not re-asking."""
    from src.pipeline import event_news

    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA", 1_000_000)
    (tmp_path / "u.csv").write_text("ticker,name,name_match\n")

    monkeypatch.setattr(event_news, "_fetch", lambda *a, **k: [])
    event_news.collect(conn, _collect_cfg(tmp_path), ["gdelt"], gdelt_scope="all")
    assert len(db.fetched_news_spans(conn, "gdelt")) == 1


def test_news_in_window_filters_by_ticker_and_time(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    db.upsert_news(conn, [
        ("u1", "AAA", "inside", "reuters.com", 150, "gdelt"),
        ("u2", "AAA", "too late", "reuters.com", 500, "gdelt"),
        ("u3", "BBB", "wrong ticker", "reuters.com", 150, "gdelt"),
    ])
    got = db.news_in_window(conn, "AAA", 100, 200)
    assert [r["title"] for r in got] == ["inside"]
