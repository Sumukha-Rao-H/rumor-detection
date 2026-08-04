"""Event construction tests (plan §6.2 keyword filter, §6.3 clustering)."""

import json

import pytest

from src import db
from src.pipeline.events import (
    Candidate,
    Event,
    build_events,
    cluster,
    iter_candidates,
    keep_event,
    keyword_regex,
)

HOUR = 3600


def _cfg(tmp_path=None, **event_over):
    event = {
        "gap_hours": 12,
        "min_posts": 2,
        "min_solo_score": 50,
        "label_horizon_hours": 72,
        "exclude_etfs": False,
        "title_ticker_priority": True,
    }
    event.update(event_over)
    return {
        "backtest_window": {"start": "2025-01-01", "end": "2026-01-01"},
        "event": event,
        "rumor_keywords": ["merger", "chapter 11", "FDA", "short squeeze", "split"],
        "tickers": {"listed_csv": str(tmp_path / "listed.csv") if tmp_path else ""},
    }


class FakeExtractor:
    """Returns whichever of the known tickers appears literally in the text."""

    def __init__(self, tickers):
        self.tickers = tickers

    def extract(self, text):
        return [t for t in self.tickers if t in (text or "")]


def test_keyword_regex_is_whole_word_and_multiword():
    pattern = keyword_regex(["merger", "chapter 11", "split"])
    assert pattern.search("rumored MERGER incoming")
    assert pattern.search("filing for chapter\n11 tomorrow")   # newline inside phrase
    assert pattern.search("two mergers announced")             # plural counts
    assert not pattern.search("splitting the position")        # not a whole word
    assert not pattern.search("emerge from the dip")           # not a substring


def test_cluster_splits_on_silence_longer_than_gap():
    cands = [
        Candidate("a", "NVDA", 0, 1, "stocks"),
        Candidate("b", "NVDA", 11 * HOUR, 1, "stocks"),      # within 12h of a
        Candidate("c", "NVDA", 22 * HOUR, 1, "stocks"),      # within 12h of b
        Candidate("d", "NVDA", 40 * HOUR, 1, "stocks"),      # 18h gap -> new event
    ]
    events = cluster(cands, gap_hours=12)
    assert [len(e.candidates) for e in events] == [3, 1]
    assert [e.t0_utc for e in events] == [0, 40 * HOUR]


def test_cluster_gap_boundary_is_inclusive():
    cands = [Candidate("a", "X", 0, 1, "s"), Candidate("b", "X", 12 * HOUR, 1, "s")]
    assert len(cluster(cands, gap_hours=12)) == 1       # exactly 12h: same event
    assert len(cluster(cands, gap_hours=11.9)) == 2


def test_cluster_keeps_tickers_apart():
    cands = [
        Candidate("a", "NVDA", 0, 1, "stocks"),
        Candidate("b", "TSLA", HOUR, 1, "stocks"),
    ]
    events = cluster(cands, gap_hours=12)
    assert sorted(e.ticker for e in events) == ["NVDA", "TSLA"]


def test_cluster_by_claim_type_key():
    """§6.2 will re-cluster on (ticker, claim_type) once triage has run."""
    cands = [Candidate("a", "X", 0, 1, "s"), Candidate("b", "X", HOUR, 1, "s")]
    types = {"a": "merger", "b": "fda"}
    events = cluster(cands, 12, key=lambda c: f"{c.ticker}|{types[c.post_id]}")
    assert len(events) == 2


@pytest.mark.parametrize("n,score,expected", [
    (2, 0, True),      # meets min_posts
    (1, 50, True),     # solo but already big
    (1, 49, False),    # solo and small
])
def test_keep_event_minimum_size(n, score, expected):
    event = Event("X", 0, [Candidate(f"p{i}", "X", i, score, "s") for i in range(n)])
    assert keep_event(event, _cfg()) is expected


def test_event_id_is_deterministic_and_row_is_json():
    event = Event("NVDA", 1700, [
        Candidate("p1", "NVDA", 1700, 5, "stocks"),
        Candidate("p2", "NVDA", 1800, 5, "wallstreetbets"),
    ])
    assert event.event_id == "NVDA-1700"
    row = event.to_row()
    assert json.loads(row["post_ids"]) == ["p1", "p2"]
    assert json.loads(row["subreddits"]) == ["stocks", "wallstreetbets"]
    assert row["n_posts"] == 2 and row["claim_summary"] is None


def _seed(conn, posts):
    db.upsert_posts(conn, [{"id": p[0], "title": p[1], "selftext": p[2],
                            "created_utc": p[3], "score": p[4],
                            "subreddit": "stocks", "source": "arctic"}
                           for p in posts])
    db.link_post_tickers(conn, [(p[0], t) for p in posts for t in p[5]])


def test_title_priority_ignores_body_name_drops(tmp_path):
    """A $BBAI post whose body mentions PLTR must not seed a PLTR event."""
    conn = db.get_conn(tmp_path / "t.db")
    _seed(conn, [("p1", "$BBAI merger rumor", "could be the next PLTR", 1_740_000_000,
                  5, ["BBAI", "PLTR"])])
    cfg = _cfg()
    extractor = FakeExtractor(["BBAI", "PLTR"])
    pattern = keyword_regex(cfg["rumor_keywords"])

    with_priority = list(iter_candidates(conn, cfg, pattern, extractor))
    assert [c.ticker for c in with_priority] == ["BBAI"]

    cfg["event"]["title_ticker_priority"] = False
    assert sorted(c.ticker for c in iter_candidates(conn, cfg, pattern, extractor)) \
        == ["BBAI", "PLTR"]


def test_body_only_tickers_still_seed_events(tmp_path):
    """Long DD with a generic title must not be dropped."""
    conn = db.get_conn(tmp_path / "t.db")
    _seed(conn, [("p1", "my thoughts on this one", "NVDA merger talk",
                  1_740_000_000, 5, ["NVDA"])])
    cfg = _cfg()
    cands = list(iter_candidates(conn, cfg, keyword_regex(cfg["rumor_keywords"]),
                                 FakeExtractor(["NVDA"])))
    assert [c.ticker for c in cands] == ["NVDA"]


def test_candidates_respect_window_and_keywords(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _seed(conn, [
        ("hit", "$NVDA merger rumor", "", 1_740_000_000, 5, ["NVDA"]),
        ("no_keyword", "$NVDA looks cheap", "", 1_740_000_000, 5, ["NVDA"]),
        ("out_of_window", "$NVDA merger", "", 1_600_000_000, 5, ["NVDA"]),
    ])
    cfg = _cfg()
    cands = list(iter_candidates(conn, cfg, keyword_regex(cfg["rumor_keywords"]),
                                 FakeExtractor(["NVDA"])))
    assert [c.post_id for c in cands] == ["hit"]


def test_etfs_excluded_when_configured(tmp_path):
    (tmp_path / "listed.csv").write_text(
        "ticker,name,etf\nSPY,SPDR,1\nNVDA,Nvidia,0\n")
    conn = db.get_conn(tmp_path / "t.db")
    _seed(conn, [("p1", "$SPY $NVDA merger", "", 1_740_000_000, 5, ["SPY", "NVDA"])])
    cfg = _cfg(tmp_path, exclude_etfs=True)
    cands = list(iter_candidates(conn, cfg, keyword_regex(cfg["rumor_keywords"]),
                                 FakeExtractor(["SPY", "NVDA"])))
    assert [c.ticker for c in cands] == ["NVDA"]


def test_build_events_end_to_end(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    base = 1_740_000_000
    _seed(conn, [
        ("a", "$NVDA merger rumor", "", base, 1, ["NVDA"]),
        ("b", "$NVDA merger confirmed?", "", base + HOUR, 1, ["NVDA"]),
        ("solo_big", "$TSLA chapter 11 rumor", "", base, 99, ["TSLA"]),
        ("solo_small", "$AMD FDA approval", "", base, 1, ["AMD"]),
    ])
    events, stats = build_events(conn, _cfg(),
                                 FakeExtractor(["NVDA", "TSLA", "AMD"]))
    assert sorted(e.ticker for e in events) == ["NVDA", "TSLA"]
    assert stats["kept"] == 2 and stats["dropped_too_small"] == 1
    assert stats["posts_in_kept"] == 3


def test_upsert_events_preserves_labels_and_prune_spares_reviewed(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    event = Event("NVDA", 100, [Candidate("p1", "NVDA", 100, 1, "stocks")])
    db.upsert_events(conn, [event.to_row()])
    conn.execute("UPDATE events SET label=1, claim_summary='merger', "
                 "human_reviewed=1 WHERE event_id='NVDA-100'")
    conn.commit()

    # Re-clustering the same event must not wipe the human's work.
    event.candidates.append(Candidate("p2", "NVDA", 200, 1, "stocks"))
    db.upsert_events(conn, [event.to_row()])
    row = conn.execute("SELECT * FROM events WHERE event_id='NVDA-100'").fetchone()
    assert row["label"] == 1 and row["human_reviewed"] == 1
    assert row["claim_summary"] == "merger"
    assert row["n_posts"] == 2            # structure still refreshed

    # Nor may pruning delete it, even though clustering no longer produces it.
    assert db.prune_events(conn, set()) == 0
    conn.execute("UPDATE events SET human_reviewed=0, label=NULL")
    conn.commit()
    assert db.prune_events(conn, set()) == 1
