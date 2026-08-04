"""LLM triage tests (plan §6.2 stage 2). No network is touched."""

import json

from src import db
from src.pipeline.triage import (
    build_prompt,
    content_key,
    normalize,
    pending_pairs,
    refresh_events,
    triage,
)


def _cfg():
    return {"llm": {"prompt_version": "v1"},
            "triage": {"priority_keywords": ["merger", "FDA"]}}


class FakeClient:
    """Answers from a {title-substring: reply} table; records what it was asked."""

    def __init__(self, replies, default=None):
        self.replies = replies
        self.default = default or {"is_rumor": False, "claim_summary": "",
                                   "claim_type": "other"}
        self.prompts = []
        self.calls = 0
        self.cache_hits = 0

    def complete_json(self, prompt, cache_key):
        self.prompts.append(prompt)
        self.calls += 1
        for needle, reply in self.replies.items():
            if needle in prompt:
                return _Reply(reply)
        return _Reply(self.default)


class _Reply:
    def __init__(self, data):
        self.data = data
        self.provider = "fake"
        self.model = "fake-1"
        self.cached = False


def _seed_event(conn, event_id, ticker, posts):
    """posts: [(id, title, score)]"""
    db.upsert_posts(conn, [{"id": p[0], "title": p[1], "selftext": "",
                            "created_utc": 1_740_000_000, "score": p[2],
                            "subreddit": "stocks", "source": "arctic"}
                           for p in posts])
    db.upsert_events(conn, [{
        "event_id": event_id, "ticker": ticker, "t0_utc": 1_740_000_000,
        "claim_summary": None, "claim_type": None,
        "post_ids": json.dumps([p[0] for p in posts]),
        "n_posts": len(posts), "subreddits": json.dumps(["stocks"]),
    }])


def test_normalize_coerces_model_sloppiness():
    assert normalize({"is_rumor": "true", "claim_summary": " x ",
                      "claim_type": "MERGER"}) == {
        "is_rumor": 1, "claim_summary": "x", "claim_type": "merger"}
    # Unknown type falls back to the Appendix D.1 catch-all
    assert normalize({"is_rumor": True, "claim_type": "acquisition-ish",
                      "claim_summary": "y"})["claim_type"] == "other"
    # Missing fields must not raise
    assert normalize({})["is_rumor"] == 0
    assert normalize({"is_rumor": True, "claim_summary": ""})["claim_summary"] is None


def test_prompt_carries_ticker_and_truncates_body():
    from src.pipeline.triage import BODY_CHARS, Pair
    pair = Pair("p1", "NVDA", "title", "x" * 5000, 10, "e1", 2)
    prompt = build_prompt(pair)
    assert "TICKER: NVDA" in prompt
    assert prompt.count("x") == BODY_CHARS


def test_content_key_is_identical_for_identical_prompts():
    assert content_key("same") == content_key("same")
    assert content_key("a") != content_key("b")


def test_pending_pairs_is_seed_first(tmp_path):
    """Every event's top post is judged before any event's second post."""
    conn = db.get_conn(tmp_path / "t.db")
    _seed_event(conn, "A-1", "AAA", [("a1", "low", 1), ("a2", "high", 90)])
    _seed_event(conn, "B-1", "BBB", [("b1", "mid", 50), ("b2", "low", 2)])

    order = [(p.post_id, p.ticker) for p in pending_pairs(conn, _cfg(), "v1")]
    assert order[:2] == [("a2", "AAA"), ("b1", "BBB")]     # seeds
    assert sorted(order[2:]) == [("a1", "AAA"), ("b2", "BBB")]


def test_priority_keywords_come_first_within_a_rank(tmp_path):
    """Quota-bound runs must spend calls on discrete claims, not chatter."""
    conn = db.get_conn(tmp_path / "t.db")
    _seed_event(conn, "A-1", "AAA", [("a1", "just vibes here", 900)])
    _seed_event(conn, "B-1", "BBB", [("b1", "rumored merger with X", 5)])
    order = [p.post_id for p in pending_pairs(conn, _cfg(), "v1")]
    assert order == ["b1", "a1"]   # despite a1 scoring far higher


def test_seeds_only_takes_one_post_per_event(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _seed_event(conn, "A-1", "AAA", [("a1", "low", 1), ("a2", "high", 90)])
    _seed_event(conn, "B-1", "BBB", [("b1", "mid", 50)])
    pairs = pending_pairs(conn, _cfg(), "v1", seeds_only=True)
    assert [p.post_id for p in pairs] == ["a2", "b1"]


def test_pending_pairs_skips_already_triaged(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _seed_event(conn, "A-1", "AAA", [("a1", "t", 5)])
    db.upsert_triage(conn, [{"post_id": "a1", "ticker": "AAA", "is_rumor": 1,
                             "claim_summary": "s", "claim_type": "merger",
                             "provider": "p", "model": "m",
                             "prompt_version": "v1", "created_utc": 1}])
    assert pending_pairs(conn, _cfg(), "v1") == []
    # A new prompt version re-opens the work
    assert len(pending_pairs(conn, _cfg(), "v2")) == 1


def test_triage_writes_verdicts_and_drops_empty_claims(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _seed_event(conn, "A-1", "AAA", [("a1", "merger news", 5), ("a2", "meme", 1)])
    client = FakeClient({
        "merger news": {"is_rumor": True, "claim_summary": "AAA merging with X",
                        "claim_type": "merger"},
        # "rumor" with no claim is self-contradictory -> not a rumor
        "meme": {"is_rumor": True, "claim_summary": "", "claim_type": "other"},
    })
    stats = triage(conn, _cfg(), client)
    assert stats["done"] == 2 and stats["rumor"] == 1

    rows = {r[0]: r for r in conn.execute(
        "SELECT post_id, is_rumor, claim_summary, claim_type FROM post_triage")}
    assert rows["a1"][1] == 1 and rows["a1"][3] == "merger"
    assert rows["a2"][1] == 0


def test_refresh_events_uses_top_scoring_rumor_as_seed(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _seed_event(conn, "A-1", "AAA",
                [("a1", "quiet", 5), ("a2", "loud", 900), ("a3", "meme", 1)])
    client = FakeClient({
        "quiet": {"is_rumor": True, "claim_summary": "quiet claim",
                  "claim_type": "legal"},
        "loud": {"is_rumor": True, "claim_summary": "loud claim",
                 "claim_type": "merger"},
    })
    triage(conn, _cfg(), client)
    assert refresh_events(conn) == {"events": 1, "rumor_events": 1}

    row = conn.execute("SELECT * FROM events WHERE event_id='A-1'").fetchone()
    assert row["claim_summary"] == "loud claim"   # highest score wins
    assert row["claim_type"] == "merger"
    assert row["n_rumor_posts"] == 2


def test_refresh_events_marks_events_with_no_rumor(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _seed_event(conn, "A-1", "AAA", [("a1", "meme", 5)])
    triage(conn, _cfg(), FakeClient({}))
    assert refresh_events(conn) == {"events": 1, "rumor_events": 0}
    row = conn.execute("SELECT * FROM events WHERE event_id='A-1'").fetchone()
    assert row["n_rumor_posts"] == 0 and row["claim_summary"] is None
