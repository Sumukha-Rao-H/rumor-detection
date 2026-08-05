"""Label proposal tests (plan §6.4 stage 2). No network is touched."""

import json

import pytest

from src import db
from src.pipeline.labeling import (
    PRIMARY, SECONDARY, Event, Headline, build_prompt, decide, event_headlines,
    is_quiet, market_move, match_headline, normalize, pending_events, propose,
    select_headlines,
)
from src.utils.llm import LLMRefused

HOUR = 3600
DAY = 86400
T0 = 1_700_000_000 // DAY * DAY + 12 * HOUR   # midday UTC, so t0-24h stays in range


def _cfg():
    return {
        "news": {"event_pre_hours": 24,
                 "whitelist": ["reuters.com", "sec.gov"],
                 "secondary_sources": ["benzinga.com", "finance.yahoo.com"]},
        "event": {"label_horizon_hours": 72},
        "market": {"interval": "60m"},
        "labeling": {"abnormal_return": 0.04, "volume_z": 2.0, "return_days": 3,
                     "volume_baseline_days": 20, "min_baseline_days": 10,
                     "max_headlines": 40, "headline_chars": 200,
                     "prompt_version": "label-v1"},
    }


def _event(conn, event_id="A-1", ticker="AAA", t0=T0, claim="AAA will be acquired",
           n_rumor_posts=1, label=None):
    db.upsert_events(conn, [{
        "event_id": event_id, "ticker": ticker, "t0_utc": t0,
        "claim_summary": claim, "claim_type": "merger",
        "post_ids": json.dumps(["p"]), "n_posts": 1,
        "subreddits": json.dumps(["stocks"]),
    }])
    conn.execute("UPDATE events SET n_rumor_posts = ?, label = ?, claim_summary = ?"
                 " WHERE event_id = ?", (n_rumor_posts, label, claim, event_id))
    conn.commit()
    return Event(event_id, ticker, t0, claim)


def _row(title, domain, seen_utc):
    return {"title": title, "source_domain": domain, "seen_utc": seen_utc}


# --- source whitelist -------------------------------------------------------

def test_whitelist_accepts_subdomains_of_credible_sources():
    kept = select_headlines([_row("a", "feeds.reuters.com", 1)], _cfg())
    assert [(h.domain, h.tier) for h in kept] == [("feeds.reuters.com", PRIMARY)]


def test_whitelist_rejects_lookalike_domains():
    """reuters.com.spam.net must not launder a headline into a TRUE label."""
    assert select_headlines([_row("a", "reuters.com.spam.net", 1)], _cfg()) == []


def test_unknown_sources_are_dropped_entirely():
    rows = [_row("real", "reuters.com", 1), _row("blog", "randomblog.io", 2)]
    assert [h.title for h in select_headlines(rows, _cfg())] == ["real"]


def test_aggregators_are_kept_as_the_second_tier():
    rows = [_row("wire", "reuters.com", 1), _row("aggregated", "benzinga.com", 2)]
    assert [(h.title, h.tier) for h in select_headlines(rows, _cfg())] == [
        ("wire", PRIMARY), ("aggregated", SECONDARY)]


def test_event_headlines_reports_the_prefilter_count(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    event = _event(conn)
    db.upsert_news(conn, [
        ("u1", "AAA", "credible", "reuters.com", T0 + HOUR, "gdelt"),
        ("u2", "AAA", "noise", "randomblog.io", T0 + HOUR, "gdelt"),
        ("u3", "AAA", "outside window", "reuters.com", T0 + 200 * HOUR, "gdelt"),
    ])
    kept, n_all = event_headlines(conn, _cfg(), event)
    assert [h.title for h in kept] == ["credible"]
    assert n_all == 2          # both in-window rows, before the tier filter


# --- the market rule --------------------------------------------------------

def _bars(conn, ticker, start_day, n_days, close=100.0, volume=1000.0):
    rows = []
    for day in range(n_days):
        for hour in range(7):
            ts = (start_day + day) * DAY + (13 + hour) * HOUR
            c = close(day) if callable(close) else close
            v = volume(day) if callable(volume) else volume
            rows.append((ticker, ts, c, c, c, c, v / 7, "60m"))
    db.upsert_bars(conn, rows)


def test_no_bars_means_unknown_not_quiet(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    event = _event(conn)
    assert market_move(conn, _cfg(), event) == (None, None)
    assert not is_quiet(None, None, _cfg())


def test_a_short_baseline_refuses_to_score_volume(tmp_path):
    """A z-score off three days of history would label events FALSE on noise."""
    conn = db.get_conn(tmp_path / "t.db")
    event = _event(conn)
    _bars(conn, "AAA", T0 // DAY - 3, 8)
    _, z = market_move(conn, _cfg(), event)
    assert z is None


def test_a_flat_quiet_stock_satisfies_the_false_rule(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    event = _event(conn)
    _bars(conn, "AAA", T0 // DAY - 21, 26, close=lambda d: 100.0 + (d % 2) * 0.1,
          volume=lambda d: 1000.0 + (d % 3))
    ret, z = market_move(conn, _cfg(), event)
    assert abs(ret) < 0.04 and z < 2.0
    assert is_quiet(ret, z, _cfg())


def test_a_price_spike_blocks_the_false_rule(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    event = _event(conn)
    _bars(conn, "AAA", T0 // DAY - 21, 26,
          close=lambda d: 100.0 if d < 21 else 130.0)
    ret, z = market_move(conn, _cfg(), event)
    assert ret > 0.04
    assert not is_quiet(ret, z, _cfg())


def test_a_volume_spike_blocks_the_false_rule(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    event = _event(conn)
    _bars(conn, "AAA", T0 // DAY - 21, 26, volume=lambda d: 1000.0 + (d % 3)
          if d < 21 else 50_000.0)
    ret, z = market_move(conn, _cfg(), event)
    assert z > 2.0
    assert not is_quiet(ret, z, _cfg())


# --- headline matching ------------------------------------------------------

def test_match_headline_ignores_punctuation_and_case():
    heads = [Headline("Acme Corp. to be ACQUIRED", "reuters.com", 50)]
    assert match_headline("acme corp to be acquired", heads).seen_utc == 50


def test_match_headline_accepts_a_truncated_quote():
    heads = [Headline("Acme Corp to be acquired by Globex for $2bn", "reuters.com", 50)]
    assert match_headline("Acme Corp to be acquired", heads) is not None


def test_match_headline_prefers_the_earliest_of_several(tmp_path):
    heads = [Headline("Acme acquired by Globex", "reuters.com", 90),
             Headline("Acme acquired by Globex", "sec.gov", 50)]
    assert match_headline("Acme acquired by Globex", heads).seen_utc == 50


def test_match_headline_returns_none_for_an_invention():
    heads = [Headline("Acme wins contract", "reuters.com", 50)]
    assert match_headline("Acme files for bankruptcy", heads) is None


# --- reply normalization ----------------------------------------------------

def test_normalize_rejects_a_verdict_outside_the_enum():
    with pytest.raises(LLMRefused):
        normalize({"verdict": "PROBABLY", "confidence": 0.9})


def test_normalize_rejects_a_non_object():
    with pytest.raises(LLMRefused):
        normalize(["TRUE"])


def test_normalize_survives_a_missing_confidence():
    assert normalize({"verdict": "TRUE"})["confidence"] is None


# --- §6.4 decision table ----------------------------------------------------

def _fields(verdict, headline=None, confidence=0.9):
    return {"llm_verdict": verdict, "deciding_headline": headline,
            "confidence": confidence}


def test_a_confirmed_claim_takes_its_timestamp_from_our_news_row():
    """t_official drives the reward's early bonus, so it must be observed."""
    event = Event("A-1", "AAA", T0, "acquired")
    heads = [Headline("Acme acquired by Globex", "reuters.com", T0 + 5 * HOUR)]
    got = decide(event, _fields("TRUE", "Acme acquired by Globex"), heads,
                 0.10, 3.0, _cfg())
    assert got["verdict"] == "TRUE" and got["rule"] == "confirmed"
    assert got["t_official_utc"] == T0 + 5 * HOUR


def test_a_confirmation_predating_the_post_is_flagged():
    """News that broke before Reddit posted is a repost, not a rumor."""
    event = Event("A-1", "AAA", T0, "acquired")
    heads = [Headline("Acme acquired", "reuters.com", T0 - 6 * HOUR)]
    got = decide(event, _fields("TRUE", "Acme acquired"), heads, 0.10, 3.0, _cfg())
    assert got["verdict"] == "TRUE" and got["rule"] == "confirmed_pre_t0"


def test_a_true_the_model_cannot_point_at_goes_to_a_human():
    event = Event("A-1", "AAA", T0, "acquired")
    heads = [Headline("Acme wins contract", "reuters.com", T0 + HOUR)]
    got = decide(event, _fields("TRUE", "Acme acquired by Globex"), heads,
                 0.10, 3.0, _cfg())
    assert got["verdict"] == "UNVERIFIED" and got["rule"] == "unmatched_headline"
    assert got["t_official_utc"] is None


def test_a_denial_the_model_cannot_cite_is_not_trusted():
    """An uncitable denial is only 'nothing confirmed it' — let the tape decide."""
    event = Event("A-1", "AAA", T0, "acquired")
    moved = decide(event, _fields("FALSE", "Acme denies merger talk"), [],
                   0.10, 3.0, _cfg())
    assert moved["verdict"] == "UNVERIFIED" and moved["rule"] == "moved"
    quiet = decide(event, _fields("FALSE", "Acme denies merger talk"), [],
                   0.01, 0.5, _cfg())
    assert quiet["verdict"] == "FALSE" and quiet["rule"] == "quiet"


def test_an_aggregator_confirmation_goes_to_review_not_to_a_label():
    """Benzinga republishing the rumor is not the same as confirming it."""
    event = Event("A-1", "AAA", T0, "acquired")
    heads = [Headline("Acme acquired by Globex", "benzinga.com", T0 + HOUR,
                      SECONDARY)]
    got = decide(event, _fields("TRUE", "Acme acquired by Globex"), heads,
                 0.10, 3.0, _cfg())
    assert got["verdict"] == "UNVERIFIED" and got["rule"] == "confirmed_secondary"
    assert got["t_official_utc"] == T0 + HOUR   # kept for the reviewer


def test_an_aggregator_report_blocks_the_quiet_false_rule():
    """A claim that WAS reported did not go unreported — a flat tape proves
    nothing about it, so the §6.4 silence rule must not fire."""
    event = Event("A-1", "AAA", T0, "acquired")
    heads = [Headline("Acme acquired by Globex", "benzinga.com", T0 + HOUR,
                      SECONDARY)]
    got = decide(event, _fields("TRUE", "Acme acquired by Globex"), heads,
                 0.001, 0.1, _cfg())
    assert got["verdict"] == "UNVERIFIED"


def test_an_aggregator_denial_also_goes_to_review():
    event = Event("A-1", "AAA", T0, "acquired")
    heads = [Headline("Acme denies talks", "benzinga.com", T0 + HOUR, SECONDARY)]
    got = decide(event, _fields("FALSE", "Acme denies talks"), heads,
                 0.10, 3.0, _cfg())
    assert got["verdict"] == "UNVERIFIED" and got["rule"] == "denied_secondary"


def test_the_prompt_tags_each_headlines_tier():
    event = Event("A-1", "AAA", T0, "acquired")
    heads = [Headline("aggregated take", "benzinga.com", T0 + 2 * HOUR, SECONDARY),
             Headline("wire copy", "reuters.com", T0 + HOUR, PRIMARY)]
    prompt = build_prompt(event, heads, _cfg())
    assert "[CREDIBLE] (reuters.com" in prompt
    assert "[aggregator] (benzinga.com" in prompt
    # Credible first, so truncation drops the headlines that cannot decide.
    assert prompt.index("wire copy") < prompt.index("aggregated take")


def test_a_denial_uses_the_denying_headlines_time():
    event = Event("A-1", "AAA", T0, "acquired")
    heads = [Headline("Acme denies merger talk", "reuters.com", T0 + 9 * HOUR)]
    got = decide(event, _fields("FALSE", "Acme denies merger talk"), heads,
                 0.10, 3.0, _cfg())
    assert got["rule"] == "denied" and got["t_official_utc"] == T0 + 9 * HOUR


def test_unconfirmed_and_quiet_becomes_false_at_the_horizon():
    event = Event("A-1", "AAA", T0, "acquired")
    got = decide(event, _fields("UNVERIFIED"), [], 0.01, 0.5, _cfg())
    assert got["verdict"] == "FALSE" and got["rule"] == "quiet"
    assert got["t_official_utc"] == T0 + 72 * HOUR


def test_unconfirmed_but_moving_stays_unverified():
    """Something happened that our sources missed — excluded, not guessed."""
    event = Event("A-1", "AAA", T0, "acquired")
    got = decide(event, _fields("UNVERIFIED"), [], 0.25, 5.0, _cfg())
    assert got["verdict"] == "UNVERIFIED" and got["rule"] == "moved"


def test_unconfirmed_with_no_bars_is_never_labeled_false():
    """Missing market data must not read as 'the stock did not move'."""
    event = Event("A-1", "AAA", T0, "acquired")
    got = decide(event, _fields("UNVERIFIED"), [], None, None, _cfg())
    assert got["verdict"] == "UNVERIFIED" and got["rule"] == "no_bars"


# --- the run ----------------------------------------------------------------

def test_pending_skips_labeled_and_already_proposed(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA")
    _event(conn, "B-1", "BBB", label=1)
    _event(conn, "C-1", "CCC")
    db.upsert_label_proposals(conn, [{"event_id": "C-1", "verdict": "FALSE",
                                      "prompt_version": "label-v1"}])
    assert [e.event_id for e in pending_events(conn, _cfg(), "label-v1")] == ["A-1"]


def test_a_prompt_revision_reproposes(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA")
    db.upsert_label_proposals(conn, [{"event_id": "A-1", "verdict": "FALSE",
                                      "prompt_version": "label-v1"}])
    assert [e.event_id for e in pending_events(conn, _cfg(), "label-v2")] == ["A-1"]


def test_an_event_with_no_credible_headline_costs_no_api_call(tmp_path):
    """Free-tier quota is the binding constraint; silence needs no model."""
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA")
    db.upsert_news(conn, [("u1", "AAA", "blog take", "randomblog.io",
                           T0 + HOUR, "gdelt")])
    _bars(conn, "AAA", T0 // DAY - 21, 26, volume=lambda d: 1000.0 + (d % 3))

    class Exploding:
        calls = cache_hits = 0

        def complete_json(self, *a, **k):
            raise AssertionError("should not have called the model")

    stats = propose(conn, _cfg(), Exploding())
    assert stats["no_call"] == 1 and stats["verdict:FALSE"] == 1
    row = conn.execute("SELECT * FROM label_proposals").fetchone()
    assert row["rule"] == "quiet" and row["provider"] is None
    assert row["n_headlines"] == 0 and row["n_headlines_all"] == 1


def test_a_proposal_never_writes_the_events_label(tmp_path):
    """The plan requires human review; proposals must not reach training."""
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA")
    propose(conn, _cfg(), None)
    assert conn.execute("SELECT label FROM events").fetchone()["label"] is None
    assert conn.execute("SELECT COUNT(*) FROM label_proposals").fetchone()[0] == 1


def test_the_prompt_carries_the_claim_and_its_headlines():
    event = Event("A-1", "AAA", T0, "AAA will be acquired")
    heads = [Headline("Acme acquired by Globex", "reuters.com", T0 + HOUR)]
    prompt = build_prompt(event, heads, _cfg())
    assert "AAA will be acquired" in prompt
    assert "reuters.com" in prompt and "Acme acquired by Globex" in prompt
    assert "72 hours" in prompt
