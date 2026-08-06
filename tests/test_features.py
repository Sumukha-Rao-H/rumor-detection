"""Feature builder tests — §7, with the leakage guarantee as the centrepiece."""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.pipeline.features import (MARKET_DIM, SOCIAL_DIM, STATE_DIM, TEXT_DIM,
                                   Post, TextEncoder, build_event,
                                   claim_type_onehot, market_block, social_block)

HOUR = 3600
DAY = 86400
T0 = 1_700_000_000 // DAY * DAY + 12 * HOUR


def _cfg(tmp_path):
    return {
        "state": {"embed_model": "x", "sent_model": "y", "dim": STATE_DIM},
        "reward": {"T_max": 48},
        "market": {"interval": "60m", "benchmark": "SPY"},
        "paths": {"db": str(tmp_path / "t.db")},
    }


def _bars(conn, ticker, start, n, step=HOUR, price=100.0, volume=1000.0):
    rows = [(ticker, start + i * step, price, price * 1.01, price * 0.99,
             price + i * 0.5, volume + i, "60m") for i in range(n)]
    db.upsert_bars(conn, rows)


def test_the_state_vector_is_the_width_the_plan_specifies():
    assert TEXT_DIM + SOCIAL_DIM + MARKET_DIM + 1 == STATE_DIM == 418


def test_claim_type_one_hot_falls_back_to_other():
    assert claim_type_onehot("merger").argmax() == 0
    assert claim_type_onehot("not-a-type").argmax() == 7   # 'other'
    assert claim_type_onehot(None).sum() == 1.0


# --- the rule that matters --------------------------------------------------

def test_social_features_ignore_posts_from_the_future():
    """A post at t0+10h must not affect the row for t0+2h."""
    posts = [Post(T0, "a", 10, 0.9, 3), Post(T0 + 10 * HOUR, "b", 999, 0.5, 80)]
    early = social_block(posts, T0, T0 + 2 * HOUR, 48)
    alone = social_block(posts[:1], T0, T0 + 2 * HOUR, 48)
    assert np.allclose(early, alone)
    # ... and the later row does see it.
    late = social_block(posts, T0, T0 + 12 * HOUR, 48)
    assert not np.allclose(early, late)


def test_market_features_ignore_bars_from_the_future(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _bars(conn, "AAA", T0 - 5 * HOUR, 20)
    bars = db.bars_in_range(conn, "AAA", T0 - 10 * DAY, T0 + 10 * DAY, "60m")

    now = T0 + 2 * HOUR
    full = market_block(bars, [], T0, now)
    truncated = market_block([b for b in bars if b["ts_utc"] <= now], [], T0, now)
    assert np.allclose(full, truncated)


def test_every_row_only_uses_its_own_past(tmp_path):
    """The whole tensor, rebuilt with the future deleted, is unchanged."""
    conn = db.get_conn(tmp_path / "t.db")
    _bars(conn, "AAA", T0 - 2 * DAY, 24 * 4)
    _bars(conn, "SPY", T0 - 2 * DAY, 24 * 4)
    db.upsert_posts(conn, [{
        "id": "p1", "subreddit": "stocks", "title": "AAA merger",
        "selftext": "", "author": "u", "created_utc": T0, "score": 10,
        "upvote_ratio": 0.9, "num_comments": 2, "flair": None, "url": "",
        "source": "test", "fetched_utc": T0,
    }])
    row = pd.Series({"event_id": "AAA-1", "ticker": "AAA", "t0_utc": T0,
                     "claim_summary": "AAA is merging", "claim_type": "merger",
                     "label": 1, "t_official_utc": T0 + 40 * HOUR,
                     "post_ids": ["p1"]})
    cfg = _cfg(tmp_path)
    encoder = TextEncoder(cfg, prefer="hashed")

    full = build_event(conn, cfg, encoder, row)["X"]
    # Descending, so each deletion is a superset of the last — otherwise the
    # later rows would be missing history they were entitled to.
    for t in (20, 5, 0):
        conn.execute("DELETE FROM bars WHERE ts_utc > ?", (T0 + t * HOUR,))
        conn.commit()
        again = build_event(conn, cfg, encoder, row)["X"]
        assert np.allclose(full[t], again[t]), f"row {t} used future data"


def test_states_are_finite_and_the_right_shape(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _bars(conn, "AAA", T0 - DAY, 48)
    row = pd.Series({"event_id": "AAA-1", "ticker": "AAA", "t0_utc": T0,
                     "claim_summary": "c", "claim_type": "other", "label": 0,
                     "t_official_utc": None, "post_ids": []})
    cfg = _cfg(tmp_path)
    data = build_event(conn, cfg, TextEncoder(cfg, prefer="hashed"), row)
    assert data["X"].shape == (48, STATE_DIM)
    assert np.isfinite(data["X"]).all()


def test_an_event_with_no_market_data_still_builds(tmp_path):
    """Thin tickers must not crash the pipeline — they read as zeros."""
    conn = db.get_conn(tmp_path / "t.db")
    row = pd.Series({"event_id": "ZZZ-1", "ticker": "ZZZ", "t0_utc": T0,
                     "claim_summary": "c", "claim_type": "other", "label": 0,
                     "t_official_utc": None, "post_ids": []})
    cfg = _cfg(tmp_path)
    data = build_event(conn, cfg, TextEncoder(cfg, prefer="hashed"), row)
    assert np.isfinite(data["X"]).all()
    assert np.allclose(data["X"][:, TEXT_DIM + SOCIAL_DIM:TEXT_DIM + SOCIAL_DIM + MARKET_DIM], 0)


def test_the_hashed_encoder_is_deterministic(tmp_path):
    cfg = _cfg(tmp_path)
    enc = TextEncoder(cfg, prefer="hashed")
    assert np.allclose(enc.embed("a merger is coming"), enc.embed("a merger is coming"))
    assert not np.allclose(enc.embed("merger"), enc.embed("bankruptcy"))
