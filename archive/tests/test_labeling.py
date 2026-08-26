"""Ground-truth labeling tests (plan §6.4) — rule logic only, no network:
label_event() reads from pre-seeded news/bars tables, it never calls the
collectors itself (that's ensure_ground_truth_fetched, tested separately
in a live smoke run, not here)."""

from src import db
from src.pipeline import labeling
from src.utils.config import load_config


def make_event(event_id, ticker, t0):
    return {"event_id": event_id, "ticker": ticker, "t0_utc": t0,
            "claim_summary": "test", "claim_type": "merger", "post_ids": "p1",
            "label": None, "t_official_utc": None, "label_source": None,
            "human_reviewed": 0}


def test_true_on_whitelisted_news(tmp_path):
    cfg = load_config()
    conn = db.get_conn(tmp_path / "t.db")
    db.upsert_news(conn, [("https://reuters.com/a", "TSLA", "TSLA merger confirmed",
                          "reuters.com", 3600, "gdelt")])

    label, t_official, source = labeling.label_event(conn, cfg, make_event("TSLA_0", "TSLA", 0))
    assert (label, t_official, source) == (1, 3600, "news_match_auto")


def test_ignores_non_whitelisted_domain(tmp_path):
    cfg = load_config()
    conn = db.get_conn(tmp_path / "t.db")
    db.upsert_news(conn, [("https://randomblog.com/a", "TSLA", "TSLA merger???",
                          "randomblog.com", 3600, "gdelt")])

    label, _, source = labeling.label_event(conn, cfg, make_event("TSLA_0", "TSLA", 0))
    assert label is None
    assert source == "ambiguous_insufficient_price_history"


def test_false_when_no_news_and_price_calm(tmp_path):
    cfg = load_config()
    conn = db.get_conn(tmp_path / "t.db")
    interval = cfg["market"]["interval"]
    bars = [("TSLA", ts, 100, 100, 100, 100, 1_000_000, interval)
            for ts in range(-720 * 3600, 72 * 3600 + 1, 3600)]
    db.upsert_bars(conn, bars)

    label, t_official, source = labeling.label_event(conn, cfg, make_event("TSLA_0", "TSLA", 0))
    assert label == 0
    assert source == "no_news_no_price_move_auto"
    assert t_official == cfg["event"]["label_horizon_hours"] * 3600


def test_ambiguous_when_price_spikes_with_no_news(tmp_path):
    cfg = load_config()
    conn = db.get_conn(tmp_path / "t.db")
    interval = cfg["market"]["interval"]
    baseline = [("TSLA", ts, 100, 100, 100, 100, 1_000_000, interval)
                for ts in range(-720 * 3600, 0, 3600)]
    spike = [("TSLA", ts, 100, 130, 100, 130, 1_000_000, interval)
             for ts in range(0, 72 * 3600 + 1, 3600)]
    db.upsert_bars(conn, baseline + spike)

    label, _, source = labeling.label_event(conn, cfg, make_event("TSLA_0", "TSLA", 0))
    assert label is None
    assert source == "ambiguous_price_moved_no_news"


def test_ambiguous_when_insufficient_bar_history(tmp_path):
    cfg = load_config()
    conn = db.get_conn(tmp_path / "t.db")
    label, _, source = labeling.label_event(conn, cfg, make_event("TSLA_0", "TSLA", 0))
    assert label is None
    assert source == "ambiguous_insufficient_price_history"


def test_label_all_updates_events_table(tmp_path, monkeypatch):
    cfg = load_config()
    conn = db.get_conn(tmp_path / "t.db")
    db.upsert_event(conn, make_event("TSLA_0", "TSLA", 0))
    db.upsert_news(conn, [("https://reuters.com/a", "TSLA", "TSLA merger confirmed",
                          "reuters.com", 3600, "gdelt")])

    monkeypatch.setattr(labeling, "ensure_ground_truth_fetched", lambda *a, **k: None)
    counts = labeling.label_all(conn, cfg)
    assert counts == {"true": 1, "false": 0, "ambiguous": 0}
    row = conn.execute("SELECT * FROM events WHERE event_id='TSLA_0'").fetchone()
    assert row["label"] == 1 and row["human_reviewed"] == 0
