"""Rule-based baseline WAIT/COMMIT policy tests."""

from src import db
from src.baselines.rule_based import decide_step, price_signal, simulate_event
from src.utils.config import load_config


def make_event(event_id, ticker, t0, claim_type, label=None, t_official_utc=None):
    return {"event_id": event_id, "ticker": ticker, "t0_utc": t0, "claim_summary": "test",
            "claim_type": claim_type, "post_ids": "p1", "label": label,
            "t_official_utc": t_official_utc, "label_source": None, "human_reviewed": 0}


def test_decide_step_waits_before_min_wait_hours():
    cfg = load_config()
    assert decide_step(cfg, "merger", hour=1, ret=0.10, z=5.0) == "WAIT"


def test_decide_step_waits_with_no_price_data():
    cfg = load_config()
    assert decide_step(cfg, "merger", hour=10, ret=None, z=None) == "WAIT"


def test_decide_step_commits_true_on_expected_direction():
    cfg = load_config()
    # "merger" is a positive-signal claim: price up => commit TRUE
    assert decide_step(cfg, "merger", hour=5, ret=0.05, z=0.5) == 1


def test_decide_step_commits_false_on_opposite_direction():
    cfg = load_config()
    assert decide_step(cfg, "merger", hour=5, ret=-0.05, z=0.5) == 0


def test_decide_step_negative_claim_commits_true_on_drop():
    cfg = load_config()
    # "bankrupt" is a negative-signal claim: price DOWN confirms it
    assert decide_step(cfg, "bankrupt", hour=5, ret=-0.05, z=0.5) == 1


def test_decide_step_neutral_claim_uses_magnitude_only():
    cfg = load_config()
    assert decide_step(cfg, "guidance", hour=5, ret=0.05, z=0.1) == 1   # big return either sign
    assert decide_step(cfg, "guidance", hour=5, ret=-0.05, z=0.1) == 1
    assert decide_step(cfg, "guidance", hour=5, ret=0.001, z=0.1) == "WAIT"


def test_decide_step_forces_false_at_timeout():
    cfg = load_config()
    t_max = cfg["reward"]["T_max"]
    assert decide_step(cfg, "guidance", hour=t_max, ret=0.001, z=0.1) == 0


def test_price_signal_no_lookahead(tmp_path):
    cfg = load_config()
    conn = db.get_conn(tmp_path / "t.db")
    interval = cfg["market"]["interval"]
    baseline = [("TSLA", ts, 100, 100, 100, 100, 1_000_000, interval)
                for ts in range(-720 * 3600, 0, 3600)]
    # price jumps at hour 10, stays flat until hour 20 (should NOT be visible at hour 5)
    early = [("TSLA", ts, 100, 100, 100, 100, 1_000_000, interval) for ts in range(0, 10 * 3600, 3600)]
    late = [("TSLA", ts, 130, 130, 130, 130, 1_000_000, interval) for ts in range(10 * 3600, 20 * 3600, 3600)]
    db.upsert_bars(conn, baseline + early + late)

    ret_at_5, _ = price_signal(conn, "TSLA", 0, hour=5, interval=interval)
    ret_at_15, _ = price_signal(conn, "TSLA", 0, hour=15, interval=interval)
    assert abs(ret_at_5) < 0.01           # no jump visible yet
    assert ret_at_15 > 0.05               # jump visible once it's happened


def test_simulate_event_commits_and_scores_correctness(tmp_path):
    cfg = load_config()
    conn = db.get_conn(tmp_path / "t.db")
    interval = cfg["market"]["interval"]
    baseline = [("TSLA", ts, 100, 100, 100, 100, 1_000_000, interval)
                for ts in range(-720 * 3600, 0, 3600)]
    jump = [("TSLA", ts, 110, 110, 110, 110, 1_000_000, interval)
            for ts in range(3 * 3600, 48 * 3600, 3600)]  # +10% from hour 3 onward
    db.upsert_bars(conn, baseline + jump)

    event = make_event("TSLA_0", "TSLA", 0, "merger", label=1, t_official_utc=20 * 3600)
    result = simulate_event(conn, cfg, event)

    assert result["verdict"] == 1
    assert result["committed_hour"] == 3   # first hour min_wait_hours has passed AND signal present
    assert result["correct"] is True
    assert result["delta_hours"] == 17     # 20h official minus 3h committed


def test_simulate_event_times_out_to_false_with_no_signal(tmp_path):
    cfg = load_config()
    conn = db.get_conn(tmp_path / "t.db")
    interval = cfg["market"]["interval"]
    bars = [("TSLA", ts, 100, 100, 100, 100, 1_000_000, interval)
            for ts in range(-720 * 3600, 48 * 3600, 3600)]
    db.upsert_bars(conn, bars)

    event = make_event("TSLA_0", "TSLA", 0, "guidance", label=0, t_official_utc=72 * 3600)
    result = simulate_event(conn, cfg, event)

    assert result["verdict"] == 0
    assert result["committed_hour"] == cfg["reward"]["T_max"]
    assert result["correct"] is True
