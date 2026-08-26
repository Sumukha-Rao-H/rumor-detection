"""A simple, explainable, non-AI WAIT/COMMIT policy (plan's "baselines").

Built ahead of the RL agent on purpose: it gives an early, honest demo and a
yardstick the eventual AI model has to actually beat. At each hourly step
after an event's t0, it looks only at price/volume data available up to
that hour (never anything later — no lookahead) and decides WAIT or COMMIT.

Decision rule, per hour:
  - Always WAIT before `min_wait_hours`.
  - Compute the cumulative price return and a volume z-score since t0, using
    only bars timestamped <= that hour (mirrors labeling.py's price check,
    but incremental instead of over the whole window).
  - Some rumor types have a predictable direction if true (a merger/approval
    rumor should push price up if real; a bankruptcy/recall rumor should
    push it down). For those, COMMIT TRUE once price moves far enough in
    the expected direction, COMMIT FALSE if it moves the opposite way.
  - For direction-neutral claim types, COMMIT TRUE once EITHER the price
    move or the volume z-score is large, regardless of sign.
  - If nothing crosses threshold by `reward.T_max` hours, COMMIT FALSE
    (timeout default) — mirrors labeling.py's own "no signal = false" rule.

Usage:
  python -m src.baselines.rule_based                  # simulate every labeled event
  python -m src.baselines.rule_based --event-id TSLA_0
"""

from __future__ import annotations

import argparse
import logging
import statistics

from src import db
from src.utils.config import load_config

log = logging.getLogger(__name__)

MIN_BASELINE_BARS = 5 * 24  # ~5 trading days of hourly bars needed for a baseline


def price_signal(
    conn, ticker: str, t0: int, hour: int, interval: str,
) -> tuple[float | None, float | None]:
    """(cumulative_return, volume_zscore) using only bars up to t0 + hour.
    (None, None) if there isn't enough baseline history to judge anything."""
    baseline = conn.execute(
        """SELECT close, volume FROM bars WHERE ticker = ? AND interval = ? AND ts_utc < ?
           ORDER BY ts_utc DESC LIMIT 720""",
        (ticker, interval, t0),
    ).fetchall()
    window = conn.execute(
        """SELECT close, volume FROM bars WHERE ticker = ? AND interval = ?
           AND ts_utc BETWEEN ? AND ? ORDER BY ts_utc ASC""",
        (ticker, interval, t0, t0 + hour * 3600),
    ).fetchall()
    if len(baseline) < MIN_BASELINE_BARS or not window:
        return None, None

    pre_event_close = baseline[0]["close"]
    ret = (window[-1]["close"] - pre_event_close) / pre_event_close
    vols = [r["volume"] for r in baseline]
    mean_v, std_v = statistics.fmean(vols), statistics.pstdev(vols) or 1.0
    window_mean_v = statistics.fmean([r["volume"] for r in window])
    z = (window_mean_v - mean_v) / std_v
    return ret, z


def decide_step(cfg: dict, claim_type: str, hour: int, ret: float | None, z: float | None):
    """WAIT, or an int verdict (1 = COMMIT TRUE, 0 = COMMIT FALSE)."""
    bcfg = cfg["baseline"]
    t_max = cfg["reward"]["T_max"]

    if hour < bcfg["min_wait_hours"] or ret is None:
        return "WAIT" if hour < t_max else 0

    positive = claim_type in bcfg["positive_signal_claims"]
    negative = claim_type in bcfg["negative_signal_claims"]
    r_thresh, z_thresh = bcfg["return_threshold"], bcfg["zscore_threshold"]

    if positive:
        if ret >= r_thresh:
            return 1
        if ret <= -r_thresh:
            return 0
    elif negative:
        if ret <= -r_thresh:
            return 1
        if ret >= r_thresh:
            return 0
    else:
        if abs(ret) >= r_thresh or abs(z) >= z_thresh:
            return 1

    return "WAIT" if hour < t_max else 0


def simulate_event(conn, cfg: dict, event: dict) -> dict:
    """Steps the policy hourly from t0 until it commits or hits T_max.
    Returns the full trace plus the final decision, and — when the event
    has a ground-truth label — whether it was correct and the resulting
    Time Delta Advantage (hours before t_official the policy committed)."""
    ticker, t0, claim_type = event["ticker"], event["t0_utc"], event["claim_type"]
    interval = cfg["market"]["interval"]
    t_max = cfg["reward"]["T_max"]

    trace = []
    verdict, committed_hour = None, None
    for hour in range(1, t_max + 1):
        ret, z = price_signal(conn, ticker, t0, hour, interval)
        decision = decide_step(cfg, claim_type, hour, ret, z)
        trace.append({"hour": hour, "return": ret, "volume_zscore": z, "decision": decision})
        if decision != "WAIT":
            verdict, committed_hour = decision, hour
            break

    result = {"event_id": event["event_id"], "trace": trace,
              "committed_hour": committed_hour, "verdict": verdict}

    label = event.get("label")
    if label is not None and verdict is not None:
        result["correct"] = (verdict == label)
        t_official = event.get("t_official_utc")
        if result["correct"] and t_official is not None:
            result["delta_hours"] = (t_official - (t0 + committed_hour * 3600)) / 3600
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", help="simulate one event instead of every labeled one")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])

    if args.event_id:
        row = conn.execute("SELECT * FROM events WHERE event_id = ?", (args.event_id,)).fetchone()
        events = [dict(row)] if row else []
    else:
        events = [dict(r) for r in conn.execute(
            "SELECT * FROM events WHERE label IS NOT NULL").fetchall()]

    correct = wrong = 0
    deltas = []
    for event in events:
        result = simulate_event(conn, cfg, event)
        if result.get("correct") is True:
            correct += 1
            if "delta_hours" in result:
                deltas.append(result["delta_hours"])
        elif result.get("correct") is False:
            wrong += 1
        log.info("%-18s verdict=%s hour=%s correct=%s",
                 event["event_id"], result["verdict"], result["committed_hour"],
                 result.get("correct"))

    log.info("Done. %d labeled events simulated: %d correct, %d wrong.",
             len(events), correct, wrong)
    if deltas:
        log.info("Mean Time Delta Advantage on correct calls: %.1f hours", statistics.fmean(deltas))


if __name__ == "__main__":
    main()
