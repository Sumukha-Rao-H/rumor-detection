"""Did anything actually happen? Scoring the live alert log.

This is what turns "the monitor raised 101 alerts" into "of the N alerts it
raised, M were followed by a filing within 48 hours" — the one claim in this
project that is not retrospective.

Three states, not two
---------------------
The obvious implementation asks "was there an 8-K in the next 48 hours?" and
records yes or no. That is wrong, and wrong in the direction that flatters
nothing — it would make the detectors look far worse than they are.

An alert raised two hours before the end of the available filing data has not
been *checked* and found clean; it simply **cannot be answered yet**. Recording
it as a miss counts absence of data as absence of an event. With the live log
as it stands, the newest alerts are hours old and the filing feed ends the same
afternoon, so a naive backfill would score a whole day of alerts as false
positives on no evidence at all.

So an alert is scored only when its **entire** outcome window falls at or
before the filing data horizon. Everything newer is left alone — no outcome
row, so it stays in `unscored()` and is picked up automatically on the next
run. `backfill` reports how many it deferred, so "pending" is visible rather
than silent.

Two clocks, deliberately
------------------------
**The outcome window is wall-clock.** "Was there a filing in the next 48
hours?" is a question about elapsed time: a company can file at any hour,
including overnight and at weekends, and the plan's claim is phrased that way.

**The lead time recorded is in trading hours**, per rule 3, because that is
what every other lead-time figure in this project means and a number that
changed units between the offline table and the live log would be unreadable.

Both are stored. Conflating them would be easy and silent — `decision.
horizon_hours` is 48 in BARS, the outcome window is 48 in wall-clock hours, and
they are different quantities that happen to share a number.

Usage:
  python -m src.live.outcomes                 # score what can be scored
  python -m src.live.outcomes --report        # the hit rate per detector
"""

from __future__ import annotations

import argparse

from src.utils.config import load_config
from src.utils.timeutils import trading_hours_between, ts_to_iso, utc_now_ts


def data_horizon(conn) -> int | None:
    """The newest filing acceptance time held locally.

    An alert whose outcome window extends past this cannot be answered, however
    long ago it was raised — the answer depends on data that has not been
    collected, not on data that does not exist.
    """
    row = conn.execute(
        "SELECT MAX(acceptance_utc) FROM filings "
        "WHERE acceptance_utc IS NOT NULL").fetchone()
    return None if row is None or row[0] is None else int(row[0])


def window_seconds(cfg: dict) -> int:
    """The outcome window, in WALL-CLOCK seconds.

    Not `decision.horizon_hours`, which is 48 BARS — a different quantity that
    happens to share the number 48. A filing can land overnight or at a
    weekend, so the question "did one follow within two days" is a wall-clock
    question.
    """
    hours = (cfg.get("live") or {}).get("outcome_window_hours", 48)
    return int(hours) * 3600


def first_filing_after(conn, ticker: str, after_ts: int,
                       until_ts: int, forms: tuple[str, ...]) -> dict | None:
    """The earliest qualifying filing in (after_ts, until_ts].

    Strictly after the alert: a filing accepted in the same second the alert
    fired was not predicted by it. Earliest rather than any, because that is
    the one the alert would have been anticipating.

    `t0_utc` prefers the event row's corrected instant — min(acceptance,
    earliest matched news) — and falls back to raw acceptance when the filing
    has no event row, which is the same fallback `sampling.py` uses.

    The window is measured on that corrected instant, NOT on acceptance time.
    Selecting on acceptance while reporting t0 was a real defect: t0 is
    min(acceptance, earliest news) and so can be up to `news.t0_lookback_hours`
    EARLIER than acceptance, so a filing accepted after the alert could carry a
    t0 before it. That is a filing whose news was already public when the alert
    fired — the alert did not anticipate it — and it made
    `trading_hours_between` raise on a negative lead, aborting the whole
    backfill before it committed. Asking the question about the same instant
    that gets reported fixes both at once: the filing simply is not one that
    followed the alert, and the next qualifying filing is considered instead.
    """
    marks = ",".join("?" * len(forms))
    # COALESCE, not f.acceptance_utc: the public instant is what "did a filing
    # follow this alert" is asking about, and it is what the row reports.
    row = conn.execute(
        f"SELECT f.accession_no, f.items, f.acceptance_utc, "
        f"       COALESCE(e.t0_utc, f.acceptance_utc) AS t0_utc "
        f"FROM filings f LEFT JOIN events e ON e.accession_no = f.accession_no "
        f"WHERE f.ticker = ? AND f.form IN ({marks}) "
        f"AND COALESCE(e.t0_utc, f.acceptance_utc) > ? "
        f"AND COALESCE(e.t0_utc, f.acceptance_utc) <= ? "
        f"ORDER BY COALESCE(e.t0_utc, f.acceptance_utc) LIMIT 1",
        (ticker, *forms, int(after_ts), int(until_ts))).fetchone()
    if row is None:
        return None
    return {"accession_no": row["accession_no"], "items": row["items"],
            "acceptance_utc": int(row["acceptance_utc"]),
            "t0_utc": int(row["t0_utc"] if row["t0_utc"] is not None
                          else row["acceptance_utc"])}


def backfill(cfg: dict, conn, horizon: int | None = None,
             limit: int | None = None) -> dict:
    """Score every alert whose outcome window has fully elapsed.

    Returns counts: scored, filed, missed, pending. `pending` is the number
    deferred because their window runs past the filing data horizon — they are
    left unscored on purpose and picked up next time.
    """
    forms = tuple(cfg["edgar"]["forms"])
    span = window_seconds(cfg)
    horizon = data_horizon(conn) if horizon is None else int(horizon)
    if horizon is None:
        raise SystemExit(
            "no filings stored, so no alert can be scored — run the EDGAR "
            "collector before the outcome backfill.")

    from src.live.alertlog import unscored

    checked_utc = utc_now_ts()
    scored = filed = pending = 0

    # The horizon is the newest ACCEPTANCE time held locally, but the window is
    # measured on t0, which is min(acceptance, earliest news) and so can be up
    # to `news.t0_lookback_hours` earlier. A filing not yet collected — accepted
    # just past the horizon — can therefore still carry a t0 that falls inside
    # an alert's window. Pulling the answerable edge back by that much keeps the
    # promise this module is built on: never score on data we do not have.
    lookback = int((cfg.get("news") or {}).get("t0_lookback_hours", 0)) * 3600
    answerable_until = horizon - lookback

    for alert in unscored(conn, limit=limit):
        ts = int(alert["ts_utc"])
        until = ts + span
        if until > answerable_until:
            # Not "checked and clean" — not answerable yet. Recording a miss
            # here would count missing data as a missing event.
            pending += 1
            continue

        hit = first_filing_after(conn, alert["ticker"], ts, until, forms)
        lead = None
        if hit is not None:
            # Trading hours, per rule 3 — the same unit every other lead-time
            # figure in this project uses.
            lead = trading_hours_between(ts, hit["t0_utc"],
                                         calendar=_calendar(cfg))
        conn.execute(
            "INSERT INTO alert_outcomes (alert_id, checked_utc, filed, "
            "accession_no, item_code, t0_utc, lead_trading_h) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT DO NOTHING",
            (alert["alert_id"], checked_utc, 1 if hit else 0,
             hit["accession_no"] if hit else None,
             hit["items"] if hit else None,
             hit["t0_utc"] if hit else None, lead))
        scored += 1
        filed += 1 if hit else 0

    conn.commit()
    return {"scored": scored, "filed": filed, "missed": scored - filed,
            "pending": pending, "horizon_utc": horizon}


def _calendar(cfg: dict):
    from src.utils.timeutils import get_market_calendar
    return get_market_calendar(cfg["market"]["calendar"])


def hit_rates(conn) -> dict:
    """Per detector: how many scored alerts were followed by a filing.

    This is the Phase 7 headline. `pending` is carried alongside so a reader
    can see how much of the log is still unanswerable rather than assuming the
    scored part is all of it.
    """
    rows = conn.execute(
        "SELECT a.detector, COUNT(*) n, SUM(o.filed) hits, "
        "       AVG(o.lead_trading_h) mean_lead "
        "FROM alerts a JOIN alert_outcomes o ON o.alert_id = a.alert_id "
        "GROUP BY a.detector ORDER BY a.detector").fetchall()

    pending = {r["detector"]: int(r["n"]) for r in conn.execute(
        "SELECT a.detector, COUNT(*) n FROM alerts a "
        "LEFT JOIN alert_outcomes o ON o.alert_id = a.alert_id "
        "WHERE o.alert_id IS NULL GROUP BY a.detector")}

    out = {}
    for r in rows:
        n, hits = int(r["n"]), int(r["hits"] or 0)
        out[r["detector"]] = {
            "scored": n, "filed": hits, "missed": n - hits,
            "hit_rate": hits / n if n else float("nan"),
            "mean_lead_trading_h": (float(r["mean_lead"])
                                    if r["mean_lead"] is not None else None),
            "pending": pending.get(r["detector"], 0),
        }
    for detector, n in pending.items():
        out.setdefault(detector, {"scored": 0, "filed": 0, "missed": 0,
                                  "hit_rate": float("nan"),
                                  "mean_lead_trading_h": None, "pending": n})
    return out


def item_breakdown(conn, detector: str | None = None) -> dict:
    """Which 8-K item codes the alerts actually caught."""
    sql = ("SELECT o.item_code, COUNT(*) n FROM alert_outcomes o "
           "JOIN alerts a ON a.alert_id = o.alert_id "
           "WHERE o.filed = 1 AND o.item_code IS NOT NULL")
    params: tuple = ()
    if detector:
        sql += " AND a.detector = ?"
        params = (detector,)
    sql += " GROUP BY o.item_code ORDER BY n DESC"
    return {r["item_code"]: int(r["n"]) for r in conn.execute(sql, params)}


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true",
                    help="show hit rates without scoring anything new")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])

    if not args.report:
        result = backfill(cfg, conn, limit=args.limit)
        print(f"scored {result['scored']} alert(s): {result['filed']} followed "
              f"by a filing, {result['missed']} not")
        print(f"deferred {result['pending']} whose {window_seconds(cfg)//3600}h "
              f"window runs past the filing data horizon "
              f"({ts_to_iso(result['horizon_utc'])})")

    rates = hit_rates(conn)
    if not rates:
        print("\nnothing scored yet.")
        return
    print(f"\n{'detector':20s} {'scored':>7s} {'filed':>6s} {'hit rate':>9s} "
          f"{'mean lead':>10s} {'pending':>8s}")
    for name, r in rates.items():
        lead = ("—" if r["mean_lead_trading_h"] is None
                else f"{r['mean_lead_trading_h']:.1f}h")
        print(f"{name:20s} {r['scored']:7,d} {r['filed']:6,d} "
              f"{r['hit_rate']:9.1%} {lead:>10s} {r['pending']:8,d}")

    items = item_breakdown(conn)
    if items:
        print("\nitem codes caught:")
        for code, n in items.items():
            print(f"  {code:10s} {n}")


if __name__ == "__main__":
    main()
