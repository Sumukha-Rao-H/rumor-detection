"""The alert log — append-only, and checkably so.

Everywhere else in this project a re-run upserts and the last write wins. That
is what makes the collectors idempotent and it is the right default. **This
table is the exception**, and the reason is that it is evidence rather than
data: an alert that could be rewritten after the outcome was known would prove
nothing about what the detector actually said at the time.

So three rules, each enforced rather than intended:

1. **The first write wins.** `INSERT ... ON CONFLICT DO NOTHING` on the natural
   key (detector, ticker, bar). Re-running the monitor over the same hours
   changes nothing, and a detector whose threshold was retuned cannot quietly
   restate its own history.
2. **Outcomes go somewhere else.** P7-03 records whether a filing followed, in
   `alert_outcomes`. Storing it here would mean updating rows in the log, which
   is the one thing the log must not do.
3. **Edits are detectable.** Each row carries the sha of the previous row for
   its detector, so the log forms a chain per detector. `verify_chain` walks it
   and reports the first break.

The chain **detects** tampering; it does not prevent it. Anyone with the SQLite
file can rewrite it and recompute every hash. What it rules out is the
realistic failure — a well-meaning later edit, a partial restore, a row quietly
dropped — and it turns "never edited after the fact" from a promise into
something a reader can check with one command.

Usage:
  python -m src.live.alertlog --verify
  python -m src.live.alertlog --tail 20
"""

from __future__ import annotations

import argparse
import hashlib
import json

from src.utils.config import load_config
from src.utils.timeutils import ts_to_iso, utc_now_ts

#: The columns that make a row's identity. Anything outside this can be
#: recomputed; anything inside it defines what the detector said.
_CHAINED = ("alert_id", "ts_utc", "raised_utc", "ticker", "detector", "score",
            "threshold", "features")


def alert_id(detector: str, ticker: str, ts_utc: int) -> str:
    """A deterministic id for one detector's call on one bar.

    Derived rather than autoincremented so re-running the monitor produces the
    same id, which is what lets the insert be a no-op instead of a duplicate.
    """
    raw = f"{detector}|{ticker}|{int(ts_utc)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def row_sha(row: dict, prev_sha: str | None) -> str:
    """Hash of this row's content plus the previous row's hash.

    `sort_keys` on the feature JSON matters: a dict that serialised in a
    different order would hash differently and break the chain for no reason.
    """
    payload = {k: row[k] for k in _CHAINED}
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{prev_sha or ''}|{body}".encode()).hexdigest()


def _chain_head(conn, detector: str) -> tuple[int, str | None]:
    """(next seq, last row_sha) for a detector."""
    row = conn.execute(
        "SELECT seq, row_sha FROM alerts WHERE detector = ? "
        "ORDER BY seq DESC LIMIT 1", (detector,)).fetchone()
    return (0, None) if row is None else (int(row[0]) + 1, str(row[1]))


def append(conn, alerts, raised_utc: int | None = None) -> int:
    """Append alerts to the log. Returns how many were newly written.

    Already-logged alerts are skipped silently — that is the point of the
    natural key, not a failure. The count returned is *new* rows, so a caller
    can tell "nothing fired" from "everything fired again".

    Alerts are sorted before writing so the chain order is deterministic
    regardless of the order `scan` happened to return them in.
    """
    raised_utc = int(raised_utc if raised_utc is not None else utc_now_ts())
    written = 0

    for alert in sorted(alerts, key=lambda a: (a.detector, a.ts_utc, a.ticker)):
        aid = alert_id(alert.detector, alert.ticker, alert.ts_utc)
        if conn.execute("SELECT 1 FROM alerts WHERE alert_id = ?",
                        (aid,)).fetchone():
            continue                       # first write wins; never restated

        seq, prev = _chain_head(conn, alert.detector)
        row = {
            "alert_id": aid,
            "ts_utc": int(alert.ts_utc),
            "raised_utc": raised_utc,
            "ticker": str(alert.ticker),
            "detector": str(alert.detector),
            "score": float(alert.score),
            "threshold": float(alert.threshold),
            "features": json.dumps(alert.features, sort_keys=True),
        }
        conn.execute(
            "INSERT INTO alerts (alert_id, ts_utc, raised_utc, ticker, "
            "detector, score, threshold, features, seq, prev_sha, row_sha) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING",
            (row["alert_id"], row["ts_utc"], row["raised_utc"], row["ticker"],
             row["detector"], row["score"], row["threshold"], row["features"],
             seq, prev, row_sha(row, prev)))
        written += 1

    conn.commit()
    return written


def verify_chain(conn, detector: str | None = None) -> dict:
    """Walk the log and report the first break, per detector.

    Returns `{detector: {"rows": n, "ok": bool, "broken_at": seq|None,
    "reason": str|None}}`.

    Three things break a chain, and all three are what "never edited" is meant
    to exclude: a row whose contents were changed, a row removed from the
    middle, and a row inserted out of order.
    """
    names = ([detector] if detector else
             [r[0] for r in conn.execute(
                 "SELECT DISTINCT detector FROM alerts ORDER BY detector")])

    report = {}
    for name in names:
        rows = conn.execute(
            "SELECT alert_id, ts_utc, raised_utc, ticker, detector, score, "
            "threshold, features, seq, prev_sha, row_sha FROM alerts "
            "WHERE detector = ? ORDER BY seq", (name,)).fetchall()

        prev = None
        result = {"rows": len(rows), "ok": True, "broken_at": None,
                  "reason": None}
        for expected_seq, r in enumerate(rows):
            row = {k: r[k] for k in _CHAINED}
            if int(r["seq"]) != expected_seq:
                result.update(ok=False, broken_at=int(r["seq"]),
                              reason=f"seq gap: expected {expected_seq}, "
                                     f"found {r['seq']} — a row was removed "
                                     f"or inserted out of order")
                break
            if (r["prev_sha"] or None) != prev:
                result.update(ok=False, broken_at=int(r["seq"]),
                              reason="prev_sha does not match the previous "
                                     "row — the chain was re-linked")
                break
            expected = row_sha(row, prev)
            if expected != r["row_sha"]:
                result.update(ok=False, broken_at=int(r["seq"]),
                              reason="row_sha does not match the row's "
                                     "contents — this row was edited")
                break
            prev = r["row_sha"]
        report[name] = result
    return report


def unscored(conn, limit: int | None = None) -> list[dict]:
    """Alerts with no outcome recorded yet. P7-03's work queue."""
    sql = ("SELECT a.* FROM alerts a LEFT JOIN alert_outcomes o "
           "ON o.alert_id = a.alert_id WHERE o.alert_id IS NULL "
           "ORDER BY a.ts_utc")
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [dict(r) for r in conn.execute(sql)]


def summary(conn) -> dict:
    """Counts per detector, and the span the log covers."""
    rows = conn.execute(
        "SELECT detector, COUNT(*) n, MIN(ts_utc) lo, MAX(ts_utc) hi "
        "FROM alerts GROUP BY detector ORDER BY detector").fetchall()
    return {r["detector"]: {"alerts": int(r["n"]), "first_utc": int(r["lo"]),
                            "last_utc": int(r["hi"])} for r in rows}


#: The CSV column order for the exported log. Fixed so a diff between two
#: exports is a diff in the data, not in the serialisation.
CSV_COLUMNS = ("alert_id", "ts_utc", "raised_utc", "ticker", "detector",
               "score", "threshold", "features", "seq", "prev_sha", "row_sha")


def export_csv(conn, path) -> int:
    """Write the whole log to CSV. Returns rows written.

    The database is a cache — it can be rebuilt from yfinance, whose hourly
    history persists about two years. **This file is the durable record**, and
    it is committed to the repository, which gives the log a second and
    independent history: git's own timestamps and hashes, kept by a service
    nobody in this project controls.

    That matters for the same reason the row hashes do. The chain proves
    internal consistency; git proves *when* each row appeared. Together they
    make "this alert was recorded before the outcome was known" checkable by
    someone who does not trust the author.
    """
    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(
        f"SELECT {', '.join(CSV_COLUMNS)} FROM alerts "
        f"ORDER BY detector, seq").fetchall()
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for r in rows:
            writer.writerow([r[c] for c in CSV_COLUMNS])
    return len(rows)


def import_csv(conn, path) -> int:
    """Restore a log from CSV. Returns rows inserted (existing ones skipped).

    The recovery path for a lost or rebuilt database. Rows are inserted exactly
    as exported — hashes included, never recomputed — so `verify_chain` after
    an import checks the *restored* data against the hashes it was written
    with. Recomputing them would make any corruption verify perfectly, which is
    the one thing the chain exists to prevent.
    """
    import csv
    from pathlib import Path

    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    inserted = 0
    for r in rows:
        cur = conn.execute(
            "INSERT INTO alerts (alert_id, ts_utc, raised_utc, ticker, "
            "detector, score, threshold, features, seq, prev_sha, row_sha) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING",
            (r["alert_id"], int(r["ts_utc"]), int(r["raised_utc"]), r["ticker"],
             r["detector"], float(r["score"]), float(r["threshold"]),
             r["features"], int(r["seq"]), r["prev_sha"] or None, r["row_sha"]))
        inserted += cur.rowcount
    conn.commit()
    return inserted


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="walk the hash chain and report any break")
    ap.add_argument("--tail", type=int, default=None,
                    help="show the most recent N alerts")
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])

    stats = summary(conn)
    if not stats:
        print("the alert log is empty.")
        return
    print("alerts logged:")
    for name, s in stats.items():
        print(f"  {name:20s} {s['alerts']:6,d}   "
              f"{ts_to_iso(s['first_utc'])} .. {ts_to_iso(s['last_utc'])}")

    if args.verify:
        print("\nchain verification:")
        for name, r in verify_chain(conn).items():
            mark = "ok" if r["ok"] else f"BROKEN at seq {r['broken_at']}"
            print(f"  {name:20s} {r['rows']:6,d} rows  {mark}")
            if not r["ok"]:
                print(f"      {r['reason']}")

    if args.tail:
        print(f"\nlast {args.tail}:")
        for r in conn.execute(
                "SELECT * FROM alerts ORDER BY ts_utc DESC, detector LIMIT ?",
                (args.tail,)):
            print(f"  {ts_to_iso(r['ts_utc'])}  {r['ticker']:6s} "
                  f"{r['detector']:18s} score {r['score']:8.4f}")


if __name__ == "__main__":
    main()
