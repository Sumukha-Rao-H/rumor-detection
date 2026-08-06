"""Human label review — plan §6.4 stage 3. The gate before any label is real.

The plan is unambiguous that a person decides every label, because noise here
propagates into the reward and poisons everything downstream. So machine
verdicts live in `label_proposals` and *nothing* reads them as truth: this
module exports them to a CSV a human edits, then writes the human's answers —
and only those — to `events.label`.

A CSV rather than a web app on purpose. Review is a couple of person-days of
squinting at claims and headlines; a spreadsheet is offline, sortable, works on
a phone, and survives being emailed to a teammate. Nothing in the loop needs a
server running.

The sheet is ordered by what is worth a human's attention, not by event id:

  moved_filed  the subset of `moved` whose ticker filed with the SEC inside the
             label horizon. Measured 2026-08-06: every TRUE label in the set so
             far came from just two domains, sec.gov (23) and cnbc.com (19),
             because the free news tiers return aggregators — GDELT yields 0.5%
             whitelisted sources and Finnhub 4.5%. A filing in the window is
             therefore the strongest remaining signal that a claim was real,
             and 184 of the 639 moved events have one. Review these first.
  moved      the rest of the largest bucket. The stock moved on the claim but no
             free source published anything we could cite — many are real
             confirmations nobody indexed.
  secondary  an aggregator supported the claim but cannot settle it alone.
             A person can tell "Reuters confirms" from "Benzinga repeats the
             rumor" in seconds; the labeler deliberately cannot.
  pre_t0     proposed TRUE, but the news predates the Reddit post. Usually a
             repost of known news rather than a rumor anyone was early on, and
             keeping them would teach Phase 3 to "predict" what had already
             broken.
  confirmed  machine-proposed TRUE/FALSE, shown for a confirming glance.

Round-trip contract: the reviewer edits `label` (TRUE/FALSE/SKIP) and may edit
`t_official`, and touches nothing else. Import matches on event_id, so rows can
be reordered, filtered or split across files freely.

Usage:
  python -m src.pipeline.review export --bucket moved --limit 200
  python -m src.pipeline.review import data/review/moved.csv
  python -m src.pipeline.review status
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from src import db
from src.pipeline.labeling import PRIMARY, event_headlines, rank_headlines
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, ts_to_iso, utc_now_ts

log = logging.getLogger(__name__)

# What the reviewer may write in the `label` column.
ACCEPTED = {"TRUE": 1, "FALSE": 0, "SKIP": None, "": None}

_FILED = ("EXISTS (SELECT 1 FROM news n WHERE n.ticker = e.ticker AND n.api = 'edgar'"
          " AND n.seen_utc BETWEEN e.t0_utc AND e.t0_utc + {horizon})")

BUCKETS = {
    "moved_filed": (f"rule = 'moved' AND {_FILED}",),
    "moved": (f"rule = 'moved' AND NOT {_FILED}",),
    "secondary": ("rule IN ('confirmed_secondary', 'denied_secondary')",),
    "pre_t0": ("rule = 'confirmed_pre_t0'",),
    "confirmed": ("rule IN ('confirmed', 'denied')",),
    "quiet": ("rule = 'quiet'",),
    "no_bars": ("rule = 'no_bars'",),
    "all": ("1 = 1",),
}

COLUMNS = [
    # The reviewer fills these two.
    "label", "t_official",
    # Everything below is context, and import ignores it.
    "event_id", "ticker", "t0", "claim_type", "claim_summary",
    "proposed", "rule", "confidence", "ret_3d", "volume_z",
    "deciding_headline", "evidence", "notes",
]


def bucket_where(bucket: str, cfg: dict) -> str:
    """The bucket's SQL predicate, with the label horizon filled in."""
    if bucket not in BUCKETS:
        raise SystemExit(f"unknown bucket {bucket!r}; pick from {sorted(BUCKETS)}")
    return BUCKETS[bucket][0].format(
        horizon=int(cfg["event"]["label_horizon_hours"]) * 3600)


def sheet_rows(conn, cfg: dict, bucket: str, limit: int | None = None,
               include_reviewed: bool = False) -> list[dict]:
    """Proposals in one bucket, rendered for review."""
    where = bucket_where(bucket, cfg)
    reviewed = "" if include_reviewed else " AND e.human_reviewed = 0"
    rows = conn.execute(
        f"""SELECT p.*, e.claim_summary, e.claim_type, e.ticker, e.t0_utc
            FROM label_proposals p JOIN events e ON e.event_id = p.event_id
            WHERE p.prompt_version = ? AND {where}{reviewed}
            ORDER BY p.confidence DESC, e.t0_utc""",
        (cfg["labeling"]["prompt_version"],),
    ).fetchall()

    out = []
    for row in rows[:limit] if limit else rows:
        event = _as_event(row)
        headlines, _ = event_headlines(conn, cfg, event)
        top = rank_headlines(event.claim_summary or "", headlines,
                             int(cfg["review"]["evidence_lines"]))
        out.append({
            "label": "", "t_official": "",
            "event_id": row["event_id"], "ticker": row["ticker"],
            "t0": ts_to_iso(row["t0_utc"]), "claim_type": row["claim_type"],
            "claim_summary": row["claim_summary"],
            "proposed": row["verdict"], "rule": row["rule"],
            "confidence": row["confidence"],
            "ret_3d": _round(row["ret_3d"]), "volume_z": _round(row["volume_z"]),
            "deciding_headline": row["deciding_headline"] or "",
            "evidence": " | ".join(
                f"{'*' if h.tier == PRIMARY else ''}{ts_to_iso(h.seen_utc)[:10]} "
                f"{h.domain}: {h.title[:90]}" for h in top),
            "notes": "",
        })
    return out


def _as_event(row):
    from src.pipeline.labeling import Event
    return Event(row["event_id"], row["ticker"], row["t0_utc"],
                 row["claim_summary"] or "")


def _round(value, places: int = 4):
    return "" if value is None else round(value, places)


def export_sheet(conn, cfg: dict, path: Path, bucket: str,
                 limit: int | None = None) -> int:
    rows = sheet_rows(conn, cfg, bucket, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_decision(row: dict) -> tuple[str, int | None, int | None]:
    """(event_id, label, t_official_utc) from one reviewed row.

    Raises on anything unrecognised rather than guessing: a typo in the label
    column silently becoming FALSE is exactly the noise this stage exists to
    keep out.
    """
    event_id = (row.get("event_id") or "").strip()
    if not event_id:
        raise ValueError("row has no event_id")
    raw = (row.get("label") or "").strip().upper()
    if raw not in ACCEPTED:
        raise ValueError(f"{event_id}: unrecognised label {raw!r}; "
                         f"use one of {sorted(k for k in ACCEPTED if k)}")
    label = ACCEPTED[raw]

    stamp = (row.get("t_official") or "").strip()
    t_official = None
    if stamp:
        try:
            t_official = (int(stamp) if stamp.isdigit()
                          else date_str_to_ts(stamp[:10]))
        except ValueError as exc:
            raise ValueError(f"{event_id}: bad t_official {stamp!r}") from exc
    return event_id, label, t_official


def import_sheet(conn, cfg: dict, path: Path) -> dict:
    """Apply a reviewed sheet. Only rows carrying a decision are written."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    decisions, skipped = [], 0
    for row in rows:
        event_id, label, t_official = parse_decision(row)
        if label is None:
            skipped += 1
            continue
        decisions.append((label, t_official, event_id))

    known = {r[0] for r in conn.execute("SELECT event_id FROM events")}
    unknown = [d[2] for d in decisions if d[2] not in known]
    if unknown:
        raise SystemExit(f"{len(unknown)} unknown event_ids, e.g. {unknown[:3]}")

    # A reviewed label falls back to the machine's t_official when the reviewer
    # did not supply one — they are correcting the verdict, not the clock.
    with conn:
        conn.executemany(
            """UPDATE events SET label = ?, human_reviewed = 1,
                 label_source = 'human',
                 t_official_utc = COALESCE(?, (
                   SELECT t_official_utc FROM label_proposals p
                   WHERE p.event_id = events.event_id))
               WHERE event_id = ?""",
            decisions,
        )
    return {"rows": len(rows), "applied": len(decisions), "left_blank": skipped}


def status(conn, cfg: dict) -> dict:
    """Where review stands, per bucket."""
    version = cfg["labeling"]["prompt_version"]
    out = {}
    for name in BUCKETS:
        where = bucket_where(name, cfg)
        row = conn.execute(
            f"""SELECT COUNT(*) total, SUM(e.human_reviewed) done
                FROM label_proposals p JOIN events e ON e.event_id = p.event_id
                WHERE p.prompt_version = ? AND {where}""", (version,)).fetchone()
        out[name] = (row["done"] or 0, row["total"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="write a review sheet")
    exp.add_argument("--bucket", default="moved", choices=sorted(BUCKETS))
    exp.add_argument("--limit", type=int)
    exp.add_argument("--out", help="CSV path (default: review dir / bucket.csv)")

    imp = sub.add_parser("import", help="apply a reviewed sheet")
    imp.add_argument("path")

    sub.add_parser("status", help="review progress per bucket")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])

    if args.command == "export":
        path = Path(args.out) if args.out else (
            Path(cfg["paths"]["review"]) / f"{args.bucket}.csv")
        n = export_sheet(conn, cfg, path, args.bucket, args.limit)
        log.info("wrote %d rows to %s", n, path)
        log.info("edit the `label` column (TRUE/FALSE/SKIP), then: "
                 "python -m src.pipeline.review import %s", path)
    elif args.command == "import":
        stats = import_sheet(conn, cfg, Path(args.path))
        log.info("applied %(applied)d decisions from %(rows)d rows "
                 "(%(left_blank)d left blank)", stats)
    else:
        for name, (done, total) in status(conn, cfg).items():
            log.info("%-10s %4d/%-4d reviewed", name, done, total)
    log.info("labeled events: %d",
             conn.execute("SELECT COUNT(*) FROM events WHERE label IS NOT NULL"
                          ).fetchone()[0])


if __name__ == "__main__":
    main()
