"""events.parquet export and the temporal split — plan §6.4 acceptance.

One row per human-reviewed labeled event, plus the split assignment Phase 3
trains on. Two rules govern the split and both exist to stop the model learning
something it could not know at decision time:

  temporal    train is the oldest 70%, val the next 15%, test the newest 15%,
              cut by t0. A random split leaks through market regimes (the same
              rate-cut week appears in train and test) and through repeated
              tickers (one company's January rumor teaching its March one).
  quarantine  no ticker may appear on both sides of a boundary within
              `split.quarantine_days`. Adjacent events on one ticker share
              posts, price history and often the same underlying story, so a
              boundary cutting through them hands the test set its answer.
              Events inside the quarantine are dropped, not reassigned —
              moving them would just relocate the leak.

Only `human_reviewed = 1` rows are exported. Machine proposals are not labels
(see review.py); an export that silently included them would be the one way
unreviewed verdicts could reach training.

Usage:
  python -m src.pipeline.dataset            # write events.parquet
  python -m src.pipeline.dataset --stats    # report without writing
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src import db
from src.utils.config import load_config
from src.utils.timeutils import ts_to_iso

log = logging.getLogger(__name__)

DAY = 86400
SPLITS = ("train", "val", "test")


PROVISIONAL = "machine-provisional"

# Machine verdicts standing in for labels. Everything downstream keys off
# label_source, so a run built this way is identifiable after the fact.
_PROVISIONAL_SQL = """
    SELECT e.event_id, e.ticker, e.t0_utc, e.claim_summary, e.claim_type,
           CASE p.verdict WHEN 'TRUE' THEN 1 ELSE 0 END AS label,
           COALESCE(p.t_official_utc, e.t0_utc + ? ) AS t_official_utc,
           ? AS label_source,
           e.n_posts, e.n_rumor_posts, e.subreddits, e.post_ids
    FROM label_proposals p JOIN events e ON e.event_id = p.event_id
    WHERE p.verdict IN ('TRUE', 'FALSE') AND p.prompt_version = ?
    ORDER BY e.t0_utc, e.event_id"""


def labeled_events(conn, cfg: dict | None = None,
                   provisional: bool = False) -> pd.DataFrame:
    """Human-reviewed labeled events, oldest first.

    `provisional` swaps in the machine's verdicts instead. They are not labels
    and must never reach a reported result — review.py exists precisely so
    unreviewed verdicts cannot — but Phase 3 onwards needs *some* dataset to be
    exercised end to end before review finishes. Callers have to ask for it,
    the rows are stamped `machine-provisional`, and the default is unchanged.
    """
    if provisional:
        if cfg is None:
            raise ValueError("provisional export needs the config")
        horizon = int(cfg["event"]["label_horizon_hours"]) * 3600
        rows = conn.execute(_PROVISIONAL_SQL,
                            (horizon, PROVISIONAL,
                             cfg["labeling"]["prompt_version"])).fetchall()
    else:
        rows = conn.execute(
            """SELECT event_id, ticker, t0_utc, claim_summary, claim_type, label,
                      t_official_utc, label_source, n_posts, n_rumor_posts,
                      subreddits, post_ids
               FROM events
               WHERE label IS NOT NULL AND human_reviewed = 1
               ORDER BY t0_utc, event_id"""
        ).fetchall()
    frame = pd.DataFrame([dict(r) for r in rows])
    if not frame.empty:
        frame["subreddits"] = frame["subreddits"].map(_loads)
        frame["post_ids"] = frame["post_ids"].map(_loads)
        frame["t0_iso"] = frame["t0_utc"].map(ts_to_iso)
    return frame


def _loads(value):
    try:
        return json.loads(value) if value else []
    except (TypeError, ValueError):
        return []


def assign_splits(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add a `split` column by time, then quarantine the boundaries.

    Boundaries are placed by *position* in the time-ordered list rather than by
    date, so each split gets its intended share even when events cluster.
    """
    if frame.empty:
        return frame.assign(split=pd.Series(dtype=str))
    scfg = cfg["split"]
    if scfg["method"] != "temporal":
        raise SystemExit(f"only temporal splits are supported, not {scfg['method']!r}")

    frame = frame.sort_values(["t0_utc", "event_id"]).reset_index(drop=True)
    n = len(frame)
    train_end = int(n * float(scfg["train"]))
    val_end = train_end + int(n * float(scfg["val"]))
    split = pd.Series(["test"] * n, index=frame.index)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"
    frame["split"] = split

    quarantine = int(scfg["quarantine_days"]) * DAY
    drop = set()
    for boundary in (train_end, val_end):
        if not 0 < boundary < n:
            continue
        cut = frame.loc[boundary, "t0_utc"]
        near = frame[(frame["t0_utc"] >= cut - quarantine)
                     & (frame["t0_utc"] <= cut + quarantine)]
        # Only tickers that actually straddle the cut are a leak; a ticker
        # sitting entirely on one side of it is fine.
        for ticker, group in near.groupby("ticker"):
            if group["split"].nunique() > 1:
                drop.update(group.index)
    if drop:
        log.info("quarantined %d events straddling a split boundary", len(drop))
    frame = frame.drop(index=sorted(drop)).reset_index(drop=True)
    return frame


def describe(frame: pd.DataFrame) -> dict:
    """Row counts and class balance, per split — the numbers the paper reports."""
    out = {"rows": len(frame)}
    if frame.empty:
        return out
    out["positives"] = int(frame["label"].sum())
    out["minority_share"] = round(
        min(frame["label"].mean(), 1 - frame["label"].mean()), 4)
    for name in SPLITS:
        part = frame[frame["split"] == name]
        out[name] = {
            "rows": len(part),
            "positives": int(part["label"].sum()) if len(part) else 0,
            "t0_range": [ts_to_iso(part["t0_utc"].min()),
                         ts_to_iso(part["t0_utc"].max())] if len(part) else None,
        }
    return out


def export(conn, cfg: dict, path: Path | None = None,
           provisional: bool = False) -> tuple[pd.DataFrame, dict]:
    frame = assign_splits(labeled_events(conn, cfg, provisional), cfg)
    stats = describe(frame)
    if path is not None and not frame.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    return frame, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="parquet path (default: paths.events)")
    parser.add_argument("--stats", action="store_true",
                        help="report without writing the file")
    parser.add_argument("--from-proposals", action="store_true",
                        help="build from machine verdicts instead of reviewed "
                             "labels — for exercising Phase 3+ before review "
                             "finishes. NOT a reportable dataset.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    path = None if args.stats else Path(args.out or cfg["paths"]["events"])
    if args.from_proposals:
        log.warning("building from MACHINE verdicts — provisional, not a "
                    "reportable dataset (rows stamped %s)", PROVISIONAL)
    frame, stats = export(conn, cfg, path, provisional=args.from_proposals)

    log.info("%d labeled events, %d positive (minority share %.1f%%)",
             stats["rows"], stats.get("positives", 0),
             100 * stats.get("minority_share", 0))
    for name in SPLITS:
        part = stats.get(name) or {}
        if part.get("rows"):
            log.info("  %-5s %4d rows, %3d positive, %s -> %s", name,
                     part["rows"], part["positives"], *part["t0_range"])
    target = int(cfg["split"]["min_events"])
    if stats["rows"] < target:
        log.warning("below the plan's %d-event floor — keep reviewing "
                    "(python -m src.pipeline.review export)", target)
    if path is not None and stats["rows"]:
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
