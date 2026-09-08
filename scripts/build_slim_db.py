#!/usr/bin/env python3
"""Build the slim bootstrap database the scheduled monitor starts from.

The working database is ~1 GB, almost all of it history the live monitor never
reads. The monitor needs three things: the universe, enough recent bars for the
features to be defined, and enough recent filings to answer `days_since_last_8k`
and to score outcomes.

`features.volume_zscore_window_h` is 480 bars and `min_baseline_bars` is 120, so
roughly 600 bars of history per ticker are required before `volume_z` exists at
all. `--days` is generous against that: at ~7 bars a session, 120 calendar days
is about 600 bars.

The alert log is deliberately NOT copied. It lives in `live-log/alerts.csv`,
committed to the repository, and is restored with `alertlog.import_csv`. The
database is a cache that can be rebuilt; the log is the record that cannot.

Usage:
  .venv/bin/python scripts/build_slim_db.py --out data/live/bootstrap.db
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db                                    # noqa: E402
from src.utils.config import load_config              # noqa: E402
from src.utils.timeutils import ts_to_iso             # noqa: E402


def _columns(conn, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def _copy_sql(out, table: str, verb: str = "INSERT") -> str:
    """`INSERT INTO t (cols) SELECT cols FROM full.t` — columns NAMED.

    `SELECT *` across two databases matches columns by POSITION, and the two
    schemas here are not built the same way: the destination is created fresh
    from SCHEMA, while the source has had columns appended by `ALTER TABLE` as
    migrations ran. Those orders agree today — checked — but they agree by
    accident of migration history, and if one ever diverged the copy would
    succeed silently with values in the wrong columns. That is the worst
    possible failure for a file whose whole job is to seed the scheduled
    monitor: no error, a plausible-looking database, and every alert wrong.

    A named copy cannot do that. Columns are taken from the DESTINATION, so a
    column the source is missing raises "no such column" — a problem you can
    see — and a source column the destination schema dropped is simply not
    copied, which is the intended direction.
    """
    names = ", ".join(f'"{c}"' for c in _columns(out, table))
    return f'{verb} INTO "{table}" ({names}) SELECT {names} FROM full."{table}"'


def build(cfg: dict, out_path: str, days: int = 120,
          filing_days: int = 400) -> dict:
    src = db.get_conn(cfg["paths"]["db"], readonly=True)
    interval = cfg["market"]["interval"]
    benchmark = cfg["market"]["benchmark"]

    newest = src.execute("SELECT MAX(ts_utc) FROM bars WHERE interval = ?",
                         (interval,)).fetchone()[0]
    if newest is None:
        raise SystemExit("the source database holds no bars.")
    cutoff = int(newest) - days * 24 * 3600
    # Filings reach back further: days_since_last_8k needs the most recent
    # filing before a bar, which can be months old.
    filing_cutoff = cutoff - filing_days * 24 * 3600

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    out = db.get_conn(out_path)                       # creates the full schema

    out.execute("ATTACH DATABASE ? AS full", (cfg["paths"]["db"],))

    # Every column named explicitly — see `_copy_sql`. A source column the
    # destination schema does not have is dropped deliberately; a destination
    # column the source lacks raises, which is what should happen.
    out.execute(_copy_sql(out, "companies"))
    out.execute(
        _copy_sql(out, "bars") + " WHERE interval = ? AND ts_utc >= ? "
        "AND (ticker IN (SELECT ticker FROM full.companies "
        "WHERE in_universe = 1) OR ticker = ?)",
        (interval, cutoff, benchmark))
    out.execute(
        _copy_sql(out, "filings") + " WHERE acceptance_utc >= ?",
        (filing_cutoff,))
    out.execute(
        _copy_sql(out, "events") + " WHERE t0_utc >= ?", (filing_cutoff,))
    # meta carries the snapshot freeze stamps; without them the collector
    # refuses to run at all. OR REPLACE because get_conn seeds meta when it
    # creates the schema, so a plain INSERT collides on schema_version.
    out.execute(_copy_sql(out, "meta", verb="INSERT OR REPLACE"))
    out.commit()
    out.execute("DETACH DATABASE full")
    out.execute("VACUUM")
    out.commit()

    stats = {t: out.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
             for t in ("companies", "bars", "filings", "events", "meta")}
    stats["cutoff_utc"] = cutoff
    stats["bytes"] = os.path.getsize(out_path)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/live/bootstrap.db")
    ap.add_argument("--days", type=int, default=120,
                    help="bar history to keep; 120 days is ~600 bars, the "
                         "minimum for volume_z to be defined")
    ap.add_argument("--filing-days", type=int, default=400,
                    help="how much further back than --days to keep filings; "
                         "days_since_last_8k needs the most recent filing "
                         "before a bar, which can be many months old")
    args = ap.parse_args()

    stats = build(load_config(), args.out, args.days, args.filing_days)
    print(f"wrote {args.out}  ({stats['bytes'] / 1e6:.0f} MB)")
    print(f"  bars from {ts_to_iso(stats['cutoff_utc'])}")
    for t in ("companies", "bars", "filings", "events", "meta"):
        print(f"  {t:10s} {stats[t]:>9,}")
    print("\nNext: gzip it and attach it to a GitHub release, then set the "
          "BOOTSTRAP_DB_URL secret. See .github/workflows/live-monitor.yml.")


if __name__ == "__main__":
    main()
