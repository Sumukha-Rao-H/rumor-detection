"""One command, the whole cycle: fetch, scan, log, score, export.

This is what the scheduled job runs. It exists as a single entry point rather
than four so an unattended run has one exit code and one log to read — a
four-step shell pipeline fails in four ways, and three of them are silent.

**It does not need to run continuously.** yfinance serves roughly two years of
hourly history, so a run that happens once a day, or after a gap of a week,
fetches every hour it missed. The alert log records the bar (`ts_utc`) and the
moment the monitor noticed (`raised_utc`) in separate columns, so a catch-up
run is self-describing: nothing is disguised as real-time.

What the evidence is, stated precisely
--------------------------------------
This is **not** live alerting, and the report should not say it is. What it is,
and what actually matters for defending a result: a **pre-registered
out-of-sample test**. The detectors, their tuned thresholds and this code were
committed to git before these bars existed, so there is no route by which they
could have been tuned to them. Git's timestamps make that ordering checkable by
someone who does not trust the author, and the alert log's hash chain makes the
record internally consistent. Neither depends on the monitor having run at any
particular instant.

Usage:
  python -m src.live.catchup                    # the full cycle
  python -m src.live.catchup --max-tickers 300  # stay inside a CI budget
  python -m src.live.catchup --no-fetch         # score what is already stored
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.utils.config import load_config
from src.utils.timeutils import ts_to_iso, utc_now_ts

#: Where the durable log lives. Committed to the repository, because the
#: database is a rebuildable cache and this is not.
#:
#: Deliberately NOT under `data/`. The code standards say never commit anything
#: there, and they are right — `data/` holds a 1 GB database, raw API caches
#: and parquet files, all regenerable bulk. This file is the opposite: small,
#: append-only, and the one artefact of Phase 7 that cannot be rebuilt from any
#: source. Putting it at the top level keeps the standard intact rather than
#: carving an exception into it, and `.gitignore`'s `data/**` would otherwise
#: have silently dropped every commit the scheduled job made.
DEFAULT_LOG_CSV = "live-log/alerts.csv"


def run(cfg: dict, conn, max_tickers: int | None = None,
        fetch: bool = True, log_csv: str | None = DEFAULT_LOG_CSV,
        as_of: int | None = None) -> dict:
    """Fetch, scan, log, backfill outcomes, export. Returns a summary dict.

    Ordering matters. Outcomes are backfilled **after** the new alerts are
    logged, so an alert raised today and a filing that lands within the same
    run are both accounted for. And the export happens last, so the committed
    CSV reflects everything this run learned.
    """
    from src.live.alertlog import append, export_csv, summary, verify_chain
    from src.live.monitor import (build_detectors, conform, fetch_latest,
                                  latest_bar_frame)
    from src.live.outcomes import backfill

    started = time.time()
    result: dict = {"started_utc": utc_now_ts()}

    tickers = None
    if max_tickers:
        tickers = [r[0] for r in conn.execute(
            "SELECT ticker FROM companies WHERE in_universe = 1 "
            "ORDER BY ticker LIMIT ?", (max_tickers,))]

    if fetch:
        result["bars_appended"] = fetch_latest(cfg, conn, tickers=tickers)
    else:
        result["bars_appended"] = 0

    frame = conform(latest_bar_frame(cfg, conn, tickers, as_of=as_of))
    result["bars_scored"] = len(frame)
    result["tickers"] = int(frame["ticker"].nunique())
    result["newest_bar_utc"] = int(frame["ts_utc"].max())

    alerts = scan_alerts(cfg, conn, frame)
    result["alerts_found"] = len(alerts)
    result["alerts_new"] = append(conn, alerts)

    # After logging, so a filing that lands in the same run is picked up.
    result["outcomes"] = backfill(cfg, conn)

    # The chain is verified on every run rather than on demand: a break found
    # weeks later is a break nobody can date.
    result["chain"] = {k: v["ok"] for k, v in verify_chain(conn).items()}
    result["log"] = summary(conn)

    if log_csv:
        result["exported_rows"] = export_csv(conn, log_csv)
        result["log_csv"] = str(log_csv)

    result["elapsed_s"] = round(time.time() - started, 1)
    return result


def scan_alerts(cfg: dict, conn, frame):
    from src.live.monitor import build_detectors, scan
    return scan(cfg, conn, frame, build_detectors(cfg))


def render(result: dict) -> str:
    """A summary an unattended run can be read from, days later."""
    lines = [
        f"bars appended : {result['bars_appended']:,}",
        f"bars scored   : {result['bars_scored']:,} "
        f"across {result['tickers']:,} tickers",
        f"newest bar    : {ts_to_iso(result['newest_bar_utc'])}",
        f"alerts        : {result['alerts_found']} found, "
        f"{result['alerts_new']} new",
    ]
    o = result["outcomes"]
    lines.append(f"outcomes      : scored {o['scored']}, filed {o['filed']}, "
                 f"deferred {o['pending']} "
                 f"(filing horizon {ts_to_iso(o['horizon_utc'])})")
    broken = [k for k, ok in result["chain"].items() if not ok]
    lines.append(f"chain         : {'ok' if not broken else 'BROKEN: ' + ', '.join(broken)}")
    for name, s in result["log"].items():
        lines.append(f"  {name:20s} {s['alerts']:6,d} alerts total")
    if "exported_rows" in result:
        lines.append(f"exported      : {result['exported_rows']:,} rows -> "
                     f"{result['log_csv']}")
    lines.append(f"elapsed       : {result['elapsed_s']}s")
    return "\n".join(lines)


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-tickers", type=int, default=None,
                    help="cap the universe, to stay inside a CI minute budget")
    ap.add_argument("--no-fetch", action="store_true",
                    help="score what is already stored; download nothing")
    ap.add_argument("--log-csv", default=DEFAULT_LOG_CSV,
                    help="where to export the durable log")
    ap.add_argument("--as-of", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    result = run(cfg, conn, max_tickers=args.max_tickers,
                 fetch=not args.no_fetch, log_csv=args.log_csv,
                 as_of=args.as_of)
    print(render(result))

    # A broken chain is the one condition that must fail the job rather than
    # be noticed in a log nobody reads.
    if any(not ok for ok in result["chain"].values()):
        raise SystemExit("alert log chain is BROKEN — see the report above")


if __name__ == "__main__":
    main()
