"""The live monitor — the same detectors, on bars that have not happened yet.

The plan calls this the highest-value, lowest-effort item in the project, and
the reason is that everything else in this repository is retrospective. Every
number so far was measured on data that already existed when the code was
written. This produces the one claim a reader cannot get any other way: *of the
N alerts it raised live, M were followed by a filing within 48 hours.*

**One code path, or the claim is worthless.** The done-when for P7-01 is that
this uses `src/pipeline/features.py` directly. A second implementation of the
features — even a careful one — would drift from the one the detectors were
tuned on, and the live result would then measure the drift rather than the
market. So the live frame is built by `features.ticker_features`, the same
function `build_matrix` and `build_eval_frame` call, and a test asserts the
values agree column for column.

**Every detector runs, not just the winner.** P6-06 found a tuned CUSUM beats
the learned policy, decisively on unscheduled events. Running only CUSUM would
be reasonable; running all of them costs almost nothing extra, because they
score the same frame, and it turns the live period into a forward-looking
replication of the Phase 5/6 comparison rather than a single-detector demo.
When the alert log is scored in P7-03, each detector gets its own hit rate.

**Live data is not the test set.** Bars after the study window were never part
of any split. `split_of` returns `LIVE` for them (fixed here in P7-01 — it
previously returned TEST for everything after `val_end`, unbounded, which would
have had the seal refuse the monitor its own inputs). The sealed period itself
is untouched.

Usage:
  python -m src.live.monitor --dry-run      # one pass, fetch nothing
  python -m src.live.monitor                # fetch latest bars, then score
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.pipeline.features import _event_times, _ticker_frame, ticker_features
from src.pipeline.split import LIVE, split_of
from src.utils.config import load_config
from src.utils.timeutils import ts_to_iso, utc_now_ts

#: Columns the contract needs on a live frame. There is no t0 — that is the
#: whole point: nobody knows yet whether news is coming, which is why the label
#: columns are null and `is_positive` is unknowable until P7-03 backfills it.
LABEL_COLS = ("window_id", "ticker", "t0_utc", "is_scheduled", "item_code")

log = logging.getLogger(__name__)


@dataclass
class Alert:
    """One detector saying something is happening, now."""

    ts_utc: int
    ticker: str
    detector: str
    score: float
    threshold: float
    features: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {"ts_utc": self.ts_utc, "ticker": self.ticker,
                "detector": self.detector, "score": float(self.score),
                "threshold": float(self.threshold),
                "raised_utc": utc_now_ts(), **self.features}


def latest_bar_frame(cfg: dict, conn, tickers: list[str] | None = None,
                     as_of: int | None = None,
                     lookback_bars: int | None = None) -> pd.DataFrame:
    """Features for each ticker's most recent bars, via the training code path.

    `ticker_features` is called on the ticker's whole history and then sliced,
    exactly as `build_matrix` does. That ordering is not a detail: `volume_z`
    needs `features.min_baseline_bars` of prior bars before it is defined at
    all, so computing on a short live slice would return NaN for every row and
    look entirely reasonable while doing it.

    Returns one row per (ticker, bar) over the last `lookback_bars`, in the
    feature matrix's shape. `window_id` is `live:<ticker>:<ts>` — each live bar
    is its own decision point, matching how the evaluation population treats a
    quiet hour.
    """
    interval = cfg["market"]["interval"]
    horizon = cfg["decision"]["horizon_hours"]
    lookback = int(lookback_bars or horizon)
    as_of = int(as_of if as_of is not None else utc_now_ts())

    if tickers is None:
        tickers = [r[0] for r in conn.execute(
            "SELECT ticker FROM companies WHERE in_universe = 1 ORDER BY ticker")]
    if not tickers:
        raise SystemExit("no in-universe companies — run the liquidity filter "
                         "(P3-02) before starting the monitor.")

    benchmark = _ticker_frame(conn, cfg["market"]["benchmark"], interval)
    out = []
    for ticker in tickers:
        frame = _ticker_frame(conn, ticker, interval)
        if frame.empty:
            continue
        filings, earnings = _event_times(conn, cfg, ticker)
        feats = ticker_features(frame, benchmark, filings, earnings, cfg)
        stamps = feats.index.to_numpy()

        end = np.searchsorted(stamps, as_of, side="right")   # at or before now
        block = feats.iloc[max(end - lookback, 0):end]
        if block.empty:
            continue
        block = block.copy()
        block.insert(0, "item_code", None)
        block.insert(0, "is_scheduled", None)
        block.insert(0, "t0_utc", None)
        block.insert(0, "ticker", ticker)
        block.insert(0, "window_id",
                     [f"live:{ticker}:{s}" for s in stamps[max(end - lookback, 0):end]])
        out.append(block.reset_index())

    if not out:
        raise SystemExit(
            f"no bars at or before {ts_to_iso(as_of)} for any in-universe "
            f"ticker — has the collector run?")
    return pd.concat(out, ignore_index=True)


def conform(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast to the contract's dtypes."""
    out = frame.copy()
    for col, dtype in (("window_id", "string"), ("ticker", "string"),
                       ("ts_utc", "Int64"), ("t0_utc", "Int64"),
                       ("is_scheduled", "boolean"), ("item_code", "string")):
        out[col] = out[col].astype(dtype)
    return out


def build_detectors(cfg: dict, policy_runs: list[str] | None = None) -> dict:
    """Every detector the monitor should run.

    CUSUM and volume z-score come from config at their P5-03/P5-04 tuned
    settings. Policies are loaded from P6-03 run directories. Always-quiet is
    omitted deliberately — it never alerts, so it would contribute nothing to
    an alert log, and its floor is already measured offline.
    """
    from src.baselines import CUSUM, VolumeZScore

    detectors = {"cusum": CUSUM(cfg), "volume_zscore": VolumeZScore(cfg)}
    for run in (policy_runs or []):
        from pathlib import Path

        from src.rl import load_policy
        detectors[f"rl_policy[{Path(run).name.split('-')[-1]}]"] = \
            load_policy(cfg, run)
    return detectors


def scan(cfg: dict, conn, frame: pd.DataFrame,
         detectors: dict, thresholds: dict | None = None) -> list[Alert]:
    """Score the live frame with every detector and return what fired.

    Thresholds come from each detector's tuned operating point, so a live alert
    means the same thing a validation alert meant. Without that, "it raised N
    alerts" would be a statement about an arbitrary cut.
    """
    thresholds = thresholds or default_thresholds(cfg)
    feature_cols = [c for c in frame.columns if c not in LABEL_COLS
                    and c != "ts_utc"]

    alerts: list[Alert] = []
    for name, model in detectors.items():
        threshold = float(thresholds.get(name, thresholds.get("default", 2.5)))
        scored = model.predict(frame, threshold=threshold, conn=conn,
                               context=f"live/{name}")
        fired = scored[scored["action"] == "FLAG"]
        for _, row in fired.iterrows():
            source = frame[(frame["ticker"] == row["ticker"]) &
                           (frame["ts_utc"] == row["ts_utc"])]
            features = ({} if source.empty
                        else {c: _clean_value(source.iloc[0][c])
                              for c in feature_cols})
            alerts.append(Alert(ts_utc=int(row["ts_utc"]),
                                ticker=str(row["ticker"]), detector=name,
                                score=float(row["score"]), threshold=threshold,
                                features=features))
    return sorted(alerts, key=lambda a: (a.ts_utc, a.detector, a.ticker))


def _clean_value(value):
    """JSON-safe: NaN is not valid JSON and would break the alert log."""
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return None if pd.isna(value) else str(value)


def default_thresholds(cfg: dict) -> dict:
    """Each detector's tuned cut, from config."""
    b = cfg["baselines"]
    return {"cusum": b["cusum"]["threshold"],
            "volume_zscore": b["volume_zscore"]["threshold"],
            "default": 0.5}          # policies emit P(FLAG)


def latest_stored_bar(conn, interval: str, ticker: str | None = None) -> int | None:
    """The newest bar already stored, so a fetch asks only for what is missing.

    `ticker` narrows it to one symbol. That matters for the benchmark, which is
    not `in_universe` and so falls behind the universe's own newest bar: asking
    globally would start the fetch after the gap and never close it.
    """
    if ticker is None:
        row = conn.execute("SELECT MAX(ts_utc) FROM bars WHERE interval = ?",
                           (interval,)).fetchone()
    else:
        row = conn.execute(
            "SELECT MAX(ts_utc) FROM bars WHERE interval = ? AND ticker = ?",
            (interval, ticker)).fetchone()
    return None if row is None or row[0] is None else int(row[0])


def fetch_latest(cfg: dict, conn, tickers: list[str] | None = None,
                 now_ts: int | None = None) -> int:
    """Append bars newer than what is stored. Never re-downloads.

    The P3-06 freeze refuses a re-download because yfinance restates history
    after splits, which would silently change bars already used in results.
    Appends are explicitly allowed, and `market.py`'s guard names this monitor
    as the reason it is written that way — it starts from the newest stored
    bar, so `assert_not_frozen`'s `requested_start_ts < stamp_ts` check never
    trips.
    """
    from src.collectors import market

    interval = cfg["market"]["interval"]
    now_ts = int(now_ts if now_ts is not None else utc_now_ts())
    start = latest_stored_bar(conn, interval)
    if start is None:
        raise SystemExit(
            "no bars stored at all — run the Phase 3 collector before the "
            "monitor; this appends to a snapshot, it does not create one.")
    if tickers is None:
        tickers = [r[0] for r in conn.execute(
            "SELECT ticker FROM companies WHERE in_universe = 1 ORDER BY ticker")]

    # +1 second: the stored bar is already held, and `collect_many` refuses an
    # empty or inverted range rather than treating it as a silent no-op.
    # `resume=False` is load-bearing, not a default left alone. `collect_many`
    # skips every ticker whose `fetch_state` row says "ok", and that row is
    # keyed on the ticker ALONE — it carries no window. Under `resume=True` the
    # first run to populate `fetch_state` would make every later run skip every
    # ticker and append nothing, while still reporting success. It survived
    # earlier runs only because each began from the bootstrap, whose
    # `fetch_state` is empty; the first run to restore a warm cache would have
    # gone quiet. Resume answers "continue an interrupted backfill"; each
    # monitor pass is a fresh window, so there is nothing to resume.
    fetched = 0
    if start + 1 < now_ts:
        fetched += market.collect_many(cfg, conn, tickers, start + 1, now_ts,
                                       interval, resume=False)

    # The benchmark is fetched separately, from ITS own newest bar. It is not
    # `in_universe`, so it is absent from the list above; left out, every
    # `ret_rel_*` feature silently degrades to NaN once the universe's bars run
    # past the benchmark's last one — measured at 74.8% of live alerts before
    # this fix. Its own start closes the gap that had already opened.
    benchmark = cfg["market"]["benchmark"]
    if benchmark not in set(tickers):
        bench_start = latest_stored_bar(conn, interval, ticker=benchmark)
        if bench_start is not None and bench_start + 1 < now_ts:
            fetched += market.collect_many(cfg, conn, [benchmark],
                                           bench_start + 1, now_ts,
                                           interval, resume=False)
    return fetched


def fetch_recent_filings(cfg: dict, conn, tickers: list[str] | None = None,
                         since_ts: int | None = None,
                         now_ts: int | None = None,
                         client=None) -> dict:
    """Append 8-Ks filed since the last one on record. The other half of live.

    Without this the monitor collects alerts it can never score. `backfill`
    only answers an alert whose whole 48-hour window falls at or before the
    newest filing held locally, so a filing horizon that never moves means
    every new alert sits deferred forever — the monitor would run for weeks,
    raise hundreds of alerts, and never learn whether one of them was right.
    That was the state of P7-04 until this was added.

    Three details carry the weight:

    **`force=True`.** `EdgarClient` is cache-first by design (P2-01), which is
    exactly right for rebuilding a fixed historical window and exactly wrong
    here: the cached submissions file would be returned unchanged and no new
    filing would ever appear. The cache is bypassed and refreshed on purpose.

    **The window starts BEFORE the horizon.** `live.filing_overlap_hours` of
    deliberate overlap, because SEC acceptance times are not strictly ordered
    against when a record becomes visible, and amendments arrive late. Re-reading
    a filing costs nothing — the upsert is idempotent — while missing one at the
    boundary would silently cost an outcome.

    **Only `recent` is read.** `pages_to_fetch` selects the older paginated
    files by date overlap, and a window that starts days ago overlaps none of
    them, so this is one request per company rather than a full history walk.
    """
    from src.collectors.edgar import (EdgarClient, fetch_company_filings,
                                      filing_rows)
    from src import db as _db

    now_ts = int(now_ts if now_ts is not None else utc_now_ts())
    if since_ts is None:
        row = conn.execute("SELECT MAX(acceptance_utc) FROM filings "
                           "WHERE acceptance_utc IS NOT NULL").fetchone()
        if row is None or row[0] is None:
            raise SystemExit(
                "no filings stored at all — run the Phase 2 collector before "
                "the monitor; this appends to a history, it does not build one.")
        overlap = int((cfg.get("live") or {}).get("filing_overlap_hours", 72))
        since_ts = int(row[0]) - overlap * 3600

    companies = _db.companies_for_collection(conn, tickers)
    if not companies:
        raise SystemExit("no companies to collect filings for.")

    client = client or EdgarClient(cfg)
    records = new_rows = failed = 0
    for company in companies:
        try:
            fetched = fetch_company_filings(cfg, client, company["cik"],
                                            since_ts=since_ts, until_ts=now_ts,
                                            force=True)
            rows = filing_rows(cfg, fetched, company["cik"], company["ticker"])
            records += len(fetched)
            new_rows += _db.upsert_filings(conn, rows)
        except Exception as exc:                  # one 404 must not cost the rest
            failed += 1
            log.warning("filings: %s (%s) failed: %s: %s", company["ticker"],
                        company["cik"], type(exc).__name__, exc)
    conn.commit()

    # Run-level, exactly as the Phase 2 collector reasons about it: a company
    # with no new 8-K is ordinary, but zero RECORDS across every company means
    # the endpoint returned nothing and the run must not report success.
    if companies and records == 0:
        raise SystemExit(
            f"EDGAR returned zero records across all {len(companies)} "
            f"companies — refusing to report success. The endpoint is "
            f"unreachable, throttling, or the User-Agent was rejected.")

    return {"companies": len(companies), "records": records,
            "new_filings": new_rows, "failed": failed, "since_utc": since_ts}


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="score what is already stored; fetch nothing")
    ap.add_argument("--policy-run", action="append", dest="policy_runs",
                    default=None, help="a P6-03 run directory; repeatable")
    ap.add_argument("--limit-tickers", type=int, default=None)
    ap.add_argument("--as-of", type=int, default=None,
                    help="score as if it were this UTC epoch second")
    ap.add_argument("--no-log", action="store_true",
                    help="print alerts without appending them to the log")
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])

    if not args.dry_run:
        print("fetching the latest bars...")
        rows = fetch_latest(cfg, conn)
        print(f"  appended {rows:,} bar rows")

    tickers = None
    if args.limit_tickers:
        tickers = [r[0] for r in conn.execute(
            "SELECT ticker FROM companies WHERE in_universe = 1 "
            "ORDER BY ticker LIMIT ?", (args.limit_tickers,))]

    frame = conform(latest_bar_frame(cfg, conn, tickers, as_of=args.as_of))
    latest = int(frame["ts_utc"].max())
    print(f"scoring {len(frame):,} bars across "
          f"{frame['ticker'].nunique():,} tickers; newest bar "
          f"{ts_to_iso(latest)} ({split_of(cfg, latest)})")

    detectors = build_detectors(cfg, args.policy_runs)
    alerts = scan(cfg, conn, frame, detectors)

    if not alerts:
        print("no alerts.")
        return
    print(f"\n{len(alerts)} alert(s):")
    for a in alerts:
        print(f"  {ts_to_iso(a.ts_utc)}  {a.ticker:6s}  {a.detector:18s} "
              f"score {a.score:8.4f} (cut {a.threshold})")

    if not args.no_log:
        from src.live.alertlog import append
        # Already-logged alerts are skipped, not restated: the first write
        # wins, so re-running over the same hours is a no-op by design.
        new = append(conn, alerts)
        print(f"\nlogged {new} new alert(s); "
              f"{len(alerts) - new} already on record")


if __name__ == "__main__":
    main()
