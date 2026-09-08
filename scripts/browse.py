#!/usr/bin/env python3
"""Browse every dataset this project generated, row by row.

The phase PDFs summarise. This does the opposite: it shows the actual rows,
one at a time if you want, from any table or file the pipeline produced —
without needing to remember whether a thing lives in SQLite, a parquet file or
a CSV. One command, one mental model.

  python scripts/browse.py --list                      what exists at all
  python scripts/browse.py bars                        first 20 rows, as a table
  python scripts/browse.py events --detail             one record per block
  python scripts/browse.py bars --where "ticker='AAPL'" --limit 5
  python scripts/browse.py events --next               page forward
  python scripts/browse.py features --export out.csv   open it in a spreadsheet

Timestamps are stored as UTC epoch seconds everywhere in this project, which is
right for arithmetic and unreadable for a person. Any column ending `_utc` is
rendered as a readable date unless you pass --raw.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd                                    # noqa: E402

from src.utils.config import load_config               # noqa: E402
from src.utils.timeutils import ts_to_iso              # noqa: E402

#: Where paging state is kept, so `--next` means something between runs.
STATE = Path.home() / ".cache" / "rumor-detection-browse.json"

#: One-line descriptions, so `--list` explains what each thing IS rather than
#: only how big it is. A row count with no meaning attached is not an inventory.
WHAT_IT_IS = {
    "companies": "every candidate company; in_universe=1 marks the 1,500 studied",
    "filings":   "every 8-K ever filed by them (back to 1994) — the answer key",
    "events":    "filings turned into labelled events, with t0 and usability",
    "bars":      "hourly + daily price/volume history, frozen 2026-08-30",
    "news":      "articles matched to tickers, used to correct t0 earlier",
    "alerts":    "what the LIVE monitor flagged — append-only, hash-chained",
    "alert_outcomes": "did an 8-K actually follow each live alert within 48h",
    "fetch_state": "collector bookkeeping: what was fetched, what failed",
    "meta":      "database stamps — the snapshot freeze, the split seal",
    "features":  "the model's inputs — one row per ticker-hour, news features off",
    "features-with-news": "the same matrix with Phase 8's news-coverage columns on",
    "baseline-comparison": "Phase 5 result table (all baselines, all slices)",
    "phase6-comparison": "Phase 6 result table (policy vs baselines)",
    "rl-seeds":  "the 5 RL training seeds and what each scored",
    "p8-with-news": "Phase 8 ablation, news channel ON",
    "p8-without-news": "Phase 8 ablation, news channel OFF — the control",
    "p8-delta":  "Phase 8: what the news channel bought, with its sign",
    "p8-policy-seeds": "Phase 8, the learned policy across seeds",
    "phase10-final": "THE FINAL TEST-SET NUMBERS — unsealed and run once",
    "live-alerts": "the durable alert log committed to git",
}

#: Files, as opposed to SQLite tables.
#:
#: Resolved against the repo root, not the working directory. They used to be
#: bare relative paths, so running this from anywhere but the repo root printed
#: "(not generated yet)" for EVERY file — an inventory tool answering the one
#: question it exists to answer, confidently and wrongly, while the SQLite half
#: of the same listing kept working because `load_config` resolves its paths
#: properly.
FILES = {
    name: (REPO_ROOT / rel, kind) for name, (rel, kind) in {
        "features": ("data/processed/features.parquet", "parquet"),
        "features-with-news": ("data/processed/features-with-news.parquet", "parquet"),
        "baseline-comparison": ("data/processed/baseline-comparison-val.csv", "csv"),
        "phase6-comparison": ("data/processed/phase6-comparison-val.csv", "csv"),
        "rl-seeds": ("data/processed/rl-policy-seeds-val.csv", "csv"),
        "p8-with-news": ("data/processed/p8-with-news-val.csv", "csv"),
        "p8-without-news": ("data/processed/p8-without-news-val.csv", "csv"),
        "p8-delta": ("data/processed/p8-news-ablation-delta.csv", "csv"),
        "p8-policy-seeds": ("data/processed/p8-policy-ablation-seeds.csv", "csv"),
        "phase10-final": ("data/processed/phase10/FINAL-test-evaluation.csv", "csv"),
        "live-alerts": ("live-log/alerts.csv", "csv"),
    }.items()
}


def _conn(cfg):
    from src import db
    return db.get_conn(cfg["paths"]["db"], readonly=True)


def tables(conn) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]


def humanise(df: pd.DataFrame, raw: bool = False) -> pd.DataFrame:
    """Render epoch-second columns as dates.

    Every timestamp in this project is UTC epoch seconds — correct for
    arithmetic, unreadable on a screen. `--raw` turns this off for when you
    need the number itself.
    """
    if raw:
        return df
    out = df.copy()
    for col in out.columns:
        if col.endswith("_utc") and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: ts_to_iso(int(v)) if pd.notna(v) else None)
    return out


def load(cfg, name: str, where: str | None, limit: int, offset: int):
    """Rows plus the total count, from a table or a file, uniformly."""
    if name in FILES:
        path, kind = FILES[name]
        if not os.path.exists(path):
            raise SystemExit(f"{path} does not exist yet.")
        df = (pd.read_parquet(path) if kind == "parquet"
              else pd.read_csv(path))
        if where:
            df = df.query(where)
        return df.iloc[offset:offset + limit], len(df)

    conn = _conn(cfg)
    if name not in tables(conn):
        raise SystemExit(
            f"no dataset called {name!r}. Run --list to see what exists.")
    clause = f" WHERE {where}" if where else ""
    total = conn.execute(f'SELECT COUNT(*) FROM "{name}"{clause}').fetchone()[0]
    rows = conn.execute(
        f'SELECT * FROM "{name}"{clause} LIMIT ? OFFSET ?',
        (limit, offset)).fetchall()
    return pd.DataFrame([dict(r) for r in rows]), total


def show_list(cfg) -> None:
    conn = _conn(cfg)
    print("DATABASE TABLES        rows        what it is")
    print("-" * 78)
    for t in tables(conn):
        n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        print(f"  {t:20s} {n:>10,}  {WHAT_IT_IS.get(t, '')}")

    print("\nFILES                  rows        what it is")
    print("-" * 78)
    for name, (path, kind) in FILES.items():
        if not os.path.exists(path):
            print(f"  {name:20s} {'—':>10}  (not generated yet: {path})")
            continue
        n = (len(pd.read_parquet(path)) if kind == "parquet"
             else sum(1 for _ in open(path)) - 1)
        print(f"  {name:20s} {n:>10,}  {WHAT_IT_IS.get(name, '')}")

    # Two locations, both real: Phase 6 wrote to `rl.runs_dir`, Phase 8's
    # ablation wrote its ten seed runs under `runs/p8`. Listing only the first
    # made half the training record invisible to the tool that claims to show
    # every dataset this project generated.
    run_roots = [REPO_ROOT / cfg["rl"]["runs_dir"], REPO_ROOT / "runs"]
    runs = sorted({m for root in run_roots if root.exists()
                   for m in root.glob("**/manifest.json")})
    if runs:
        print(f"\nRL TRAINING RUNS: {len(runs)} — each with a manifest.json "
              f"recording seed, config and data fingerprint")
        for r in runs[:5]:
            print(f"  {r.parent.relative_to(REPO_ROOT)}")
        if len(runs) > 5:
            print(f"  ... and {len(runs) - 5} more")

    print("\nBrowse any of them:  python scripts/browse.py <name> --detail")


def show_detail(df: pd.DataFrame, start: int) -> None:
    """One record per block — the 'one by one' view.

    A wide table printed as a grid is unreadable past about six columns; this
    prints every field of one row on its own line, which is what you want when
    you are checking a single event or a single alert rather than scanning.
    """
    width = max((len(c) for c in df.columns), default=0)
    for i, (_, row) in enumerate(df.iterrows(), start=start):
        print(f"\n───────── row {i:,} " + "─" * 46)
        for col in df.columns:
            value = row[col]
            if isinstance(value, str) and len(value) > 200:
                value = value[:200] + f" … ({len(value)} chars)"
            print(f"  {col:<{width}}  {value}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", help="table or file name")
    ap.add_argument("--list", action="store_true", help="inventory everything")
    ap.add_argument("--detail", action="store_true",
                    help="one record per block, every field on its own line")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=None)
    ap.add_argument("--next", action="store_true",
                    help="continue from where the last view of this dataset ended")
    ap.add_argument("--where", help="SQL WHERE for tables, pandas query for files")
    ap.add_argument("--columns", help="comma-separated subset of columns")
    ap.add_argument("--raw", action="store_true",
                    help="leave *_utc columns as epoch seconds")
    ap.add_argument("--export", metavar="PATH",
                    help="write the selected rows to CSV instead of printing")
    args = ap.parse_args()

    cfg = load_config()
    if args.list or not args.dataset:
        show_list(cfg)
        return

    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if args.offset is not None:
        offset = args.offset
    elif args.next:
        offset = int(state.get(args.dataset, 0))
    else:
        offset = 0

    df, total = load(cfg, args.dataset, args.where, args.limit, offset)
    if args.columns:
        keep = [c.strip() for c in args.columns.split(",")]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise SystemExit(f"no such column(s): {missing}\n"
                             f"available: {list(df.columns)}")
        df = df[keep]
    df = humanise(df, args.raw)

    if args.export:
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.export, index=False)
        print(f"wrote {len(df):,} rows x {len(df.columns)} cols -> {args.export}")
        return

    shown_to = offset + len(df)
    print(f"{args.dataset}: rows {offset:,}–{shown_to:,} of {total:,}"
          + (f"   where {args.where}" if args.where else ""))

    if df.empty:
        print("(no rows)")
        return
    if args.detail:
        show_detail(df, offset)
    else:
        with pd.option_context("display.max_columns", None,
                               "display.width", 200,
                               "display.max_colwidth", 28):
            print(df.to_string(index=False))

    STATE.parent.mkdir(parents=True, exist_ok=True)
    state[args.dataset] = shown_to
    STATE.write_text(json.dumps(state))
    if shown_to < total:
        print(f"\n… {total - shown_to:,} more. Next page:  "
              f"python scripts/browse.py {args.dataset} --next"
              + (" --detail" if args.detail else ""))


if __name__ == "__main__":
    main()
