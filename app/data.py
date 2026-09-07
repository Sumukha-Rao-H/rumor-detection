"""Data loading for the dashboard. Every read is cached; none of it writes.

The dashboard is a READER. It opens the database read-only and never calls a
collector, so it cannot move the frozen snapshot, spend an API quota, or touch
the sealed test split by accident.

Timestamps come out of here as UTC epoch integers, exactly as they are stored.
Formatting — and the timezone label that `UI-context.md` rule 7 requires on
every displayed time — happens at the point of display, never here.
"""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

import pandas as pd
import streamlit as st

from src import db
from src.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
ALERT_LOG = REPO / "live-log" / "alerts.csv"


@st.cache_data(ttl=300)
def config() -> dict:
    return load_config()


def _conn():
    """Read-only connection, wrapped so `with` actually CLOSES it.

    Not cached: SQLite connections are not shareable across Streamlit's script
    reruns, and reopening costs microseconds. `closing` is the point — a bare
    `with sqlite3.connect(...)` commits or rolls back and leaves the handle
    open, so every cached read here leaked one for the life of the session.
    """
    return closing(db.get_conn(config()["paths"]["db"], readonly=True))


# --------------------------------------------------------------------------
# alerts
# --------------------------------------------------------------------------
@st.cache_data(ttl=300)
def alerts() -> pd.DataFrame:
    """The live alert log, with its features unpacked into columns.

    Read from the CSV in git rather than the database on purpose: that file is
    the durable, append-only record the monitor commits after every run, and
    the local database may be a rebuild that never saw those rows.
    """
    if not ALERT_LOG.exists():
        return pd.DataFrame()
    df = pd.read_csv(ALERT_LOG)
    if df.empty:
        return df

    feats = df["features"].map(lambda s: json.loads(s) if isinstance(s, str) else {})
    for col in sorted({k for d in feats for k in d}):
        df[col] = feats.map(lambda d, c=col: d.get(c))
    return df.sort_values("ts_utc", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=300)
def outcomes() -> pd.DataFrame:
    """Backfilled outcomes: did a filing follow within the window?

    `filed` is deliberately three-valued once joined — 1 filed, 0 did not, and
    MISSING for an alert whose window has not closed yet. Counting a pending
    alert as a miss would understate the hit rate every single day.
    """
    with _conn() as conn:
        try:
            return pd.read_sql_query(
                "SELECT alert_id, checked_utc, filed, accession_no, item_code, "
                "t0_utc, lead_trading_h FROM alert_outcomes", conn)
        except Exception:
            return pd.DataFrame(columns=["alert_id", "filed"])


@st.cache_data(ttl=300)
def alerts_with_outcomes() -> pd.DataFrame:
    a, o = alerts(), outcomes()
    if a.empty:
        return a
    return a.merge(o, on="alert_id", how="left", suffixes=("", "_outcome"))


def hit_rate(df: pd.DataFrame) -> tuple[int, int, float | None]:
    """(resolved, filed, rate) over alerts whose window has actually closed.

    Pending alerts are excluded from BOTH numerator and denominator, which is
    the only way the figure means "of the ones we can grade, how many were
    right" rather than drifting with how recently the monitor last ran.
    """
    if df.empty or "filed" not in df:
        return 0, 0, None
    resolved = df[df["filed"].notna()]
    if resolved.empty:
        return 0, 0, None
    filed = int((resolved["filed"] == 1).sum())
    return len(resolved), filed, filed / len(resolved)


# --------------------------------------------------------------------------
# the alert budget — UI-context.md rules 5 and 6
# --------------------------------------------------------------------------
@st.cache_data(ttl=300)
def universe_size() -> int:
    with _conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM companies WHERE in_universe = 1").fetchone()[0]


def budget_line(df: pd.DataFrame) -> dict:
    """How much of this month's allowance the log has spent.

    The budget is the constraint the whole system is tuned to, so rule 6 puts
    it on screen rather than in a footnote. Counted over the calendar month of
    the newest alert, not "now": the monitor runs daily and a viewer opening
    this on the 1st should still see the month the data is about.
    """
    cfg = config()
    rate = cfg["eval"]["alert_budget_per_stock_per_month"]
    n_universe = universe_size()
    allowance = int(rate * n_universe)

    used = 0
    month = None
    if not df.empty:
        # tz dropped explicitly rather than by pandas' warning: these are UTC
        # epoch seconds, so the calendar month IS the UTC month and there is
        # no local-time question to get wrong.
        ts = pd.to_datetime(df["ts_utc"], unit="s", utc=True).dt.tz_localize(None)
        month = ts.max().to_period("M")
        used = int((ts.dt.to_period("M") == month).sum())
    return {"rate": rate, "universe": n_universe, "allowance": allowance,
            "used": used, "month": str(month) if month is not None else "—"}


# --------------------------------------------------------------------------
# per-ticker detail
# --------------------------------------------------------------------------
@st.cache_data(ttl=300)
def bars(ticker: str, lo_utc: int, hi_utc: int) -> pd.DataFrame:
    cfg = config()
    with _conn() as conn:
        return pd.read_sql_query(
            "SELECT ts_utc, open, high, low, close, volume FROM bars "
            "WHERE ticker = ? AND interval = ? AND ts_utc BETWEEN ? AND ? "
            "ORDER BY ts_utc",
            conn, params=(ticker, cfg["market"]["interval"], lo_utc, hi_utc))


@st.cache_data(ttl=300)
def news(ticker: str, lo_utc: int, hi_utc: int) -> pd.DataFrame:
    with _conn() as conn:
        return pd.read_sql_query(
            "SELECT published_utc, title, source_name, source_tier FROM news "
            "WHERE ticker = ? AND published_utc BETWEEN ? AND ? "
            "ORDER BY published_utc DESC",
            conn, params=(ticker, lo_utc, hi_utc))


@st.cache_data(ttl=300)
def filings(ticker: str, limit: int = 20) -> pd.DataFrame:
    cfg = config()
    forms = cfg["edgar"]["forms"]
    marks = ",".join("?" * len(forms))
    with _conn() as conn:
        return pd.read_sql_query(
            f"SELECT accession_no, form, items, acceptance_utc FROM filings "
            f"WHERE ticker = ? AND form IN ({marks}) "
            f"AND acceptance_utc IS NOT NULL "
            f"ORDER BY acceptance_utc DESC LIMIT ?",
            conn, params=(ticker, *forms, limit))


# --------------------------------------------------------------------------
# evaluation tables
# --------------------------------------------------------------------------
@st.cache_data(ttl=300)
def comparison(name: str) -> pd.DataFrame:
    path = Path(config()["paths"]["processed"]) / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
