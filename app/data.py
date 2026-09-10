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

    Call `db_present()` first. `get_conn(readonly=True)` raises rather than
    fabricating an empty database, which is the right behaviour and the reason
    every reader below asks before it opens.
    """
    return closing(db.get_conn(config()["paths"]["db"], readonly=True))


def db_present() -> bool:
    """Is there a database to read at all?

    A MISSING DATABASE IS AN ORDINARY STATE HERE, not a failure. The database
    is gitignored and about a gigabyte of rebuildable cache; `live-log/
    alerts.csv` is committed and is the durable record. So a fresh clone — an
    examiner, a second machine — has the evidence file and nothing beside it.

    Every read below asks this before opening a connection. It used not to,
    and the consequence was worse than a missing chart: `budget_line` runs
    before routing, so a fresh clone met a raw `FileNotFoundError` traceback on
    every screen, including *Today's alerts*, which needs nothing but the CSV.
    An empty frame and a screen that says what is missing is the honest answer;
    a stack trace is not.
    """
    return Path(config()["paths"]["db"]).exists()


def window_hours() -> int:
    """The outcome window, in WALL-CLOCK hours, from config.

    Wall-clock, not trading hours: a company can file overnight or at a
    weekend, so "did a filing follow within two days" is a question about
    elapsed time. `src.live.outcomes` reasons about it the same way and two
    captions on these screens used to call it "48 trading hours", which is a
    different quantity — roughly seven days rather than two.
    """
    return int((config().get("live") or {}).get("outcome_window_hours", 48))


@st.cache_data(ttl=300)
def answerable_edge() -> int | None:
    """The newest instant at which an alert's window can close and be graded.

    The same edge `src.live.outcomes.backfill` uses: the newest filing
    acceptance time held locally, pulled back by `news.t0_lookback_hours`
    because t₀ is min(acceptance, earliest news) and so can be earlier than the
    acceptance time the horizon is measured on. An alert whose window closes at
    or before this could have been answered; one whose window closes after it
    genuinely cannot be, yet.

    Returns None when there is no database, or no filing in it — in which case
    nothing can be graded and no alert should be described as "still open".
    """
    if not db_present():
        return None
    with _conn() as conn:
        row = conn.execute("SELECT MAX(acceptance_utc) FROM filings "
                           "WHERE acceptance_utc IS NOT NULL").fetchone()
    if row is None or row[0] is None:
        return None
    lookback = int((config().get("news") or {}).get("t0_lookback_hours", 0))
    return int(row[0]) - lookback * 3600


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


_OUTCOME_COLS = ["alert_id", "checked_utc", "filed", "accession_no",
                 "item_code", "t0_utc", "lead_trading_h"]


@st.cache_data(ttl=300)
def outcomes() -> pd.DataFrame:
    """Backfilled outcomes: did a filing follow within the window?

    `filed` is 1 filed, 0 did not, and MISSING for an alert this database has
    no outcome row for at all. Counting a missing row as a miss would
    understate the hit rate; calling every one of them "pending" — which is
    what the join used to imply — overstates how much of the log is genuinely
    unanswerable. `outcome_state` below separates those two cases.

    The empty frame is returned with the full column list rather than two
    columns, so a caller that merges on it gets the same shape either way. The
    old `try` sat INSIDE the `with`, so it never caught the one error that
    actually happens here: a missing database file, raised while opening.
    """
    if not db_present():
        return pd.DataFrame(columns=_OUTCOME_COLS)
    try:
        with _conn() as conn:
            return pd.read_sql_query(
                "SELECT alert_id, checked_utc, filed, accession_no, item_code, "
                "t0_utc, lead_trading_h FROM alert_outcomes", conn)
    except Exception:
        return pd.DataFrame(columns=_OUTCOME_COLS)


#: The four states an alert can be in, and the words each gets on screen.
#:
#: The design was three: filed, not filed, and not answerable yet. A fourth
#: arrived silently and was collapsed into the third. The committed CSV and the
#: local database are SEPARATE records — the CSV is the durable one and the
#: database is a rebuildable cache — so an alert can be in the log with no
#: outcome row beside it, long after its window closed. Calling that "the
#: window has not closed" is false, and it hides that the hit rate's
#: denominator is a small fraction of the alerts that could be graded.
OUTCOME_WORDS = {
    "filed": "8-K followed",
    "none": "no 8-K in window",
    "open": "window still open",
    "unscored": "not scored — no outcome in this database",
}


def _outcome_states(df: pd.DataFrame) -> pd.Series:
    """Which of the four states each alert is in, derived from the DATA.

    Derived from the alert's own timestamp against the answerable edge, not
    from whether the join found a row. The join can only ever say "there is no
    outcome here", which is a statement about this database rather than about
    the alert.
    """
    filed = (df["filed"] if "filed" in df
             else pd.Series(float("nan"), index=df.index, dtype="float64"))
    state = pd.Series("unscored", index=df.index, dtype="object")
    state[filed == 1] = "filed"
    state[filed == 0] = "none"

    edge = answerable_edge()
    if edge is not None:
        closes = df["ts_utc"].astype("int64") + window_hours() * 3600
        state[filed.isna() & (closes > edge)] = "open"
    return state


@st.cache_data(ttl=300)
def alerts_with_outcomes() -> pd.DataFrame:
    a, o = alerts(), outcomes()
    if a.empty:
        return a
    merged = a.merge(o, on="alert_id", how="left", suffixes=("", "_outcome"))
    merged["outcome_state"] = _outcome_states(merged)
    return merged


def coverage() -> dict:
    """How much of the committed log this database can actually say anything about.

    Surfaced on screen rather than left for a reader to infer from a
    denominator. The two records disagree by construction — the CSV is
    append-only and committed on every run, the database is a cache that may
    have been rebuilt since — and the size of the disagreement is the honest
    context for every rate on the live screen.
    """
    df = alerts_with_outcomes()
    if df.empty:
        return {"logged": 0, "graded": 0, "open": 0, "unscored": 0,
                "answerable": 0}
    state = df["outcome_state"]
    return {
        "logged": len(df),
        "graded": int(state.isin(("filed", "none")).sum()),
        "open": int((state == "open").sum()),
        "unscored": int((state == "unscored").sum()),
        # Every alert whose window has closed — graded or not. The denominator
        # the hit rate WOULD have if this database held every outcome.
        "answerable": int((state != "open").sum()),
    }


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


def _is_scheduled(item_code, codes: set[str]) -> bool:
    """Did this 8-K carry a scheduled item code?

    `item_code` is the comma-joined list of every item on the filing, so a
    filing carrying 2.02 alongside three others is still scheduled: the results
    announcement had a date published weeks in advance, and a run-up before one
    is exactly the easy half rule 7 refuses to pool. Codes are compared as
    STRINGS — 1.10 and 1.1 are different item codes and a float destroys that.
    """
    if not isinstance(item_code, str):
        return False
    return any(part.strip() in codes for part in item_code.split(","))


def split_hit_rates(df: pd.DataFrame) -> dict:
    """The live hit rate, split scheduled vs unscheduled. AGENTS rule 7.

    A pooled figure is not allowed anywhere in this project, and the live
    screen is no exception: of the alerts graded so far, most of the hits are
    results announcements, whose dates were published weeks ahead. Pooling lets
    that easy half carry the headline — the pooled rate is more than double the
    unscheduled one — and the unscheduled number is the one the project exists
    to produce.

    The denominator is the same for both: every alert whose outcome is known.
    An alert that was followed by nothing is a miss on both sides, and there is
    no event to attach a slice to, so splitting the denominator would mean
    inventing one. What is split is the numerator, which is what "of the alerts
    we graded, how many anticipated an unscheduled disclosure" asks.
    """
    resolved, filed, rate = hit_rate(df)
    out = {"resolved": resolved, "filed": filed, "pooled": rate,
           "scheduled": 0, "scheduled_rate": None,
           "unscheduled": 0, "unscheduled_rate": None}
    if not resolved or "item_code" not in df:
        return out

    codes = {str(c) for c in config()["items"]["scheduled"]}
    hits = df[df["filed"] == 1]
    scheduled = int(hits["item_code"].map(
        lambda c: _is_scheduled(c, codes)).sum())
    out.update(scheduled=scheduled, scheduled_rate=scheduled / resolved,
               unscheduled=filed - scheduled,
               unscheduled_rate=(filed - scheduled) / resolved)
    return out


def outcome_slice(df: pd.DataFrame) -> pd.Series:
    """Per-alert: was the 8-K that followed scheduled, unscheduled, or none?"""
    codes = {str(c) for c in config()["items"]["scheduled"]}
    item = (df["item_code"] if "item_code" in df
            else pd.Series(None, index=df.index, dtype="object"))
    return pd.Series(
        [("scheduled" if _is_scheduled(c, codes) else "unscheduled")
         if isinstance(c, str) and c else "—" for c in item],
        index=df.index, dtype="object")


# --------------------------------------------------------------------------
# the alert budget — UI-context.md rules 5 and 6
# --------------------------------------------------------------------------
@st.cache_data(ttl=300)
def universe_size() -> int | None:
    """How many companies are in the universe, or None if there is no database.

    None rather than 0, and the distinction is the point: 0 would mean "the
    universe is empty", which is a claim about the study. None means "this
    clone cannot answer that", and every figure derived from it renders as "—"
    instead of a confident zero.
    """
    if not db_present():
        return None
    with _conn() as conn:
        return int(conn.execute(
            "SELECT COUNT(*) FROM companies WHERE in_universe = 1").fetchone()[0])


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
    # No universe, no allowance. The alerts themselves come from the committed
    # CSV and are still countable, so "used" stays real while the two figures
    # that need the database render as "—" rather than as a fabricated zero.
    allowance = None if n_universe is None else int(rate * n_universe)

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
    if not db_present():
        return pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close",
                                     "volume"])
    with _conn() as conn:
        return pd.read_sql_query(
            "SELECT ts_utc, open, high, low, close, volume FROM bars "
            "WHERE ticker = ? AND interval = ? AND ts_utc BETWEEN ? AND ? "
            "ORDER BY ts_utc",
            conn, params=(ticker, cfg["market"]["interval"], lo_utc, hi_utc))


@st.cache_data(ttl=300)
def news(ticker: str, lo_utc: int, hi_utc: int) -> pd.DataFrame:
    if not db_present():
        return pd.DataFrame(columns=["published_utc", "title", "source_name",
                                     "source_tier"])
    with _conn() as conn:
        return pd.read_sql_query(
            "SELECT published_utc, title, source_name, source_tier FROM news "
            "WHERE ticker = ? AND published_utc BETWEEN ? AND ? "
            "ORDER BY published_utc DESC",
            conn, params=(ticker, lo_utc, hi_utc))


@st.cache_data(ttl=300)
def filings(ticker: str, limit: int = 20) -> pd.DataFrame:
    cfg = config()
    if not db_present():
        return pd.DataFrame(columns=["accession_no", "form", "items",
                                     "acceptance_utc"])
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
