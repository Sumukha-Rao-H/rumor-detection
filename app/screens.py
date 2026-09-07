"""The four dashboard screens (P9-02 … P9-05).

Each is a plain function taking no arguments and rendering into the current
Streamlit page. `dashboard.py` owns routing, the header and the disclaimer, so
a screen cannot render without them.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data, ui

HOUR = 3600
DAY = 86400

#: Rendered beside a feature so a number carries a comparison, not just a
#: value — "4.2x trailing 20-day volume", never "4.2".
_REASON = {
    "volume_z": lambda v: f"volume {v:+.1f} sd vs its own trailing normal",
    "ret_rel_4h": lambda v: f"{v:+.2%} vs SPY over 4h",
    "ret_rel_24h": lambda v: f"{v:+.2%} vs SPY over 24h",
    "ret_4h": lambda v: f"{v:+.2%} over 4h",
    "ret_24h": lambda v: f"{v:+.2%} over 24h",
    "days_since_last_8k": lambda v: f"last 8-K {v:.0f}d ago",
    "hours_since_news": lambda v: f"last article {v/24:.1f}d ago",
    "news_count_24h": lambda v: f"{v:.0f} articles in 24h",
    "volatility": lambda v: f"volatility {v:.2%}/h",
}

#: Order matters: the first few are what a triage analyst reads first.
_REASON_ORDER = ["volume_z", "ret_rel_4h", "ret_rel_24h", "ret_4h",
                 "days_since_last_8k", "hours_since_news", "news_count_24h"]


def _reasons(row: pd.Series, limit: int = 4) -> list[str]:
    """The features that triggered this alert, rendered with their units.

    Rule 1: reasons travel WITH the alert, in the row, never behind a click.
    A number with no reason attached is a black box, and the point of the
    dashboard is that a human can sanity-check it in about thirty seconds.
    """
    out = []
    for key in _REASON_ORDER:
        if key in row.index and pd.notna(row[key]):
            out.append(_REASON[key](row[key]))
        if len(out) >= limit:
            break
    return out or ["(features not recorded for this alert)"]


# --------------------------------------------------------------------------
# P9-02 — today's alerts
# --------------------------------------------------------------------------
def alerts_today() -> None:
    df = data.alerts_with_outcomes()
    if df.empty:
        st.subheader("No alerts yet")
        st.info("The alert log is empty. The system flags roughly 2 per stock "
                "per month by design, so an empty list is a normal state, not "
                "a failure.", icon="🗓️")
        return

    newest = int(df["ts_utc"].max())
    day_start = newest - (newest % DAY)
    scope = st.radio(
        "Show", ["Most recent session", "Last 7 days", "Everything"],
        horizontal=True, label_visibility="collapsed")
    cutoff = {"Most recent session": day_start,
              "Last 7 days": newest - 7 * DAY,
              "Everything": 0}[scope]
    view = df[df["ts_utc"] >= cutoff]

    resolved, filed, rate = data.hit_rate(view)
    st.markdown(ui.honest_rate(resolved, filed, rate))

    if view.empty:
        st.info("No alerts in this window. The system flags roughly 2 per "
                "stock per month by design.", icon="🗓️")
        return

    # Sorted by strength, not alphabetically — the list is a work queue.
    view = view.assign(
        _mult=view.apply(
            lambda r: ui.strength(r["score"], r["threshold"])[2], axis=1)
    ).sort_values("_mult", ascending=False)

    st.caption(f"{len(view):,} alerts · strongest first")
    for _, r in view.head(60).iterrows():
        icon, words, mult = ui.strength(r["score"], r["threshold"])
        outcome = ""
        if pd.notna(r.get("filed")):
            outcome = ("✅ 8-K followed" if r["filed"] == 1
                       else "— no 8-K in window")
        else:
            outcome = "⏳ window still open"

        with st.container(border=True):
            a, b = st.columns([1, 3])
            a.markdown(f"### `{r['ticker']}`")
            a.markdown(f"{icon} **{words}** · {mult:.1f}× threshold")
            a.caption(f"{r['detector']} · {outcome}")

            b.markdown(f"**Flagged at** {ui.utc(r['ts_utc'])}")
            b.caption(f"Noticed by the monitor at {ui.utc(r['raised_utc'], False)}")
            for line in _reasons(r):
                b.markdown(f"- {line}")


# --------------------------------------------------------------------------
# P9-03 — ticker detail
# --------------------------------------------------------------------------
def ticker_detail() -> None:
    df = data.alerts_with_outcomes()
    if df.empty:
        st.info("No alerts to inspect yet.", icon="🗓️")
        return

    tickers = sorted(df["ticker"].unique())
    ticker = st.selectbox("Ticker", tickers)
    rows = df[df["ticker"] == ticker].sort_values("ts_utc", ascending=False)
    label = {int(r.ts_utc): f"{ui.utc(r.ts_utc, False)} · {r.detector}"
             for r in rows.itertuples()}
    flagged = st.selectbox("Flagged hour", list(label), format_func=label.get)
    row = rows[rows["ts_utc"] == flagged].iloc[0]

    live = pd.isna(row.get("filed"))
    if live:
        st.warning(
            "This alert's window is still open, so **nothing after the "
            "flagged hour is shown**. Revealing what happened next would turn "
            "a surveillance tool into a hindsight demo.", icon="🔒")

    # The hard boundary. While an alert is live the chart stops at the flagged
    # hour; once resolved, the window is allowed so an analyst can review it.
    hi = int(flagged) if live else int(flagged) + 48 * HOUR
    lo = int(flagged) - 30 * DAY

    price = data.bars(ticker, lo, hi)
    if price.empty:
        st.info("No bars stored for this window.")
    else:
        ts = pd.to_datetime(price["ts_utc"], unit="s", utc=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=price["close"], name="close",
                                 line=dict(color="#1f4e79")))
        fig.add_vline(x=dt.datetime.fromtimestamp(int(flagged), dt.timezone.utc),
                      line_dash="dash", line_color="#b5502a",
                      annotation_text="flagged")
        fig.update_layout(height=300, margin=dict(t=30, b=10),
                          xaxis_title="UTC", yaxis_title="close")
        st.plotly_chart(fig, width='stretch')

        vol = go.Figure()
        vol.add_trace(go.Bar(x=ts, y=price["volume"], name="volume",
                             marker_color="#5a6672"))
        vol.add_vline(x=dt.datetime.fromtimestamp(int(flagged), dt.timezone.utc),
                      line_dash="dash", line_color="#b5502a")
        vol.update_layout(height=200, margin=dict(t=10, b=10),
                          xaxis_title="UTC", yaxis_title="volume")
        st.plotly_chart(vol, width='stretch')

    st.subheader("Why this hour was flagged")
    st.caption("Feature values at the flagged hour. A value alone means "
               "little; each is shown against what it is measured relative to.")
    for line in _reasons(row, limit=12):
        st.markdown(f"- {line}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("News")
        st.caption(
            "Headlines as published, quoted verbatim with their publisher. "
            "Some carry analyst language — that is the outlet's wording, not "
            "this tool's, and nothing here is a recommendation. They are shown "
            "because a quiet stretch before a move is the interesting shape, "
            "and you cannot see a gap without seeing the coverage.")
        n = data.news(ticker, lo, hi)
        if n.empty:
            st.caption("No articles in this window. A quiet stretch before a "
                       "move is the interesting shape, not a gap in the data — "
                       "every week of the study window was fetched for every "
                       "in-universe ticker.")
        else:
            for _, a in n.head(12).iterrows():
                st.markdown(f"`{ui.utc(a['published_utc'], False)}` — "
                            f"{a['title']}  \n*{a['source_name']}*")
    with c2:
        st.subheader("Past 8-K filings")
        f = data.filings(ticker)
        if f.empty:
            st.caption("No 8-K filings on record.")
        else:
            for _, k in f.iterrows():
                st.markdown(f"`{ui.utc(k['acceptance_utc'], False)}` — "
                            f"items **{k['items'] or '—'}**")


# --------------------------------------------------------------------------
# P9-04 — evaluation
# --------------------------------------------------------------------------
def evaluation() -> None:
    ui.no_accuracy_note()
    ui.split_note()

    table = data.comparison("p8-without-news-val.csv")
    if table.empty:
        table = data.comparison("baseline-comparison-val.csv")
    if table.empty:
        st.info("No comparison table built yet. Run "
                "`python -m src.baselines.compare --split val --out ...`")
        return

    variant = st.selectbox(
        "t₀ variant", sorted(table["t0_variant"].unique()),
        index=list(sorted(table["t0_variant"].unique())).index("news_adjusted")
        if "news_adjusted" in set(table["t0_variant"]) else 0,
        help="`filing` uses the 8-K acceptance time. `news_adjusted` uses the "
             "earlier of that and the first news article — the honest clock, "
             "and the project's main contribution.")
    slices = ["all", "scheduled", "unscheduled"]
    sl = st.radio("Slice", slices, horizontal=True)

    view = table[(table["t0_variant"] == variant) & (table["slice"] == sl)]
    cols = [c for c in ["baseline", "precision", "max_precision", "lift",
                        "recall", "median_lead_trading_h", "n_alerts",
                        "degenerate"] if c in view.columns]
    st.dataframe(view[cols].sort_values("precision", ascending=False),
                 width='stretch', hide_index=True)

    st.caption(
        "`max_precision` is the ceiling, not a typo: the budget is SPENT, not "
        "capped, so when it exceeds the number of events even a flawless "
        "detector cannot reach 1.0. Read precision against it. **If a simple "
        "baseline wins, it is shown winning** — that is the finding, not a "
        "failure to hide.")

    with_news = data.comparison("p8-with-news-val.csv")
    if not with_news.empty:
        st.subheader("Phase 8 — does the news channel help?")
        a = table[(table.t0_variant == variant) & (table.baseline == "gradient_boosting")]
        b = with_news[(with_news.t0_variant == variant)
                      & (with_news.baseline == "gradient_boosting")]
        m = a.merge(b, on="slice", suffixes=("_without", "_with"))
        m = m[m["slice"].isin(slices)]
        m["precision_delta"] = m.precision_with - m.precision_without
        m["lead_delta_h"] = (m.median_lead_trading_h_with
                             - m.median_lead_trading_h_without)
        st.dataframe(
            m[["slice", "precision_without", "precision_with",
               "precision_delta", "median_lead_trading_h_without",
               "median_lead_trading_h_with", "lead_delta_h"]],
            width='stretch', hide_index=True)
        st.caption(
            "Gradient boosting only — it is the sole baseline that reads more "
            "than one column, so it is the only one that can carry this "
            "comparison. Lead time falls where precision rises: press coverage "
            "accumulates close to the event, so it buys confidence at the cost "
            "of warning. The delta is reported with its sign either way.")


# --------------------------------------------------------------------------
# P9-05 — live monitor log
# --------------------------------------------------------------------------
def monitor_log() -> None:
    df = data.alerts_with_outcomes()
    if df.empty:
        st.info("The live alert log is empty.", icon="🗓️")
        return

    resolved, filed, rate = data.hit_rate(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alerts logged", f"{len(df):,}")
    c2.metric("Windows closed", f"{resolved:,}")
    c3.metric("Followed by an 8-K", f"{filed:,}")
    c4.metric("Hit rate", f"{rate:.1%}" if rate is not None else "—")

    st.markdown(ui.honest_rate(resolved, filed, rate))
    st.info(
        "**Append-only, and checkably so.** Every row carries a hash of itself "
        "and of the row before it, so the log cannot be edited after the fact "
        "without breaking the chain. Verify with "
        "`python -m src.live.alertlog --verify`. The monitor re-scores a "
        "rolling 48-bar window each run and suppresses re-detections by "
        "natural key, so a repeated scan writes nothing.", icon="🔗")

    show = df.copy()
    show["flagged"] = show["ts_utc"].map(lambda t: ui.utc(t, False))
    show["noticed"] = show["raised_utc"].map(lambda t: ui.utc(t, False))
    show["outcome"] = show["filed"].map(
        {1.0: "8-K followed", 0.0: "no 8-K in window"}).fillna("window open")
    cols = ["flagged", "noticed", "ticker", "detector", "score", "threshold",
            "outcome", "lead_trading_h"]
    st.dataframe(show[[c for c in cols if c in show.columns]],
                 width='stretch', hide_index=True)
