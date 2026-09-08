"""The four dashboard screens (P9-02 … P9-05).

Each is a plain function taking no arguments, rendering into the current page.
`dashboard.py` owns routing, the masthead and the disclaimer, so no screen can
render without them.

The screens differ in audience and are laid out accordingly. *Today's alerts*
and *Ticker detail* are worked by an analyst deciding in about thirty seconds
whether something deserves a closer look, so they lead with the alert and put
the reasoning beside it. *Evaluation* is read by an examiner, so it leads with
the comparison and states the measurement rules on the page rather than
assuming them.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data, ui

HOUR = 3600
DAY = 86400

#: How each feature is put into words, with the comparison that makes it mean
#: something: "4.2x its own normal", never a bare "4.2".
_REASON = {
    "volume_z": lambda v: f"vol {v:+.1f} sd vs normal",
    "ret_rel_4h": lambda v: f"{v:+.1%} vs SPY 4h",
    "ret_rel_24h": lambda v: f"{v:+.1%} vs SPY 24h",
    "ret_4h": lambda v: f"{v:+.1%} 4h",
    "ret_24h": lambda v: f"{v:+.1%} 24h",
    "ret_120h": lambda v: f"{v:+.1%} 120h",
    "volatility": lambda v: f"vol'y {v:.2%}/h",
    "days_since_last_8k": lambda v: f"8-K {v:.0f}d ago",
    "days_since_last_earnings": lambda v: f"results {v:.0f}d ago",
    "hours_since_news": lambda v: f"news {v / 24:.1f}d ago",
    "news_count_24h": lambda v: f"{v:.0f} articles 24h",
    "trading_hours_to_close": lambda v: f"{v:.1f}h to close",
}

#: Reading order for a triage analyst: the volume anomaly first, then whether
#: the move was market-wide, then how quiet the company had been.
_ORDER = ["volume_z", "ret_rel_4h", "ret_rel_24h", "ret_4h", "ret_24h",
          "days_since_last_8k", "hours_since_news", "news_count_24h",
          "volatility", "trading_hours_to_close"]


def _reasons(row: pd.Series, limit: int = 4) -> list[str]:
    """The features that fired this alert, in words with their units.

    Rule 1: reasons travel WITH the alert, in the row, never behind a click. A
    number with no reason attached is a black box, and the point of the screen
    is that a human can sanity-check it in half a minute.
    """
    out = []
    for key in _ORDER:
        if key in row.index and pd.notna(row.get(key)):
            out.append(_REASON[key](row[key]))
        if len(out) >= limit:
            break
    return out or ["no features recorded for this alert"]


def _outcome(row: pd.Series) -> tuple[str, str]:
    filed = row.get("filed")
    if pd.isna(filed):
        return "open", "window still open"
    return ("filed", "8-K followed") if filed == 1 else ("none", "no 8-K in window")


# --------------------------------------------------------------------------
# P9-02 — today's alerts
# --------------------------------------------------------------------------
def alerts_today() -> None:
    df = data.alerts_with_outcomes()
    if df.empty:
        ui.section("No alerts yet")
        st.info("The alert log is empty. The system flags roughly **2 per stock "
                "per month by design**, so an empty queue is a normal state "
                "rather than a failure.")
        return

    newest = int(df["ts_utc"].max())
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        scope = st.radio("Window", ["Latest session", "Last 7 days", "All"],
                         horizontal=True, label_visibility="collapsed")
    which = c2.selectbox("Detector",
                         ["All detectors"] + sorted(df["detector"].unique()),
                         label_visibility="collapsed")
    state = c3.selectbox("Outcome",
                         ["All outcomes", "8-K followed", "No 8-K", "Window open"],
                         label_visibility="collapsed")

    cutoff = {"Latest session": newest - (newest % DAY),
              "Last 7 days": newest - 7 * DAY, "All": 0}[scope]
    view = df[df["ts_utc"] >= cutoff]
    if which != "All detectors":
        view = view[view["detector"] == which]

    resolved, filed, rate = data.hit_rate(view)
    st.caption(ui.honest_rate(resolved, filed, rate))

    if view.empty:
        st.info("No alerts in this window. The system flags roughly 2 per stock "
                "per month by design.")
        return

    rows = []
    for _, r in view.iterrows():
        key, words, mult = ui.strength(r["score"], r["threshold"])
        outcome = _outcome(r)[1]
        rows.append({
            "Ticker": r["ticker"],
            "Strength": words,
            "× thresh": round(mult, 1),
            "Detector": r["detector"],
            "Bar (UTC)": ui.short_utc(r["ts_utc"]),
            "Outcome": outcome,
            # Rule 1: the reasons travel WITH the alert, in the row, never
            # behind a click. A number with no reason attached is a black box.
            "Why it fired": " · ".join(_reasons(r)),
        })
    table = pd.DataFrame(rows)
    if state != "All outcomes":
        want = {"8-K followed": "8-K followed", "No 8-K": "no 8-K in window",
                "Window open": "window still open"}[state]
        table = table[table["Outcome"] == want]
    table = table.sort_values("× thresh", ascending=False)

    ui.section(
        f"{len(table):,} alerts",
        "Strongest first — a work queue, not an index. Sort any column by "
        "clicking it. Every row carries the features that triggered it, "
        "because a score with no reason beside it is a black box.")
    st.dataframe(
        table, width="stretch", hide_index=True, height=520,
        column_config={
            "× thresh": st.column_config.NumberColumn(
                "× thresh", format="%.1f×", width="small",
                help="How far above its own alert threshold this score sat. "
                     "NOT a probability — these detectors emit raw statistics."),
            "Why it fired": st.column_config.TextColumn("Why it fired", width="large"),
            "Ticker": st.column_config.TextColumn(width="small"),
            "Strength": st.column_config.TextColumn(width="small"),
        })
    st.caption(
        "**Strength bands come from the observed distribution**, not round "
        "numbers: the median alert sits at 1.6× its threshold and the 90th "
        "percentile at 4.3×. Extreme ≥10×, Strong ≥4×, Elevated ≥2×.")


# --------------------------------------------------------------------------
# P9-03 — ticker detail
# --------------------------------------------------------------------------
def ticker_detail() -> None:
    df = data.alerts_with_outcomes()
    if df.empty:
        ui.note("No alerts to inspect yet.")
        return

    c1, c2 = st.columns([1, 2])
    ticker = c1.selectbox("Ticker", sorted(df["ticker"].unique()))
    rows = df[df["ticker"] == ticker].sort_values("ts_utc", ascending=False)
    label = {int(r.ts_utc): f"{ui.utc(r.ts_utc, False)} · {r.detector}"
             for r in rows.itertuples()}
    flagged = c2.selectbox("Flagged hour", list(label), format_func=label.get)
    row = rows[rows["ts_utc"] == flagged].iloc[0]

    key, words, mult = ui.strength(row["score"], row["threshold"])
    state, _ = _outcome(row)
    s = st.columns(4)
    ui.stat(s[0], "Ticker", ticker, row["detector"])
    ui.stat(s[1], "Strength", f"{mult:.1f}×", f"{words} — {mult:.1f}× threshold")
    ui.stat(s[2], "Flagged", dt.datetime.fromtimestamp(
        int(flagged), dt.timezone.utc).strftime("%d %b %H:%M"), ui.utc(flagged))
    ui.stat(s[3], "Outcome", {"filed": "8-K followed", "none": "No 8-K",
                              "open": "Pending"}[state],
            "within 48 trading hours")

    live = state == "open"
    if live:
        ui.note("This alert's window is still open, so <b>nothing after the "
                "flagged hour is shown</b>. Revealing what happened next would "
                "turn a surveillance tool into a hindsight demo.")

    hi = int(flagged) if live else int(flagged) + 48 * HOUR
    lo = int(flagged) - 30 * DAY
    price = data.bars(ticker, lo, hi)

    if price.empty:
        ui.note("No price bars stored for this window.")
    else:
        ts = pd.to_datetime(price["ts_utc"], unit="s", utc=True)
        marker = dt.datetime.fromtimestamp(int(flagged), dt.timezone.utc)
        sev = ui.MARKER

        ui.section("Price and volume",
                   "Hourly bars for the 30 days before the flag. The dashed "
                   "line is the flagged hour.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=price["close"], mode="lines",
                                 name="close", line=dict(color=ui.SERIES, width=1.5)))
        fig.add_vline(x=marker, line_dash="dash", line_color=sev, line_width=1.4)
        st.plotly_chart(ui.chart(fig, 230, "close"), width="stretch")

        vol = go.Figure()
        vol.add_trace(go.Bar(x=ts, y=price["volume"], marker_color=ui.DIM))
        vol.add_vline(x=marker, line_dash="dash", line_color=sev, line_width=1.4)
        st.plotly_chart(ui.chart(vol, 150, "volume"), width="stretch")

        # The z-score band, computed with the SAME function the detector used.
        # A lookalike written here could drift from it and would then explain
        # the wrong thing convincingly.
        from src.pipeline.features import volume_zscore

        frame = price.set_index("ts_utc")[["close", "volume"]]
        z = volume_zscore(frame, data.config())["volume_z"]
        if z.notna().any():
            ui.section("Volume z-score",
                       "How unusual each hour's volume is against this stock's "
                       "own trailing normal. The dotted line is the alert "
                       "threshold.")
            band = go.Figure()
            band.add_trace(go.Scatter(x=ts, y=z.to_numpy(), mode="lines",
                                      line=dict(color=ui.SERIES, width=1.5)))
            if pd.notna(row.get("threshold")) and row["detector"] == "volume_zscore":
                band.add_hline(y=float(row["threshold"]), line_dash="dot",
                               line_color=sev, line_width=1.2)
            band.add_vline(x=marker, line_dash="dash", line_color=sev, line_width=1.4)
            st.plotly_chart(ui.chart(band, 165, "standard deviations"),
                            width="stretch")

    ui.section("Why this hour was flagged",
               "Each value beside the same measure's trailing normal for this "
               "ticker over the 30 days before the flag. A number alone means "
               "little — the comparison is what makes it mean something.")
    _feature_table(row, price)

    a, b = st.columns(2)
    with a:
        ui.section("News", "Headlines as published, quoted verbatim with their "
                           "publisher. Some carry analyst language — that is "
                           "the outlet's wording, not this tool's.")
        n = data.news(ticker, lo, hi)
        if n.empty:
            st.caption("No articles in this window. A quiet stretch before a "
                       "move is the interesting shape, not a gap in the data — "
                       "every week of the study window was fetched for every "
                       "in-universe ticker.")
        else:
            for _, art in n.head(10).iterrows():
                st.markdown(
                    f'<div style="margin-bottom:.5rem"><span style="opacity:.65;font-size:.8rem">'
                    f'{ui.utc(art["published_utc"], False)}</span><br>'
                    f'<span style="font-size:.85rem;color:inherit">'
                    f'{art["title"]}</span> '
                    f'<span style="opacity:.65;font-size:.8rem">*{art["source_name"]}*</span></div>',
                    unsafe_allow_html=True)
    with b:
        ui.section("Filing history", "Past 8-K filings with their item codes.")
        f = data.filings(ticker)
        if f.empty:
            st.caption("No 8-K filings on record.")
        else:
            st.dataframe(pd.DataFrame({
                "accepted (UTC)": f["acceptance_utc"].map(lambda t: ui.utc(t, False)),
                "items": f["items"].fillna("—"),
            }), width="stretch", hide_index=True, height=320)


def _feature_table(row: pd.Series, price: pd.DataFrame) -> None:
    """Feature values beside the same measure's trailing normal (P9-03).

    The trailing normal is recomputed from this ticker's own bars over the 30
    days STRICTLY BEFORE the flagged hour — the whole series would put the
    spike inside the baseline it is being judged against, which is the same
    reason `volume_zscore` carries its own `shift(1)`.
    """
    from src.pipeline.features import returns, volume_zscore

    rows = []
    if not price.empty:
        frame = price.set_index("ts_utc")[["close", "volume"]]
        cfg = data.config()
        hist = pd.concat([returns(frame, cfg), volume_zscore(frame, cfg)], axis=1)
        hist = hist[hist.index < int(row["ts_utc"])]

        for col in ("volume_z", "ret_1h", "ret_4h", "ret_24h", "ret_120h"):
            if col not in row.index or pd.isna(row.get(col)) or col not in hist:
                continue
            past = hist[col].dropna()
            if past.empty:
                continue
            fmt = ((lambda v: f"{v:+.2f} sd") if col == "volume_z"
                   else (lambda v: f"{v:+.2%}"))
            rows.append({
                "feature": col,
                "at the flagged hour": fmt(row[col]),
                "trailing median": fmt(past.median()),
                "trailing 5–95%": f"{fmt(past.quantile(.05))} … {fmt(past.quantile(.95))}",
                "percentile": f"{(past < row[col]).mean() * 100:.0f}th",
            })

    # Context features have no price-derived trailing normal. They are shown
    # as-is rather than given a fabricated comparison.
    for col in ("days_since_last_8k", "hours_since_news", "news_count_24h"):
        if col in row.index and pd.notna(row.get(col)):
            rows.append({"feature": col,
                         "at the flagged hour": _REASON[col](row[col]),
                         "trailing median": "—", "trailing 5–95%": "—",
                         "percentile": "—"})

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.caption("No features recorded for this alert.")


# --------------------------------------------------------------------------
# P9-04 — evaluation
# --------------------------------------------------------------------------
def evaluation() -> None:
    table = data.comparison("phase10/FINAL-test-evaluation.csv")
    final = not table.empty
    if not final:
        table = data.comparison("p8-without-news-val.csv")
    if table.empty:
        table = data.comparison("baseline-comparison-val.csv")
    if table.empty:
        ui.note("No comparison table built yet. Run "
                "<code>python -m src.baselines.compare --split val --out …</code>")
        return

    ui.note(
        "<b>Plain accuracy is not reported anywhere, by design.</b> Only ~0.285% "
        "of hours precede an event, so a system that always says \"nothing is "
        "coming\" is <b>99.71%</b> accurate and useless. The function that would "
        "compute it raises an error instead. The headline is precision at the "
        "fixed alert budget.")
    ui.note(
        "<b>Scheduled and unscheduled are never pooled.</b> Scheduled events — "
        "results announcements — have dates published weeks ahead, so a run-up "
        "before one is far less interesting. Unscheduled events are the real "
        "target, and pooling would let the easy half carry the number.")

    if final:
        st.caption("Source: **Phase 10 final evaluation on the sealed test "
                   "set**, run once on 2026-09-08.")

    c1, c2 = st.columns([1, 2])
    variants = sorted(table["t0_variant"].unique())
    variant = c1.selectbox(
        "t₀ variant", variants,
        index=variants.index("news_adjusted") if "news_adjusted" in variants else 0,
        help="`filing` uses the 8-K acceptance time. `news_adjusted` takes the "
             "earlier of that and the first news article — the honest clock, "
             "and the project's main contribution.")
    with c2:
        sl = st.radio("Slice", ["all", "scheduled", "unscheduled"],
                      horizontal=True)

    view = table[(table["t0_variant"] == variant) & (table["slice"] == sl)]
    view = view.sort_values("precision", ascending=False)

    ui.section("Detector comparison",
               "Precision at the fixed alert budget. `max_precision` is the "
               "ceiling: the budget is SPENT, not capped, so when it exceeds "
               "the number of events even a flawless detector cannot reach 1.0.")
    show = pd.DataFrame({
        "detector": view["baseline"],
        "precision": view["precision"].map(lambda v: ui.pct(v, 3)),
        "ceiling": view["max_precision"].map(lambda v: ui.pct(v, 2)),
        "lift vs floor": view["lift"].map(lambda v: f"{v:.1f}×"),
        "recall": view["recall"].map(lambda v: ui.pct(v, 1)),
        "median lead": view["median_lead_trading_h"].map(
            lambda v: "—" if pd.isna(v) else f"{v:.1f} h"),
        "alerts": view["n_alerts"].map(ui.num),
    })
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption("**If a simple baseline wins, it is shown winning** — that is "
               "the finding, not something to hide.")

    ui.section("Calibration",
               "When a detector says 70%, is it right about 70% of the time? A "
               "detector can rank well and still be badly calibrated, which "
               "matters when a human decides what to act on.")
    st.dataframe(view[["baseline", "brier", "brier_skill_score", "ece"]],
                 width="stretch", hide_index=True)
    st.caption(
        "**Blank is the honest entry, not a gap.** CUSUM and the volume z-score "
        "emit scores that are **not probabilities** — the evaluation contract "
        "says so — and scoring them with Brier or ECE would invent a "
        "calibration they never claimed. A negative skill score means the "
        "probabilities are worse than always predicting the base rate: these "
        "models rank far better than they calibrate.")

    ui.section("Action distribution",
               "WAIT versus FLAG. A detector that cannot detect shows as "
               "`degenerate` — either it never flags, or its scores are all "
               "one value and so cannot rank.")
    st.dataframe(view[["baseline", "n_wait_hours", "n_flag_hours",
                       "pct_hours_flagged", "pct_windows_alerted", "degenerate"]],
                 width="stretch", hide_index=True)
    st.caption("`pct_hours_flagged` never exceeds ~2% even for a busy detector, "
               "because at most one FLAG is allowed per 48-hour window; "
               "`pct_windows_alerted` is the interpretable one.")

    with_news = data.comparison("p8-with-news-val.csv")
    without = data.comparison("p8-without-news-val.csv")
    if not with_news.empty and not without.empty:
        ui.section("Phase 8 — does the news channel help?",
                   "Gradient boosting only: it is the sole baseline reading "
                   "more than one column, so it is the only one that can carry "
                   "this comparison. Validation, not test.")
        a = without[(without.t0_variant == variant)
                    & (without.baseline == "gradient_boosting")]
        b = with_news[(with_news.t0_variant == variant)
                      & (with_news.baseline == "gradient_boosting")]
        m = a.merge(b, on="slice", suffixes=("_without", "_with"))
        m = m[m["slice"].isin(["all", "scheduled", "unscheduled"])]
        st.dataframe(pd.DataFrame({
            "slice": m["slice"],
            "without news": m["precision_without"].map(lambda v: ui.pct(v, 3)),
            "with news": m["precision_with"].map(lambda v: ui.pct(v, 3)),
            "change": ((m.precision_with / m.precision_without - 1)
                       .map(lambda v: f"{v * 100:+.1f}%")),
            "lead without": m["median_lead_trading_h_without"].map(lambda v: f"{v:.1f} h"),
            "lead with": m["median_lead_trading_h_with"].map(lambda v: f"{v:.1f} h"),
            "lead change": ((m.median_lead_trading_h_with
                             - m.median_lead_trading_h_without)
                            .map(lambda v: f"{v:+.1f} h")),
        }), width="stretch", hide_index=True)
        st.caption("Lead time falls where precision rises: press coverage "
                   "accumulates close to the event, so it buys confidence at "
                   "the cost of warning. The delta is reported with its sign "
                   "either way.")


# --------------------------------------------------------------------------
# P9-05 — live monitor log
# --------------------------------------------------------------------------
def monitor_log() -> None:
    df = data.alerts_with_outcomes()
    if df.empty:
        ui.note("The live alert log is empty.")
        return

    resolved, filed, rate = data.hit_rate(df)
    c = st.columns(4)
    ui.stat(c[0], "Alerts logged", ui.num(len(df)), "append-only, hash-chained")
    ui.stat(c[1], "Windows closed", ui.num(resolved), "48 trading hours elapsed")
    ui.stat(c[2], "Followed by an 8-K", ui.num(filed), "within the window")
    ui.stat(c[3], "Hit rate", ui.pct(rate, 1) if rate is not None else "—",
            "of closed windows only")

    ui.note(ui.honest_rate(resolved, filed, rate))
    ui.note(
        "<b>Append-only, and checkably so.</b> Every row carries a hash of "
        "itself and of the row before it, so the log cannot be edited after the "
        "fact without breaking the chain — verify with "
        "<code>python -m src.live.alertlog --verify</code>. The monitor "
        "re-scores a rolling 48-bar window each run and suppresses "
        "re-detections by natural key, so a repeated scan writes nothing.")

    ui.section("The log", "Bar time and notice time are separate columns on "
                          "purpose: the monitor runs once a day after the "
                          "close, so an alert is noticed later than the hour it "
                          "describes.")
    show = pd.DataFrame({
        "bar (UTC)": df["ts_utc"].map(lambda t: ui.utc(t, False)),
        "noticed (UTC)": df["raised_utc"].map(lambda t: ui.utc(t, False)),
        "ticker": df["ticker"],
        "detector": df["detector"],
        "score": df["score"].map(lambda v: f"{v:.3f}"),
        "threshold": df["threshold"].map(lambda v: f"{v:.3f}"),
        "outcome": df["filed"].map({1.0: "8-K followed", 0.0: "no 8-K"})
                              .fillna("window open"),
        "lead (trading h)": df.get("lead_trading_h", pd.Series(index=df.index))
                              .map(lambda v: "—" if pd.isna(v) else f"{v:.1f}"),
    })
    st.dataframe(show, width="stretch", hide_index=True, height=460)
