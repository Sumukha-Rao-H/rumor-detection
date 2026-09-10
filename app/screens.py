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

#: Reading order for a triage analyst, as SLOTS rather than a flat list: one
#: reason per idea, first available alternative wins.
#:
#: `UI-context.md` rule 1 names four things that must be in the row — volume
#: multiple, benchmark-relative move, hours since the last 8-K, news coverage —
#: and a flat list could not deliver them. It ran volume_z, ret_rel_4h,
#: ret_rel_24h, ret_4h and stopped at four, so on the 513 alerts that have a
#: benchmark-relative move the four slots went to volume plus three restatements
#: of the same move, and 8-K recency never appeared at all. Grouping the returns
#: into one "the move" slot spends each of the four on a different idea.
_SLOTS = (
    ("volume_z",),
    # The move, benchmark-relative where we have it. `ret_rel_*` is missing on
    # 1,520 of the 2,033 logged alerts — bars scored past the benchmark's own
    # newest bar — so the raw return is the fallback rather than a second slot.
    ("ret_rel_4h", "ret_rel_24h", "ret_4h", "ret_24h"),
    ("days_since_last_8k",),
    ("hours_since_news", "news_count_24h"),
    # Beyond the four rule 1 requires, for the expanded view.
    ("ret_120h",),
    ("volatility",),
    ("trading_hours_to_close",),
    ("days_since_last_earnings",),
)


def _reasons(row: pd.Series, limit: int = 4) -> list[str]:
    """The features that fired this alert, in words with their units.

    Rule 1: reasons travel WITH the alert, in the row, never behind a click. A
    number with no reason attached is a black box, and the point of the screen
    is that a human can sanity-check it in half a minute.
    """
    out = []
    for slot in _SLOTS:
        for key in slot:
            if key in row.index and pd.notna(row.get(key)):
                out.append(_REASON[key](row[key]))
                break                      # one reason per idea, not three
        if len(out) >= limit:
            break
    return out or ["no features recorded for this alert"]


def _outcome(row: pd.Series) -> tuple[str, str]:
    """(state, words) for one alert — four states, not three.

    The state is read from the column `data.alerts_with_outcomes` derives from
    the alert's own timestamp against the answerable edge. It used to be read
    from the join alone, which conflated "the window has not closed" with "this
    database holds no outcome row" and so labelled 1,236 alerts pending whose
    windows had closed weeks earlier.
    """
    state = row.get("outcome_state")
    if not isinstance(state, str) or state not in data.OUTCOME_WORDS:
        state = "unscored"
    return state, data.OUTCOME_WORDS[state]


# --------------------------------------------------------------------------
# P9-02 — today's alerts
# --------------------------------------------------------------------------
def _queue_stats(view: pd.DataFrame, all_rows: pd.DataFrame) -> None:
    """The four numbers a triage analyst needs before anything else.

    Eye-tracking work on dashboards is consistent that the top-left carries
    most of the attention and that four to six figures above the fold is the
    limit before they stop being read. So this row answers "what is in my
    queue right now", not "how is the system configured" — the alert budget
    and universe size are real (rule 6) but they are context for a result, not
    the first question a queue poses. They now sit in the strip below.
    """
    split = data.split_hit_rates(view)
    resolved = split["resolved"]
    hours = data.window_hours()
    strongest = view.apply(
        lambda r: ui.strength(r["score"], r["threshold"])[2], axis=1).max() \
        if not view.empty else float("nan")
    state = (view["outcome_state"] if "outcome_state" in view
             else pd.Series("unscored", index=view.index))
    open_n = int((state == "open").sum())
    unscored_n = int((state == "unscored").sum())

    c = st.columns(5)
    # No `delta` here: Streamlit renders one with a directional arrow, and an
    # arrow beside "of 2,033 logged" reads as a trend when it is a denominator.
    c[0].metric(f"In view · of {ui.num(len(all_rows))}", ui.num(len(view)),
                help="Alerts matching the filters above, out of every alert "
                     "ever logged.")
    c[1].metric("Awaiting outcome", ui.num(open_n),
                help=f"The {hours}-hour window (wall-clock) has not closed yet, "
                     f"so these cannot be graded. They are excluded from the "
                     f"hit rate rather than counted as misses.")
    c[2].metric("Not scored", ui.num(unscored_n),
                help="The window closed, but this database holds no outcome "
                     "row for the alert. Not a miss and not pending — an "
                     "answer nobody has looked up. Kept as its own count so "
                     "the hit rate's denominator is not mistaken for the set "
                     "of alerts that could be graded.")
    c[3].metric("Strongest", f"{strongest:.1f}×" if pd.notna(strongest) else "—",
                help="Highest multiple of its own alert threshold in this view. "
                     "NOT a probability.")
    # Rule 7: unscheduled is the headline, never the pooled figure. Results
    # announcements are published weeks in advance and would carry the number.
    c[4].metric("Unscheduled hit rate",
                ui.pct(split["unscheduled_rate"], 1)
                if split["unscheduled_rate"] is not None else "—",
                help=(f"{split['unscheduled']} of {resolved} graded alerts were "
                      f"followed by an UNSCHEDULED 8-K within {hours} hours. A "
                      f"further {split['scheduled']} were followed by a "
                      f"scheduled one (a results announcement); pooled that is "
                      f"{ui.pct(split['pooled'], 1)}, which this project does "
                      f"not report as one number." if resolved else
                      "No alert in this view has both a closed window and an "
                      "outcome on record, so there is no rate to quote. An "
                      "empty figure is reported rather than a flattering one."))


def _volume_trend(df: pd.DataFrame) -> None:
    """Alert volume over time — named in the triage literature as a core metric.

    It is the fastest way to see the thing a count cannot show: whether today
    is unusual, and whether a step in the series is the market or a change we
    made. This log has one such step by construction — coverage widened from
    400 tickers to 1,500 on 2026-09-07 — and a reader who cannot see it would
    mistake it for a signal.
    """
    if df.empty:
        return
    day = pd.to_datetime(df["ts_utc"], unit="s", utc=True).dt.floor("D")
    counts = day.value_counts().sort_index()
    if len(counts) < 2:
        return
    fig = go.Figure(go.Bar(x=counts.index, y=counts.to_numpy(),
                           marker_color=ui.SERIES))
    st.plotly_chart(ui.chart(fig, 120, "alerts"), width="stretch")


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
                         ["All outcomes", "8-K followed", "No 8-K",
                          "Window open", "Not scored"],
                         label_visibility="collapsed")

    cutoff = {"Latest session": newest - (newest % DAY),
              "Last 7 days": newest - 7 * DAY, "All": 0}[scope]
    view = df[df["ts_utc"] >= cutoff]
    if which != "All detectors":
        view = view[view["detector"] == which]
    if state != "All outcomes":
        # Filter on the derived state, not on `filed`: "window open" and "not
        # scored" both show a missing `filed` and are different answers.
        want = {"8-K followed": "filed", "No 8-K": "none",
                "Window open": "open", "Not scored": "unscored"}[state]
        view = view[view["outcome_state"] == want]

    _queue_stats(view, df)
    if view.empty:
        st.info("No alerts match these filters. The system flags roughly 2 per "
                "stock per month by design.")
        return

    # Sort ONCE, on the source frame, so a selected table row maps back to its
    # alert by position. Sorting the rendered table separately would silently
    # open the wrong alert the moment the two orders diverged.
    view = view.assign(_mult=view.apply(
        lambda r: ui.strength(r["score"], r["threshold"])[2], axis=1)
    ).sort_values("_mult", ascending=False).reset_index(drop=True)

    table = pd.DataFrame([{
        "Ticker": r["ticker"],
        "Strength": ui.strength(r["score"], r["threshold"])[1],
        "× thresh": round(r["_mult"], 1),
        "Detector": r["detector"],
        "Bar (UTC)": ui.short_utc(r["ts_utc"]),
        "Outcome": _outcome(r)[1],
        # Rule 1: the reasons travel WITH the alert, never behind a click.
        "Why it fired": " · ".join(_reasons(r)),
    } for _, r in view.iterrows()])

    ui.section(f"{len(table):,} alerts",
               "Strongest first — a work queue, not an index. Click any column "
               "to sort, or a row to open it below.")
    picked = st.dataframe(
        table, width="stretch", hide_index=True, height=430,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "× thresh": st.column_config.NumberColumn(
                "× thresh", format="%.1f×", width="small",
                help="How far above its own threshold this score sat. NOT a "
                     "probability — these detectors emit raw statistics."),
            "Why it fired": st.column_config.TextColumn(width="large"),
            "Ticker": st.column_config.TextColumn(width="small"),
            "Strength": st.column_config.TextColumn(width="small"),
        })

    # Master-detail: the row is the summary, selecting it reveals the depth,
    # without losing the queue position that a page change would cost.
    sel = picked.selection.rows if picked and picked.selection else []
    if sel and sel[0] < len(view):
        _alert_detail(view.iloc[sel[0]])
    else:
        st.caption("**Strength bands come from the observed distribution**, not "
                   "round numbers: the median alert sits at 1.6× its threshold "
                   "and the 90th percentile at 4.3×. Extreme ≥10×, Strong ≥4×, "
                   "Elevated ≥2×. Select a row above to open it.")

    # Rule 1 names four reasons. Two of them are not always available, and
    # saying so once is better than leaving a reader to wonder which alerts are
    # missing an explanation and why.
    st.caption(
        "**Why some rows show fewer than four reasons.** Rule 1 asks for four: "
        "volume multiple, benchmark-relative move, hours since the last 8-K, "
        "and news coverage. **News coverage is absent from every live alert** — "
        "`features.include_news_coverage` is off in the config, because whether "
        "the news channel helps is the Phase 8 experiment and the live monitor "
        "runs the same arm every earlier phase ran. And the benchmark-relative "
        "move is recorded on 513 of the 2,033 logged alerts; the rest were "
        "scored on bars past the benchmark's own newest bar, and show the raw "
        "return instead. Neither gap is filled with a fabricated value.")

    with st.expander("Alert volume over time — is today unusual?"):
        _volume_trend(df)
        st.caption("One step in this series is ours, not the market's: coverage "
                   "widened from 400 tickers to the full 1,500 on 2026-09-07, "
                   "so alerts per day rises there by construction.")


def _alert_detail(r: pd.Series) -> None:
    """One alert opened in place — the master-detail half of the queue."""
    _, words, mult = ui.strength(r["score"], r["threshold"])
    with st.container(border=True):
        a, b = st.columns([1, 3])
        a.metric(r["ticker"], f"{mult:.1f}×", delta=words, delta_color="off")
        a.caption(f"{r['detector']} · {_outcome(r)[1]}")
        b.markdown(f"**Bar** {ui.utc(r['ts_utc'])}  \n"
                   f"**Noticed** {ui.utc(r['raised_utc'], False)}  \n"
                   f"**Score** {r['score']:.4f} against a threshold of "
                   f"{r['threshold']:.4f}")
        b.markdown("**Why it fired** — " + " · ".join(_reasons(r, limit=8)))
        st.caption("Open **Ticker detail** for the price chart, the volume "
                   "z-score band and this company's filing history.")


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
    hours = data.window_hours()
    ui.stat(s[3], "Outcome", {"filed": "8-K followed", "none": "No 8-K",
                              "open": "Pending", "unscored": "Not scored"}[state],
            # Wall-clock, not trading hours. `outcome_window_hours` is elapsed
            # time — a company can file overnight or at a weekend — and calling
            # it trading hours would stretch two days into about seven.
            f"within {hours} hours (wall-clock)")

    live = state == "open"
    if live:
        ui.note("This alert's window is still open, so **nothing after the "
                "flagged hour is shown**. Revealing what happened next would "
                "turn a surveillance tool into a hindsight demo.")

    hi = int(flagged) if live else int(flagged) + hours * HOUR
    lo = int(flagged) - 30 * DAY
    price = data.bars(ticker, lo, hi)

    if not data.db_present():
        # This screen is the one that genuinely needs the database: bars, news
        # and filing history all live there and none of them are in the
        # committed CSV. Say so plainly instead of rendering three empty panels.
        ui.note("**No local database, so the evidence panels below are empty.** "
                "The alert and its recorded features come from the committed "
                "log, but the price bars, the news timeline and the filing "
                "history are read from `paths.db`, which is a rebuildable cache "
                "and is not committed. Rebuild it with the Phase 2 and 3 "
                "collectors, or restore the bootstrap snapshot.")
    elif price.empty:
        ui.note("No price bars stored for this window.")

    if not price.empty:
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
#: The Phase 10 artifacts, newest first. The corrected file is preferred and
#: the original is the fallback, never the other way round: it fixes three
#: defects derivable from the run's own stored counts — a `max_precision`
#: ceiling ten rows beat, `degenerate = False` on all 42 `always_quiet` rows,
#: and 36 rows for item codes that `config.items.exclude` drops. Showing the
#: superseded file made the screen contradict itself, captioning "a detector
#: that cannot detect shows as `degenerate`" above a table of `False`.
_PHASE10 = (
    ("phase10/FINAL-test-evaluation-corrected.csv",
     "**Phase 10 final evaluation on the sealed test set**, run once on "
     "2026-09-08, with the arithmetic corrections of 2026-09-09 applied "
     "(`FINAL-test-evaluation-corrected.csv` — see `CORRECTION-NOTE.md` beside "
     "it). No model was re-scored and the sealed test set was not touched "
     "again."),
    ("phase10/FINAL-test-evaluation.csv",
     "**Phase 10 final evaluation on the sealed test set**, run once on "
     "2026-09-08 — the ORIGINAL file. The corrected artifact is not present, "
     "so three known defects stand in the numbers below: the `max_precision` "
     "ceiling is measured on the allowance rather than the alerts issued, "
     "`always_quiet` reads `degenerate = False` when it emits one constant "
     "score, and excluded item codes 9.01 and 5.07 still have rows."),
)

#: Kept as a standing caption rather than a footnote, because it disqualifies a
#: column that is on screen. Quoted from `CORRECTION-NOTE.md`.
_LIFT_CAVEAT = (
    "**The `lift vs floor` column predates the evaluation-frame fix and is "
    "pending recomputation.** Both Phase 10 files were produced under a frame "
    "that rewarded window length rather than detection: a positive episode got "
    "48 bars and a quiet window got one, while a window scores as the maximum "
    "over its rows. Measured on that exact shape, **a scorer made of pure "
    "random noise reaches precision 0.0943 and 29.6× lift, beating every "
    "detector in the table**; with both classes the same length the same noise "
    "scores 1.0×. The metric is fixed in commit `61083d4`, which also adds a "
    "permanent `random_noise` row so the null is visible in every future table, "
    "but no phase has been re-run. The figures are kept and labelled rather "
    "than deleted — precision, recall and the counts are unaffected."
)


def evaluation() -> None:
    table = pd.DataFrame()
    source = ""
    for name, caption in _PHASE10:
        table = data.comparison(name)
        if not table.empty:
            source = caption
            break
    if table.empty:
        table = data.comparison("p8-without-news-val.csv")
    if table.empty:
        table = data.comparison("baseline-comparison-val.csv")
    if table.empty:
        ui.note("No comparison table built yet. Run "
                "`python -m src.baselines.compare --split val --out …`")
        return

    ui.note(
        "**Plain accuracy is not reported anywhere, by design.** Only ~0.285% "
        "of hours precede an event, so a system that always says \"nothing is "
        "coming\" is **99.71%** accurate and useless. The function that would "
        "compute it raises an error instead. The headline is precision at the "
        "fixed alert budget.")
    ui.note(
        "**Scheduled and unscheduled are never pooled.** Scheduled events — "
        "results announcements — have dates published weeks ahead, so a run-up "
        "before one is far less interesting. Unscheduled events are the real "
        "target, and pooling would let the easy half carry the number.")
    st.warning(_LIFT_CAVEAT)

    if source:
        st.caption(f"Source: {source}")

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

    # State the outcome rather than leaving a reader to rank nine rows by eye.
    # This is the project's actual finding and the plan committed to reporting
    # it either way: "if the simple threshold wins, that is a finding".
    if len(view):
        top = view.iloc[0]
        floor = view[view.baseline == "always_quiet"]["precision"]
        lift = (top["precision"] / floor.iloc[0]) if len(floor) and floor.iloc[0] else None
        st.success(
            f"**{top['baseline']}** leads this slice at "
            f"**{ui.pct(top['precision'], 3)}** precision"
            # The lift is quoted with its disqualifier attached, not silently.
            # A bare "22.1× the do-nothing floor" reads as the finding, and
            # pure noise scores 29.6× on this evaluation frame.
            + (f", {lift:.1f}× the do-nothing floor — a figure the caveat above "
               f"disqualifies until the evaluation frame is recomputed, since "
               f"pure noise reaches 29.6× on this frame" if lift else "")
            + f", against a ceiling of {ui.pct(top['max_precision'], 2)}. "
            f"**If a simple baseline wins, it is shown winning** — that is the "
            f"finding, not something to hide.")

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

    hours = data.window_hours()
    split = data.split_hit_rates(df)
    cov = data.coverage()

    c = st.columns(4)
    ui.stat(c[0], "Alerts logged", ui.num(cov["logged"]),
            "append-only, hash-chained, committed to git after every run")
    ui.stat(c[1], "Windows closed", ui.num(cov["answerable"]),
            f"{hours} hours (wall-clock) elapsed, so an answer exists")
    ui.stat(c[2], "Graded here", ui.num(cov["graded"]),
            "closed AND with an outcome row in this database")
    ui.stat(c[3], "Not scored", ui.num(cov["unscored"]),
            "closed, but no outcome row here — an answer nobody looked up")

    # Rule 7, on the live screen as much as the offline one: the unscheduled
    # figure first and at least equal prominence, never a single pooled rate.
    ui.section("Hit rate — split scheduled vs unscheduled",
               f"Of the {cov['graded']:,} alerts this database can grade, how "
               f"many were followed by an 8-K within {hours} hours "
               f"(wall-clock). The denominator is the same for both halves: an "
               f"alert followed by nothing is a miss either way, and there is "
               f"no event to attach a slice to.")
    h = st.columns(3)
    ui.stat(h[0], "Unscheduled — the headline",
            ui.pct(split["unscheduled_rate"], 1)
            if split["unscheduled_rate"] is not None else "—",
            f"{split['unscheduled']} of {split['resolved']} graded alerts. "
            f"Genuinely unscheduled disclosures are the target of the whole "
            f"project.")
    ui.stat(h[1], "Scheduled (item "
            + ", ".join(data.config()["items"]["scheduled"]) + ")",
            ui.pct(split["scheduled_rate"], 1)
            if split["scheduled_rate"] is not None else "—",
            f"{split['scheduled']} of {split['resolved']} graded alerts. "
            f"Results announcements, whose dates are published weeks ahead — "
            f"the easy half.")
    ui.stat(h[2], "Pooled (not the headline)",
            ui.pct(split["pooled"], 1) if split["pooled"] is not None else "—",
            f"{split['filed']} of {split['resolved']}. Shown for completeness "
            f"and never quoted alone: it is more than double the unscheduled "
            f"figure, which is exactly why the two are kept apart.")

    ui.note(ui.honest_rate(split, hours))

    if cov["unscored"]:
        ui.note(
            f"**The committed log and this database disagree, and the gap is "
            f"the denominator.** `live-log/alerts.csv` holds "
            f"**{cov['logged']:,}** alerts; **{cov['answerable']:,}** of them "
            f"have a window that closed long enough ago to be answerable, and "
            f"this database holds an outcome row for **{cov['graded']:,}** of "
            f"those. The remaining **{cov['unscored']:,}** are shown as *not "
            f"scored*, not as pending: their windows closed, nobody has looked "
            f"the answer up here, and calling that \"still open\" would present "
            f"a rate measured on "
            f"{cov['graded'] / cov['answerable']:.1%} of the gradeable alerts "
            f"as if it were measured on all of them. Run "
            f"`python -m src.live.outcomes` against a database holding the "
            f"filings to close the gap.")

    ui.note(
        f"**The live rate is not comparable to the offline precision figures.** "
        f"Offline, each event gets one {hours}-bar window and at most one FLAG "
        f"inside it, so precision counts one alert per event-window. Live, "
        f"`window_id` is `live:<ticker>:<ts>` — **every bar is its own window, "
        f"with no dedupe** — so one sustained anomaly writes several alerts and "
        f"several denominator entries. Measured on this log, cusum's 1,105 "
        f"alerts span 593 distinct {hours}-hour ticker-clusters, 1.86 alerts "
        f"per cluster. The two numbers answer different questions and neither "
        f"is adjusted to match the other.")

    ui.note(
        "**Append-only, and checkably so — with one gap named.** Every row "
        "carries a hash of itself and of the row before it, so an edit to a "
        "logged row, a row removed from the middle, and a row inserted out of "
        "order are all caught — verify with `python -m src.live.alertlog "
        "--verify`. What the chain does **not** prove is that the log is "
        "complete: nothing anchors the head, so truncating the newest rows or "
        "dropping a whole detector leaves a chain that still verifies. Two "
        "things cover that instead — git's own history of this file, and an "
        "export that refuses to write a log with fewer rows per detector than "
        "the one it would overwrite. The monitor re-scores a rolling 48-bar "
        "window each run and suppresses re-detections by natural key, so a "
        "repeated scan writes nothing.")

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
        "outcome": df["outcome_state"].map(data.OUTCOME_WORDS),
        # Rule 7 reaches the table too, not just the figures above it.
        "8-K type": data.outcome_slice(df),
        "lead (trading h)": df.get("lead_trading_h", pd.Series(index=df.index))
                              .map(lambda v: "—" if pd.isna(v) else f"{v:.1f}"),
    })
    st.dataframe(show, width="stretch", hide_index=True, height=460)
    st.caption(
        "**`lead (trading h)` really is trading hours** — rule 3, the same unit "
        "every other lead-time figure in this project uses — while the "
        f"{hours}-hour outcome window above is wall-clock. They are different "
        "clocks on purpose: a company can file overnight or at a weekend, so "
        "\"did a filing follow within two days\" is a question about elapsed "
        "time, but \"how much warning was there\" is only meaningful in hours "
        "the market was open.")
