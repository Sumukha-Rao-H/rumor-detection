"""Streamlit demo — plan §12. Watch the agent decide, hour by hour.

Three tabs:

  Replay      one rumor event, unrolled. The price chart, the Reddit posts that
              raised the claim, and the agent's action at every hour — WAIT,
              WAIT, ... COMMIT — against the moment the official record caught
              up. This is the picture the whole project is about.
  Evaluation  the agent against the baselines on the same split: accuracy,
              calibration, and the Time Delta distribution.
  Dataset     what the model was trained on, including the class balance,
              which currently explains most of the behaviour.

Run:
  streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db                                              # noqa: E402
from src.eval.metrics import summarize                          # noqa: E402
from src.rl.env import (ACTION_NAMES, COMMIT_FALSE, COMMIT_TRUE,  # noqa: E402
                        WAIT, EventStore, RumorVerificationEnv, manifest)
from src.rl.train import ImmediatePolicy, RandomPolicy, load_policy, rollout  # noqa: E402
from src.utils.config import load_config                        # noqa: E402

HOUR = 3600
ACTION_COLOR = {WAIT: "#9aa0a6", COMMIT_TRUE: "#1a7f37", COMMIT_FALSE: "#b3261e"}

st.set_page_config(page_title="Rumor Verification Agent", page_icon="📈",
                   layout="wide")


# ------------------------------------------------------------------ loading

@st.cache_resource
def get_cfg():
    return load_config()


@st.cache_data
def get_events(_cfg):
    path = Path(_cfg["paths"]["events"])
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def get_bars(_cfg, ticker: str, start: int, end: int):
    conn = db.get_conn(_cfg["paths"]["db"])
    rows = db.bars_in_range(conn, ticker, start, end, _cfg["market"]["interval"])
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data
def get_posts(_cfg, post_ids: tuple):
    if not post_ids:
        return pd.DataFrame()
    conn = db.get_conn(_cfg["paths"]["db"])
    marks = ",".join("?" * len(post_ids))
    rows = conn.execute(
        f"""SELECT id, subreddit, title, author, created_utc, score, num_comments
            FROM posts WHERE id IN ({marks}) ORDER BY created_utc""",
        list(post_ids)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_resource
def get_policy(_cfg, algo: str, split: str, _frame_key: str):
    """A policy plus a sequential env over the chosen split."""
    frame = get_events(_cfg)
    store = EventStore(_cfg["paths"]["states"], frame, split)
    env = RumorVerificationEnv(store, _cfg, sequential=True)
    model_path = Path("models") / f"{algo}.zip"
    policy = load_policy(algo, model_path, env, seed=0)
    return policy, env


def utc(ts: int) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


# ------------------------------------------------------------------ sidebar

cfg = get_cfg()
events = get_events(cfg)

st.sidebar.title("Rumor Verification")
st.sidebar.caption("Cross-domain rumor verification via sequential decision-making")

if events.empty:
    st.error("No dataset found. Build one first:\n\n"
             "```\npython -m src.pipeline.dataset --from-proposals\n"
             "python -m src.pipeline.features --encoder hashed\n```")
    st.stop()

info = manifest(cfg)
sources = set(events.get("label_source", []))
provisional = any("provisional" in str(s) for s in sources)

available = ["random", "immediate"] + [
    a for a in ("ppo", "dqn") if (Path("models") / f"{a}.zip").exists()]
algo = st.sidebar.selectbox("Policy", available,
                            index=len(available) - 1 if len(available) > 2 else 0)
split = st.sidebar.selectbox("Split", ["test", "val", "train"], index=0)
st.sidebar.divider()
st.sidebar.metric("Events in dataset", len(events))
st.sidebar.metric("Positive class", f"{events['label'].mean():.1%}")
if info:
    st.sidebar.caption(f"Encoder: `{info.get('encoder')}` · "
                       f"D={info.get('state_dim')}")

if provisional:
    st.warning(
        "**Provisional dataset.** Labels are the *machine's* verdicts, not the "
        "human-reviewed ones — Phase 2 review is still in progress. The "
        "pipeline is real end to end; the numbers are not reportable yet.",
        icon="⚠️")

policy, env = get_policy(cfg, algo, split, f"{len(events)}-{algo}-{split}")
split_frame = events[events["split"] == split].reset_index(drop=True)

tab_replay, tab_eval, tab_data = st.tabs(
    ["▶  Replay an event", "📊  Evaluation", "🗃  Dataset"])


# ------------------------------------------------------------------- replay

with tab_replay:
    ids = [e.event_id for e in env.store.events]
    labels = {e.event_id: e.label for e in env.store.events}
    choice = st.selectbox(
        "Rumor event", ids,
        format_func=lambda e: f"{e.split('-')[0]:6}  ·  {e}  "
                              f"({'TRUE' if labels[e] else 'FALSE'})")

    episode = next(e for e in env.store.events if e.event_id == choice)
    row = events[events["event_id"] == choice].iloc[0]

    # Unroll the policy over this one event.
    steps, obs = [], episode.X
    committed_at, verdict, p_true = None, None, None
    for t in range(episode.T):
        action, p = policy.act(obs[t])
        steps.append(action)
        if action != WAIT:
            committed_at, verdict, p_true = t, action, p
            break

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"#### {row['ticker']} — {row.get('claim_type', 'claim')}")
        st.write(row.get("claim_summary") or "_no claim summary_")
    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("Truth", "TRUE" if episode.label else "FALSE")
        if committed_at is None:
            c2.metric("Agent", "no commit")
            c3.metric("Δ", "—")
        else:
            said = "TRUE" if verdict == COMMIT_TRUE else "FALSE"
            hit = (verdict == COMMIT_TRUE) == bool(episode.label)
            c2.metric("Agent", said, "correct" if hit else "wrong",
                      delta_color="normal" if hit else "inverse")
            t_commit = episode.t0_utc + committed_at * HOUR
            delta = (episode.t_official_utc - t_commit) / HOUR
            c3.metric("Δ vs news", f"{delta:+.0f} h")

    # --- price + decision timeline -------------------------------------
    t0 = episode.t0_utc
    bars = get_bars(cfg, row["ticker"], t0 - 2 * 86400, t0 + 3 * 86400)
    fig = go.Figure()
    if not bars.empty:
        fig.add_trace(go.Scatter(
            x=[utc(t) for t in bars["ts_utc"]], y=bars["close"],
            mode="lines", name=f"{row['ticker']} price",
            line=dict(color="#1f77b4", width=2)))
    fig.add_vline(x=utc(t0), line=dict(color="#9aa0a6", dash="dot"),
                  annotation_text="rumor t₀", annotation_position="top left")
    if episode.t_official_utc:
        fig.add_vline(x=utc(episode.t_official_utc),
                      line=dict(color="#6a3ab2", dash="dash"),
                      annotation_text="official news", annotation_position="top right")
    if committed_at is not None:
        fig.add_vline(x=utc(t0 + committed_at * HOUR),
                      line=dict(color=ACTION_COLOR[verdict], width=3),
                      annotation_text="agent commits", annotation_position="bottom right")
    posts = get_posts(cfg, tuple(row["post_ids"]))
    if not posts.empty and not bars.empty:
        near = bars["close"].iloc[0]
        fig.add_trace(go.Scatter(
            x=[utc(t) for t in posts["created_utc"]],
            y=[near] * len(posts), mode="markers", name="Reddit posts",
            marker=dict(symbol="triangle-up", size=11, color="#e8710a"),
            hovertext=posts["title"].str.slice(0, 110), hoverinfo="text"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12),
                      yaxis_title="price", xaxis_title=None)
    st.plotly_chart(fig, width="stretch")

    # --- the action tape -----------------------------------------------
    st.markdown("##### Hourly decisions")
    tape = go.Figure()
    tape.add_trace(go.Bar(
        x=list(range(len(steps))), y=[1] * len(steps),
        marker_color=[ACTION_COLOR[a] for a in steps],
        hovertext=[f"hour {i}: {ACTION_NAMES[a]}" for i, a in enumerate(steps)],
        hoverinfo="text", showlegend=False))
    tape.update_layout(height=110, margin=dict(l=10, r=10, t=6, b=24),
                       yaxis=dict(visible=False), xaxis_title="hours since t₀",
                       bargap=0.15)
    st.plotly_chart(tape, width="stretch")
    waited = len(steps) - 1 if committed_at is not None else len(steps)
    st.caption(f"WAIT ×{waited}, then "
               f"{ACTION_NAMES[verdict] if committed_at is not None else 'timeout'}"
               + (f" · p(TRUE)={p_true:.2f}" if p_true is not None else ""))

    if not posts.empty:
        with st.expander(f"The {len(posts)} post(s) behind this event"):
            st.dataframe(
                posts.assign(posted=[utc(t).strftime("%Y-%m-%d %H:%M")
                                     for t in posts["created_utc"]])
                     [["posted", "subreddit", "title", "score", "num_comments"]],
                width="stretch", hide_index=True)


# --------------------------------------------------------------- evaluation

with tab_eval:
    st.markdown(f"#### {algo.upper()} vs baselines on the **{split}** split")
    compare = st.multiselect("Policies to compare", available,
                             default=list(dict.fromkeys([algo, "random", "immediate"])))

    rows, roll_cache = [], {}
    for name in compare:
        pol, ev = get_policy(cfg, name, split, f"{len(events)}-{name}-{split}")
        rolls = rollout(ev, pol)
        roll_cache[name] = rolls
        s = summarize(rolls)
        rows.append({
            "policy": name, "accuracy": s["accuracy"], "f1": s["f1"],
            "committed": s["committed"], "abstained": s["abstention_rate"],
            "brier": s["brier"], "ece": s["ece"],
            "mean commit (h)": s["mean_commit_hours"],
            "Δ median (h)": s["time_delta"]["median"],
            "Δ ≥ 24h": s["time_delta"]["share_ge_24h"],
            "mean reward": s["mean_reward"]})
    table = pd.DataFrame(rows).set_index("policy")
    st.dataframe(table.style.format(precision=3, na_rep="—"),
                 width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Time Delta Advantage")
        hist = go.Figure()
        for name, rolls in roll_cache.items():
            deltas = [r.delta_hours for r in rolls
                      if r.delta_hours is not None and r.pred == r.label]
            if deltas:
                hist.add_trace(go.Histogram(x=deltas, name=name, opacity=0.6,
                                            nbinsx=24))
        hist.add_vline(x=0, line=dict(color="#b3261e", dash="dash"),
                       annotation_text="news breaks")
        hist.update_layout(barmode="overlay", height=300,
                           margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="Δ hours (positive = agent first)")
        st.plotly_chart(hist, width="stretch")
    with c2:
        st.markdown("##### Confusion — " + algo.upper())
        s = summarize(roll_cache.get(algo, roll_cache[compare[0]]))
        c = s["confusion"]
        mat = go.Figure(go.Heatmap(
            z=[[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
            x=["said FALSE", "said TRUE"], y=["is FALSE", "is TRUE"],
            text=[[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
            texttemplate="%{text}", colorscale="Blues", showscale=False))
        mat.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(mat, width="stretch")

    st.caption("Δ is measured against aggregator-visible timestamps, which lag "
               "the wire by minutes (plan §11.3). Abstentions are excluded from "
               "accuracy and reported separately.")


# ------------------------------------------------------------------ dataset

with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", len(events))
    c2.metric("TRUE", int(events["label"].sum()))
    c3.metric("FALSE", int((1 - events["label"]).sum()))
    c4.metric("Minority share", f"{min(events['label'].mean(), 1 - events['label'].mean()):.1%}")

    counts = (events.groupby(["split", "label"]).size()
              .unstack(fill_value=0).rename(columns={0: "FALSE", 1: "TRUE"}))
    order = [s for s in ("train", "val", "test") if s in counts.index]
    counts = counts.loc[order]
    bar = go.Figure()
    for col, color in (("FALSE", "#b3261e"), ("TRUE", "#1a7f37")):
        if col in counts:
            bar.add_trace(go.Bar(x=counts.index, y=counts[col], name=col,
                                 marker_color=color))
    bar.update_layout(barmode="stack", height=300,
                      margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="events")
    st.plotly_chart(bar, width="stretch")

    st.markdown("##### Split boundaries (temporal, never random)")
    st.dataframe(
        events.groupby("split")
              .agg(events=("event_id", "count"), positives=("label", "sum"),
                   first=("t0_iso", "min"), last=("t0_iso", "max"))
              .loc[order],
        width="stretch")

    if "claim_type" in events:
        st.markdown("##### Claim types")
        st.bar_chart(events["claim_type"].value_counts())
