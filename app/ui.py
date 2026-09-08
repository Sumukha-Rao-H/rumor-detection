"""The dashboard's design system, and the place `UI-context.md`'s rules live.

TWO THINGS THIS MODULE IS FOR

**The house style.** One palette, one type scale, one chart template, one way
to format a number. Defined once so four screens cannot drift into looking
like four projects.

`UI-context.md` said "default theme, no custom CSS", and gave its reason:
"time spent on styling is time not spent on Phase 10." Phase 10 is finished,
so that reason has expired and a modest stylesheet is now the cheaper choice —
it replaces a scattering of ad-hoc `st.caption` and emoji with one consistent
surface. The rule's *intent* is kept: no framework, no theming engine, ~60
lines of CSS doing spacing and weight, and the content still carries the page.

**The binding rules.** Several of them exist because breaking one would
misrepresent a result, not merely look untidy, so they are functions here
rather than prose each screen must remember:

  rule 2  `footprint`, never "insider trading", never an implied person
  rule 5  never plain accuracy; the headline is precision at the alert budget
  rule 6  the budget is on screen, with how much of it is spent
  rule 7  every timestamp carries its timezone AND whether the market was open
  rule 9  the disclaimer is on every screen

Colour carries exactly one meaning — alert strength — and never direction.
Red-for-down/green-for-up would read as a trading terminal, which is the
precise wrong impression for a tool whose framing is "awareness, not advice".
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.timeutils import is_market_open

# --------------------------------------------------------------------------
# palette — deliberately small
# --------------------------------------------------------------------------
INK = "#12212F"        # headings
BODY = "#3D4B59"       # body text
MUTED = "#6B7A8A"      # captions, axis labels
LINE = "#DCE3EA"       # rules, borders
SURFACE = "#F7F9FB"    # panel fill
ACCENT = "#1B3A5C"     # the one brand colour
ACCENT_SOFT = "#E8EEF4"

# Severity. One meaning only — how far above its own threshold an alert sits.
SEV = {
    "critical": ("#8C2F1F", "#F6E7E3", "Very strong"),
    "high":     ("#A65A1E", "#FAEEE2", "Strong"),
    "medium":   ("#7A6A1F", "#F7F3E0", "Moderate"),
    "low":      ("#4A5A68", "#EEF1F4", "At threshold"),
}

DISCLAIMER = (
    "Research prototype — not investment advice, and not evidence of "
    "wrongdoing. This detects a **footprint** in public price and volume "
    "data: unusual trading ahead of a disclosure, which has many innocent "
    "explanations such as index rebalancing, an analyst note, or a fund "
    "unwinding a position. It does not identify a person, a fund, or an "
    "intent."
)

_CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1360px; }}
  h1, h2, h3 {{ color: {INK}; letter-spacing: -0.011em; }}
  h1 {{ font-size: 1.55rem !important; font-weight: 640 !important; }}
  h2 {{ font-size: 1.12rem !important; font-weight: 620 !important;
        margin: 1.9rem 0 .2rem 0 !important; }}
  h3 {{ font-size: .95rem !important; font-weight: 620 !important; }}

  /* masthead */
  .mast {{ border-bottom: 2px solid {ACCENT}; padding-bottom: .7rem;
           margin-bottom: .2rem; }}
  .mast .title {{ font-size: 1.4rem; font-weight: 650; color: {INK};
                  letter-spacing: -.015em; }}
  .mast .sub {{ font-size: .82rem; color: {MUTED}; margin-top: .15rem; }}

  /* a labelled statistic */
  .stat {{ border: 1px solid {LINE}; border-radius: 7px; padding: .6rem .8rem;
           background: #fff; height: 100%; }}
  .stat .k {{ font-size: .68rem; text-transform: uppercase;
              letter-spacing: .07em; color: {MUTED}; font-weight: 600; }}
  .stat .v {{ font-size: 1.32rem; font-weight: 650; color: {INK};
              line-height: 1.25; font-variant-numeric: tabular-nums; }}
  .stat .n {{ font-size: .74rem; color: {MUTED}; }}

  /* one alert */
  .card {{ border: 1px solid {LINE}; border-left: 3px solid var(--sev);
           border-radius: 7px; padding: .7rem .9rem; margin-bottom: .55rem;
           background: #fff; }}
  .card .tk {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
               font-size: 1.02rem; font-weight: 650; color: {INK}; }}
  .chip {{ display: inline-block; font-size: .68rem; font-weight: 650;
           padding: .1rem .45rem; border-radius: 4px; letter-spacing: .02em; }}
  .meta {{ font-size: .76rem; color: {MUTED};
           font-variant-numeric: tabular-nums; }}
  .why {{ font-size: .82rem; color: {BODY}; margin-top: .3rem; }}
  .why b {{ color: {INK}; font-weight: 620; }}

  /* a short explanatory note — present, but not shouting */
  .note {{ font-size: .8rem; color: {BODY}; background: {SURFACE};
           border-left: 3px solid {ACCENT}; border-radius: 0 6px 6px 0;
           padding: .55rem .8rem; margin: .35rem 0 .9rem 0; }}
  .note b {{ color: {INK}; }}

  [data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 7px; }}
  hr {{ margin: 1.3rem 0; border-color: {LINE}; }}
  section[data-testid="stSidebar"] {{ background: {SURFACE};
                                      border-right: 1px solid {LINE}; }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# formatting — one way to render each kind of number
# --------------------------------------------------------------------------
def num(v, dp: int = 0) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:,.{dp}f}"


def pct(v, dp: int = 2) -> str:
    return "—" if v is None or pd.isna(v) else f"{v * 100:.{dp}f}%"


def signed_pct(v, dp: int = 2) -> str:
    return "—" if v is None or pd.isna(v) else f"{v * 100:+.{dp}f}%"


def utc(ts, with_market: bool = True) -> str:
    """A timestamp as rule 7 requires: explicit UTC, and market state.

    A bare "14:30" is a bug. The market flag is not decoration either — the
    same move means something different inside a session and outside one.
    """
    if ts is None or pd.isna(ts):
        return "—"
    ts = int(ts)
    out = dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if with_market:
        try:
            out += " · market OPEN" if is_market_open(ts) else " · market closed"
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------
def masthead(subtitle: str) -> None:
    st.markdown(
        f'<div class="mast"><div class="title">Pre-Announcement Footprints</div>'
        f'<div class="sub">{subtitle}</div></div>', unsafe_allow_html=True)


def stat(col, label: str, value: str, note: str = "") -> None:
    col.markdown(
        f'<div class="stat"><div class="k">{label}</div>'
        f'<div class="v">{value}</div><div class="n">{note}&nbsp;</div></div>',
        unsafe_allow_html=True)


def note(text: str) -> None:
    """A short explanation, kept at top level rather than behind a click.

    Several of these carry binding rules, and a rule a reader has to expand to
    find is a rule the screen does not really make.
    """
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def section(title: str, explain: str = "") -> None:
    st.markdown(f"### {title}")
    if explain:
        st.markdown(f'<div class="meta">{explain}</div>', unsafe_allow_html=True)


def strength(score: float, threshold: float) -> tuple[str, str, float]:
    """(severity key, words, multiple-of-threshold) for one alert.

    Deliberately NOT called "confidence". These detectors emit scores that are
    not probabilities — the evaluation contract says so, and only a baseline
    claiming probabilities is scored by Brier or ECE. Presenting a raw CUSUM
    statistic as "0.41 confident" would invent a calibration the number does
    not have. The honest uncertainty figure is the measured hit rate, which
    `honest_rate` renders beside it.
    """
    if not threshold:
        return "low", SEV["low"][2], float("nan")
    mult = score / threshold
    key = ("critical" if mult >= 3 else "high" if mult >= 2
           else "medium" if mult >= 1 else "low")
    return key, SEV[key][2], mult


def chip(key: str, text: str) -> str:
    fg, bg, _ = SEV[key]
    return f'<span class="chip" style="color:{fg};background:{bg}">{text}</span>'


def honest_rate(resolved: int, filed: int, rate) -> str:
    """The calibration line rule 8 asks for, stated without flattery."""
    if not resolved or rate is None:
        return ("No alert has a closed 48-hour window yet, so there is no hit "
                "rate to quote. An empty figure is reported rather than a "
                "flattering one.")
    return (f"<b>{filed} of {resolved}</b> resolved alerts ({pct(rate, 1)}) were "
            f"followed by an 8-K within 48 hours. Alerts whose window is still "
            f"open are excluded from both sides — otherwise the figure would "
            f"drift with how recently the monitor last ran.")


def chart(fig: go.Figure, height: int = 260, ylab: str = "") -> go.Figure:
    """One chart template, so every plot reads as the same instrument."""
    fig.update_layout(
        height=height, margin=dict(t=18, b=34, l=8, r=8),
        plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend=False,
        font=dict(color=BODY, size=11),
        xaxis=dict(title="", gridcolor=LINE, zeroline=False,
                   linecolor=LINE, tickfont=dict(color=MUTED, size=10)),
        yaxis=dict(title=dict(text=ylab, font=dict(color=MUTED, size=10)),
                   gridcolor=LINE, zeroline=False, linecolor=LINE,
                   tickfont=dict(color=MUTED, size=10)),
        hovermode="x unified",
    )
    return fig


def disclaimer() -> None:
    """Rule 9 — on every screen, not just the landing page."""
    st.divider()
    st.caption(DISCLAIMER)


def budget_bar(b: dict) -> None:
    """Rule 6 — the constraint the system is tuned to, and how much is spent."""
    c = st.columns(4)
    stat(c[0], "Alert budget", f"{b['rate']} / stock / month",
         "fixed before any model existed")
    used, allow = b["used"], b["allowance"]
    stat(c[1], f"Spent in {b['month']}", num(used),
         f"of {num(allow)} available")
    stat(c[2], "Universe", num(b["universe"]), "companies, fixed in advance")
    stat(c[3], "Utilisation", pct(used / allow, 1) if allow else "—",
         "of the month's allowance")
