"""Shared UI helpers, and the place `UI-context.md`'s binding rules live.

ON STYLING — a correction, recorded because it was an instructive mistake.

An earlier version of this file hardcoded a light palette: white panels, near
-black headings, fixed greys. It looked acceptable in a light browser and was
close to unreadable in a dark one — the title rendered near-black on a
near-black page, and a wall of white cards glared out of a dark background.
Streamlit follows the viewer's system preference and this project sets no
theme, so roughly half of all viewers got the broken version. The lesson is
narrow and worth keeping: **do not hardcode colour in a themed app.**

So colour is now Streamlit's, entirely. What is left here is a handful of
layout rules — spacing, table density — that hold in either theme, plus the
formatting and rule helpers below. Native components (`st.metric`,
`st.dataframe`, bordered containers) carry the look, which is also what
`UI-context.md` asked for when it said "default theme, no custom CSS".

THE BINDING RULES. Several exist because breaking one would misrepresent a
result rather than merely look untidy, so they are functions here instead of
prose each screen has to remember:

  rule 2  `footprint`, never "insider trading", never an implied person
  rule 5  never plain accuracy; the headline is precision at the alert budget
  rule 6  the budget is on screen, with how much of it is spent
  rule 7  every timestamp carries its timezone AND whether the market was open
  rule 9  the disclaimer is on every screen
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.timeutils import is_market_open

DISCLAIMER = (
    "Research prototype — not investment advice, and not evidence of "
    "wrongdoing. This detects a **footprint** in public price and volume data: "
    "unusual trading ahead of a disclosure, which has innocent explanations "
    "such as index rebalancing, an analyst note, or a fund unwinding a "
    "position. It does not identify a person, a fund, or an intent."
)

#: Severity bands, set from the OBSERVED distribution rather than round
#: numbers. Across the live log the median alert sits at 1.6x its threshold
#: and the 90th percentile at 4.3x, so a "3x and above is critical" scale — the
#: first thing tried — labelled every visible row "very strong" and carried no
#: information at all. These bands put roughly the top 2%, 10% and 40% in the
#: three upper bands, which is what makes a queue sortable by eye.
_BANDS = [(10.0, "Extreme"), (4.0, "Strong"), (2.0, "Elevated"), (0.0, "Marginal")]

#: Layout only — no colour, so it holds in either theme.
_CSS = """
<style>
  .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1400px; }
  h1 { font-size: 1.5rem !important; font-weight: 640 !important;
       letter-spacing: -.012em; }
  h2 { font-size: 1.05rem !important; font-weight: 620 !important;
       margin-top: 1.6rem !important; }
  h3 { font-size: .92rem !important; font-weight: 620 !important; }
  [data-testid="stMetricValue"] { font-size: 1.28rem; }
  [data-testid="stMetricLabel"] { font-size: .72rem; text-transform: uppercase;
                                  letter-spacing: .06em; opacity: .75; }
  hr { margin: 1.1rem 0; }
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


def short_utc(ts) -> str:
    """Compact form for a dense table, where the column header carries "UTC"."""
    if ts is None or pd.isna(ts):
        return "—"
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%d %b %H:%M")


def strength(score: float, threshold: float) -> tuple[str, str, float]:
    """(band key, words, multiple-of-threshold) for one alert.

    Deliberately NOT called "confidence". These detectors emit scores that are
    not probabilities — the evaluation contract says so, and only a baseline
    claiming probabilities is scored by Brier or ECE. Presenting a raw CUSUM
    statistic as "0.41 confident" would invent a calibration the number does
    not have. The honest uncertainty figure is the measured hit rate, which
    `honest_rate` renders beside it.
    """
    if not threshold:
        return "marginal", "Marginal", float("nan")
    mult = score / threshold
    for edge, words in _BANDS:
        if mult >= edge:
            return words.lower(), words, mult
    return "marginal", "Marginal", mult


def honest_rate(resolved: int, filed: int, rate) -> str:
    """The calibration line rule 8 asks for, stated without flattery."""
    if not resolved or rate is None:
        return ("No alert has a closed 48-hour window yet, so there is no hit "
                "rate to quote. An empty figure is reported rather than a "
                "flattering one.")
    return (f"**{filed} of {resolved}** resolved alerts ({pct(rate, 1)}) were "
            f"followed by an 8-K within 48 hours. Alerts whose window is still "
            f"open are excluded from both sides — otherwise the figure would "
            f"drift with how recently the monitor last ran.")


def section(title: str, explain: str = "") -> None:
    st.subheader(title, anchor=False)
    if explain:
        st.caption(explain)


def chart(fig: go.Figure, height: int = 240, ylab: str = "") -> go.Figure:
    """One chart template. Transparent, so the page theme shows through.

    No background colour is set, for the same reason the palette went: a white
    plot area punched into a dark page is exactly the mistake this module now
    documents.
    """
    fig.update_layout(
        height=height, margin=dict(t=14, b=30, l=6, r=6),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False, hovermode="x unified",
        xaxis=dict(title="", showgrid=True, gridcolor="rgba(128,128,128,.18)",
                   zeroline=False, linecolor="rgba(128,128,128,.35)"),
        yaxis=dict(title=dict(text=ylab, font=dict(size=11)), showgrid=True,
                   gridcolor="rgba(128,128,128,.18)", zeroline=False,
                   linecolor="rgba(128,128,128,.35)"),
    )
    return fig


def disclaimer() -> None:
    """Rule 9 — on every screen, not just the landing page."""
    st.divider()
    st.caption(DISCLAIMER)


def budget_bar(b: dict) -> None:
    """Rule 6 — the constraint the system is tuned to, and how much is spent."""
    c = st.columns(4)
    c[0].metric("Alert budget", f"{b['rate']} / stock / month",
                help="The operational constraint the whole system is tuned to. "
                     "Fixed before any model existed, so it cannot have been "
                     "chosen to flatter a result.")
    c[1].metric(f"Spent in {b['month']}", num(b["used"]),
                help=f"of {num(b['allowance'])} available this month")
    c[2].metric("Universe", num(b["universe"]),
                help="Companies, selected once in advance by fixed liquidity "
                     "rules measured as of the study's start date.")
    c[3].metric("Utilisation",
                pct(b["used"] / b["allowance"], 1) if b["allowance"] else "—",
                help="Share of the month's allowance spent.")


#: Two chart colours chosen to read on BOTH a light and a dark page — a
#: mid-tone blue and a warm marker. Everything else takes the theme's own
#: colours; these two exist because a line has to be some colour, and the
#: previous palette failed precisely by assuming a light background.
SERIES = "#4C8DBF"
MARKER = "#D2705A"
DIM = "rgba(128,128,128,.45)"


def masthead(subtitle: str) -> None:
    """Title and one line of orientation. Native, so it follows the theme."""
    st.title("Pre-Announcement Footprints", anchor=False)
    st.caption(subtitle)


def stat(col, label: str, value: str, note: str = "") -> None:
    """One statistic. `st.metric` so the theme colours it, not this module."""
    col.metric(label, value, help=note or None)


def note(text: str) -> None:
    """A short explanation, kept at top level rather than behind a click.

    Several of these carry binding rules, and a rule a reader must expand to
    find is a rule the screen does not really make.
    """
    st.info(text)


def budget_strip(b: dict) -> None:
    """Rule 6, kept VISIBLE rather than behind a click.

    An earlier attempt put this in an expander, on the grounds that a triage
    queue should lead with the queue. The first half of that is right — the
    headline row now answers "what is in front of me" — but the conclusion was
    not: rule 6 says a user must SEE how much of the budget is spent, and a
    rule a reader has to expand to find is a rule the screen does not really
    make. So it stays on the page, as one compact line rather than four
    competing panels.
    """
    used, allow = b["used"], b["allowance"]
    st.caption(
        f"**Alert budget** {b['rate']} / stock / month across "
        f"{num(b['universe'])} companies — **{num(used)} of {num(allow)} "
        f"spent in {b['month']}** ({pct(used / allow, 1) if allow else '—'} of "
        f"the allowance). The budget is the operational constraint the whole "
        f"system is tuned to, fixed before any model existed so it cannot have "
        f"been chosen to flatter a result; precision is measured at exactly it."
    )
