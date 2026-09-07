"""Shared UI pieces, and the place `UI-context.md`'s binding rules are enforced.

The rules there are not style preferences — several exist because breaking
them would misrepresent the result. They live in functions here rather than in
each screen's prose so a new screen cannot forget one:

  rule 2  `footprint`, never "insider trading", and never an implied person
  rule 5  never plain accuracy; the headline is precision at the alert budget
  rule 6  the budget is on screen, with how much is spent
  rule 7  every timestamp carries its timezone AND whether the market was open
  rule 9  the disclaimer is on every screen

Colour carries exactly one meaning — alert strength — and never direction.
Red-for-down/green-for-up would read as a trading terminal, which is the
precise wrong impression (rule: "awareness, not advice").
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src.utils.timeutils import is_market_open

DISCLAIMER = (
    "Research prototype. Not investment advice. Not evidence of wrongdoing. "
    "This detects a **footprint** in public price and volume data — unusual "
    "trading before a disclosure, which has many innocent causes (index "
    "rebalancing, an analyst note, a fund unwinding). It does not identify a "
    "person, a fund, or an intent."
)

#: Strength bands. One meaning only, and never direction.
_BANDS = [(3.0, "🔴", "very strong"), (2.0, "🟠", "strong"),
          (1.0, "🟡", "moderate"), (0.0, "⚪", "at threshold")]


def utc(ts: int | float | None, with_market: bool = True) -> str:
    """A timestamp as the rules require: explicit UTC, and market state.

    A bare "14:30" is a bug — rule 7. The market flag matters because a move
    outside the session means something different from the same move inside it.
    """
    if ts is None or pd.isna(ts):
        return "—"
    ts = int(ts)
    when = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
    out = when.strftime("%Y-%m-%d %H:%M UTC")
    if with_market:
        try:
            out += " · market OPEN" if is_market_open(ts) else " · market closed"
        except Exception:
            pass
    return out


def strength(score: float, threshold: float) -> tuple[str, str, float]:
    """(icon, words, multiple-of-threshold) for one alert.

    Deliberately NOT called "confidence". These detectors emit scores that are
    not probabilities — the evaluation contract says so, and only a baseline
    claiming probabilities is scored by Brier or ECE. Presenting a raw CUSUM
    statistic as "0.41 confident" would invent a calibration the number does
    not have. The honest uncertainty figure is the measured hit rate, which
    `honest_rate` renders beside it.
    """
    if not threshold:
        return "⚪", "at threshold", float("nan")
    mult = score / threshold
    for edge, icon, words in _BANDS:
        if mult >= edge:
            return icon, words, mult
    return "⚪", "at threshold", mult


def honest_rate(resolved: int, filed: int, rate: float | None) -> str:
    """The calibration line rule 8 asks for, stated without flattery."""
    if not resolved or rate is None:
        return ("No alert has a closed 48-hour window yet, so there is no hit "
                "rate to quote. An empty figure is reported rather than a "
                "flattering one.")
    return (f"Measured so far: **{filed} of {resolved} resolved alerts "
            f"({rate:.1%})** were followed by an 8-K within 48 hours. Alerts "
            f"whose window is still open are excluded from both sides, or the "
            f"figure would drift with how recently the monitor ran.")


def header(page: str, budget: dict, newest_ts: int | None) -> None:
    """Title, clock, market state and the alert budget — rules 6 and 7."""
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())
    st.title("Pre-Announcement Footprints")
    st.caption(f"{page} · now {utc(now)}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Alert budget", f"{budget['rate']} / stock / month",
              help="The operational constraint the whole system is tuned to. "
                   "Fixed before any model existed, so it cannot be chosen to "
                   "flatter a result.")
    c2.metric(f"Spent in {budget['month']}", f"{budget['used']:,}",
              delta=f"of {budget['allowance']:,} available",
              delta_color="off")
    c3.metric("Universe", f"{budget['universe']:,} companies")
    if newest_ts:
        st.caption(f"Newest scored bar: {utc(newest_ts)}")


def split_note() -> None:
    """Rule 4 — the scheduled/unscheduled split is visible, never a tooltip."""
    st.info(
        "**Scheduled and unscheduled are never pooled.** Scheduled events "
        "(results announcements) have dates published weeks ahead, so "
        "detecting a run-up before one is far less interesting. Unscheduled "
        "events are the real target, and pooling would let the easy half carry "
        "the number.", icon="⚖️")


def disclaimer() -> None:
    """Rule 9 — on every screen, not just the landing page."""
    st.divider()
    st.caption(DISCLAIMER)


def no_accuracy_note() -> None:
    """Rule 5, stated where an examiner will look for it."""
    st.warning(
        "**Plain accuracy is not reported anywhere, by design.** Only ~0.285% "
        "of hours precede an event, so a system that always says \"nothing is "
        "coming\" is 99.71% accurate and useless. The function that would "
        "compute it raises an error instead. The headline is precision at the "
        "fixed alert budget above.", icon="🚫")
