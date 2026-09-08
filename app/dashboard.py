"""P9-01 — the dashboard shell: routing, masthead, disclaimer.

    .venv/bin/streamlit run app/dashboard.py

A TRIAGE QUEUE, not a prediction display. The user is a compliance analyst —
or an examiner in a viva — deciding in about thirty seconds per row whether
something deserves a closer look. Its job is to make a human faster at
judging, never to make the judgement for them.

Routing lives here so the masthead and the disclaimer wrap every screen. Both
are binding rules in `UI-context.md` (6, 7 and 9), and a screen that forgot one
would present a number without the constraint it was measured against.

Read-only throughout: it opens the database read-only and calls no collector,
so it cannot move the frozen snapshot or spend an API quota.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import data, screens, ui  # noqa: E402

SCREENS = {
    "Today's alerts": (screens.alerts_today,
                       "The triage queue — what fired, and why"),
    "Ticker detail": (screens.ticker_detail,
                      "One alert in full, with the evidence behind it"),
    "Evaluation": (screens.evaluation,
                   "How well the detectors actually do, and against what"),
    "Live monitor log": (screens.monitor_log,
                         "Every alert raised live, with its outcome"),
}


def main() -> None:
    st.set_page_config(page_title="Pre-Announcement Footprints",
                       page_icon="◱", layout="wide",
                       initial_sidebar_state="expanded")
    ui.inject_css()

    with st.sidebar:
        st.markdown("##### Screens")
        page = st.radio("Screen", list(SCREENS), label_visibility="collapsed")
        st.divider()
        st.markdown(
            f'<div class="meta"><b>Vocabulary, held strictly.</b><br>'
            f'We say <i>footprint</i>, never <i>insider trading</i>. Unusual '
            f'trading before a disclosure has innocent explanations, and '
            f'establishing intent would need regulator-held records this '
            f'project does not use.</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown(
            '<div class="meta">Detecting material corporate news before it is '
            'disclosed, from public price and volume data alone.<br><br>'
            'NMAM Institute of Technology · Dept. of ISE</div>',
            unsafe_allow_html=True)

    alerts = data.alerts()
    budget = data.budget_line(alerts)
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())
    newest = int(alerts["ts_utc"].max()) if not alerts.empty else None

    fn, blurb = SCREENS[page]
    ui.masthead(f"{page} — {blurb}")
    st.markdown(
        f'<div class="meta" style="margin:.45rem 0 .9rem 0">'
        f'{ui.utc(now)}'
        + (f' &nbsp;·&nbsp; newest scored bar {ui.utc(newest)}' if newest else "")
        + '</div>', unsafe_allow_html=True)

    # The screen's own headline row comes first: eye-tracking work is
    # consistent that the top-left carries most of the attention, and for a
    # triage queue the first question is "what is in front of me", not "how is
    # the system configured". The budget is still on screen — rule 6 — but as
    # the context for a result rather than the result itself.
    fn()
    ui.budget_strip(budget)
    ui.disclaimer()


if __name__ == "__main__":
    main()
