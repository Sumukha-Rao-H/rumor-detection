"""P9-01 — the dashboard shell: routing, header, disclaimer.

    .venv/bin/streamlit run app/dashboard.py

A TRIAGE QUEUE, not a prediction display. The user is a compliance analyst —
or an examiner in a viva — deciding in about thirty seconds per row whether
something is worth a closer look. Its job is to make a human faster at
judging, never to make the judgement for them.

Routing lives here so the header and the disclaimer wrap every screen: both
are binding rules in `UI-context.md` (6, 7 and 9), and a screen that forgot one
would be a screen presenting a number without the constraint it was measured
against.

Read-only throughout. It opens the database read-only and calls no collector,
so it cannot move the frozen snapshot or spend an API quota.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import data, screens, ui  # noqa: E402

SCREENS = {
    "Today's alerts": screens.alerts_today,
    "Ticker detail": screens.ticker_detail,
    "Evaluation": screens.evaluation,
    "Live monitor log": screens.monitor_log,
}


def main() -> None:
    st.set_page_config(page_title="Pre-Announcement Footprints",
                       page_icon="📈", layout="wide")

    with st.sidebar:
        st.markdown("### Screens")
        page = st.radio("Screen", list(SCREENS), label_visibility="collapsed")
        st.divider()
        st.caption(
            "**Vocabulary held strictly.** We say *footprint*, never *insider "
            "trading*. Unusual trading before a disclosure has innocent "
            "explanations, and identifying intent would need regulator-held "
            "records this project does not use.")

    alerts = data.alerts()
    newest = int(alerts["ts_utc"].max()) if not alerts.empty else None
    ui.header(page, data.budget_line(alerts), newest)
    st.divider()

    SCREENS[page]()
    ui.disclaimer()


if __name__ == "__main__":
    main()
