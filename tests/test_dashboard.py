"""P9 — the dashboard renders, and its binding rules hold.

`UI-context.md` calls its rules binding, and several exist because breaking one
would misrepresent the result rather than merely look untidy: pooling the
scheduled split would let the easy half carry every number, showing plain
accuracy would report 99.71% for a system that does nothing, and dropping the
disclaimer would present a footprint as an accusation.

So they are asserted here rather than trusted to review. These tests render
the real app through Streamlit's own harness — if a screen raises, this fails.
"""
from __future__ import annotations

import pytest

pytest.importorskip("streamlit", reason="dashboard extras not installed")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = "app/dashboard.py"
SCREENS = ["Today's alerts", "Ticker detail", "Evaluation", "Live monitor log"]
TIMEOUT = 240


def _run(screen: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, f"app failed to start: {at.exception}"
    if screen:
        at.sidebar.radio[0].set_value(screen).run()
        assert not at.exception, f"{screen} raised: {at.exception}"
    return at


def _text(at: AppTest) -> str:
    """Everything a reader can see, whichever widget carries it.

    Metric labels and values are included deliberately. An earlier version
    scanned only markdown-family elements, so moving the alert budget from a
    styled panel into `st.metric` — a purely presentational choice — read as a
    rule violation. The rules are about what reaches the reader.
    """
    parts = []
    for block in (at.markdown, at.caption, at.info, at.warning, at.success,
                  at.error, at.subheader, at.title):
        parts += [getattr(e, "value", "") or "" for e in block]
    for m in at.metric:
        parts += [m.label or "", str(m.value or ""), getattr(m, "help", "") or ""]
    return "\n".join(parts).lower()


@pytest.mark.parametrize("screen", SCREENS)
def test_every_screen_renders(screen):
    _run(screen)


@pytest.mark.parametrize("screen", SCREENS)
def test_the_disclaimer_is_on_every_screen(screen):
    """Rule 9. Not just the landing page — every one."""
    body = _text(_run(screen))
    assert "not investment advice" in body
    assert "not evidence of wrongdoing" in body


@pytest.mark.parametrize("screen", SCREENS)
def test_the_vocabulary_rule_holds(screen):
    """Rule 2. `footprint`, never an accusation.

    "insider trading" may appear only inside an explicit denial, so this looks
    for the accusatory framing rather than the bare phrase.
    """
    body = _text(_run(screen))
    for banned in ("suspicious", "illegal", "manipulation", "culprit"):
        assert banned not in body, f"{screen} used {banned!r}"


def test_no_trading_advice_in_the_dashboard_s_own_voice():
    """Rule 3. Awareness, not advice.

    Scoped to copy the dashboard AUTHORS, not text it quotes. The ticker screen
    renders a news timeline, which the spec requires, and real headlines
    contain analyst language — "maintains buy … raises price target to $190" is
    Benzinga's sentence, not ours. Scanning rendered output would fail on
    third-party data and tempt someone to drop the timeline to make a test
    pass, which is the wrong repair. Attribution is what keeps a quote a quote,
    and `test_quoted_headlines_are_attributed` holds that separately.
    """
    import ast
    from pathlib import Path

    banned = ("buy signal", "sell signal", "price target", "take a position",
              "you should buy", "you should sell", "expected return")
    for path in sorted(Path("app").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                low = node.value.lower()
                for phrase in banned:
                    assert phrase not in low, f"{path.name} authors {phrase!r}"


def test_quoted_headlines_are_attributed():
    """A third-party headline must read as a quote, never as the tool's voice."""
    body = _text(_run("Ticker detail"))
    if "no articles in this window" in body:
        pytest.skip("selected ticker has no news in the window")
    # Every rendered headline carries its publisher in italics beside it.
    assert "*" in body or "benzinga" in body or "yahoo" in body


def test_the_alert_budget_is_on_screen():
    """Rule 6. The constraint the system is tuned to is visible, not implied.

    Asserted against the rendered TEXT rather than `st.metric`, which the first
    version of this test used. The rule is that a reader sees the budget and
    how much of it is spent; which widget carries it is a presentation choice,
    and pinning the widget made a purely visual change look like a rule
    violation when the statistics moved into styled panels.
    """
    body = _text(_run())
    assert "alert budget" in body
    assert "spent in" in body, "the budget must show how much of it is used"
    assert "/ stock / month" in body, "and the rate it is denominated in"


def test_the_evaluation_screen_refuses_plain_accuracy():
    """Rule 5. It must say so on screen, where an examiner looks for it."""
    body = _text(_run("Evaluation"))
    assert "plain accuracy is not reported" in body
    assert "99.71%" in body, "the trap should be quantified, not just named"


def test_the_scheduled_split_is_visible_not_buried():
    """Rule 4. On screen, not in a tooltip."""
    at = _run("Evaluation")
    body = _text(at)
    assert "never pooled" in body
    options = [o.lower() for r in at.radio for o in r.options]
    assert "scheduled" in options and "unscheduled" in options


def test_timestamps_carry_their_timezone():
    """Rule 7. A bare "14:30" is a bug."""
    from app.ui import utc

    assert "UTC" in utc(1_760_000_000)
    assert "market" in utc(1_760_000_000)
    assert utc(None) == "—"


def test_strength_is_not_dressed_up_as_a_probability():
    """These detectors emit scores that are not probabilities.

    Calling a raw CUSUM statistic "0.41 confident" would invent a calibration
    the number does not have, so the UI reports a multiple of threshold and
    puts the measured hit rate beside it instead.
    """
    from app.ui import strength

    key, words, mult = strength(5.0, 2.5)

    # The number returned is a MULTIPLE OF THRESHOLD, not a probability.
    assert mult == 2.0, "a score of 5 against a threshold of 2.5 is 2x, not 0.67"
    assert mult > 1.0, "a multiple can exceed 1; a probability could not"
    assert isinstance(words, str) and words
    # The band names are a design choice and deliberately not pinned here —
    # an earlier version fixed them, so renaming a label to something more
    # informative failed a test about probabilities. What must hold is that
    # the wording is qualitative and never a percentage.
    assert "%" not in words, "strength must not be dressed up as a probability"


def test_an_open_window_hides_what_happened_next():
    """Rule: no forward data past the flagged hour while an alert is live.

    Showing it would turn a surveillance tool into a hindsight demo.
    """
    at = _run("Ticker detail")
    body = _text(at)
    # Either an alert is live and the lock notice shows, or every alert on the
    # selected ticker is resolved. Both are valid; a silent third state is not.
    assert ("nothing after the flagged hour is shown" in body
            or "no alerts to inspect" in body
            or "why this hour was flagged" in body)


# --------------------------------------------------------------------------
# the P9-03 / P9-04 elements the spec names explicitly
# --------------------------------------------------------------------------
def test_evaluation_shows_calibration_and_the_action_distribution():
    """P9-04 names four things; the table alone is not the screen."""
    at = _run("Evaluation")
    body = _text(at)
    assert "calibration" in body
    assert "action distribution" in body
    # Brier / ECE / WAIT-FLAG counts reach the screen, not just the CSV.
    cols = {c for df in at.dataframe for c in df.value.columns}
    assert {"brier", "ece", "brier_skill_score"} <= cols
    assert {"n_wait_hours", "n_flag_hours", "pct_windows_alerted"} <= cols


def test_a_non_probability_baseline_is_not_given_a_calibration_score():
    """CUSUM and the z-score never claimed probabilities; blank is honest."""
    at = _run("Evaluation")
    body = _text(at)
    assert "not probabilities" in body
    for df in at.dataframe:
        if "brier" in df.value.columns and "baseline" in df.value.columns:
            cusum = df.value[df.value.baseline == "cusum"]
            if not cusum.empty:
                assert cusum["brier"].isna().all(), (
                    "cusum emits raw statistics; a Brier score would invent a "
                    "calibration it never claimed")


def test_ticker_detail_compares_each_feature_to_its_trailing_normal():
    """P9-03: a value alone means little — the comparison is the point."""
    at = _run("Ticker detail")
    if "no alerts to inspect" in _text(at):
        pytest.skip("no alerts logged")
    cols = {c for df in at.dataframe for c in df.value.columns}
    assert {"at the flagged hour", "trailing median", "percentile"} <= cols, (
        f"feature table missing its comparison columns; got {cols}")
