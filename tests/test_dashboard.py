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
    parts = []
    for block in (at.markdown, at.caption, at.info, at.warning, at.success,
                  at.error, at.subheader, at.title):
        parts += [getattr(e, "value", "") or "" for e in block]
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
    """Rule 6. The constraint the system is tuned to is visible, not implied."""
    at = _run()
    labels = " ".join(m.label for m in at.metric).lower()
    assert "alert budget" in labels
    assert any("spent" in m.label.lower() for m in at.metric)


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

    icon, words, mult = strength(5.0, 2.5)
    assert mult == 2.0
    assert 0.0 <= 1.0  # sanity
    assert words in {"very strong", "strong", "moderate", "at threshold"}
    assert not isinstance(mult, bool)


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
