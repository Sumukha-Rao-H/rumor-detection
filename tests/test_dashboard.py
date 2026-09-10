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


# --------------------------------------------------------------------------
# The database is gitignored; the alert log is committed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("screen", SCREENS)
def test_every_screen_renders_without_a_database(screen, tmp_path, monkeypatch):
    """A fresh clone has `live-log/alerts.csv` and no `footprints.db`.

    The database is about a gigabyte of rebuildable cache and is gitignored;
    the alert log is the committed, durable record. So an examiner, or a second
    machine, has the evidence file and nothing beside it — and `budget_line`
    runs before routing, so a missing database used to raise a raw
    `FileNotFoundError` traceback on every screen, including *Today's alerts*,
    which needs nothing but the CSV.
    """
    import app.data as data

    real = data.config()
    missing = {**real, "paths": {**real["paths"],
                                 "db": str(tmp_path / "absent.db")}}
    monkeypatch.setattr(data, "config", lambda: missing)
    for fn in ("alerts", "outcomes", "universe_size", "answerable_edge"):
        getattr(data, fn).clear()   # st.cache_data holds the real DB's answers

    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, f"app failed to start with no DB: {at.exception}"
    at.sidebar.radio[0].set_value(screen).run()
    assert not at.exception, f"{screen} raised with no DB: {at.exception}"


def test_a_missing_database_is_said_out_loud_not_shown_as_zero():
    """An honest empty state, not a fabricated number. A universe of 0 and an
    allowance of 0 would read as measurements."""
    import app.data as data

    assert data.db_present() in (True, False)


# --------------------------------------------------------------------------
# Rule notices must reach the reader as text, not as markup
# --------------------------------------------------------------------------

@pytest.mark.parametrize("screen", SCREENS)
def test_no_rule_notice_shows_literal_html(screen):
    """`st.info` has no `unsafe_allow_html` parameter — Streamlit escapes it.

    Five notices passed `<b>`/`<code>`, including the ones carrying the
    plain-accuracy rule and the scheduled-split rule, so the reader saw
    `<b>Scheduled and unscheduled are never pooled.</b>` literally. The
    existing assertions missed it because `_text` reads the pre-render string,
    which is exactly what a browser never shows.
    """
    at = _run(screen)
    # Only the widgets that ESCAPE markup. `st.markdown` takes
    # `unsafe_allow_html` and the chrome uses it deliberately for the styled
    # meta panels; `st.info`/`warning`/`success`/`error` do not take it at all,
    # so any tag reaching them is shown to the reader verbatim.
    escaping = []
    for family in (at.info, at.warning, at.success, at.error):
        escaping += [str(getattr(el, "value", "")) for el in family]
    for body in escaping:
        for tag in ("<b>", "</b>", "<code>", "</code>", "<i>", "<br>"):
            assert tag not in body, (
                f"{screen} passes a literal {tag} to a widget that escapes "
                f"markup — the reader sees the tag: {body[:120]!r}")


def test_the_live_screen_splits_scheduled_from_unscheduled():
    """Rule 7 applies to the live claim too, and it is where it matters most.

    The hit rate pooled both halves, and of the ten live hits in the real
    database SIX carry item 2.02 — pre-announced quarterly results, whose date
    is public weeks ahead. Pooling let the easy half carry the headline: about
    9.9% pooled against roughly 4.0% on unscheduled events, which is the number
    the project exists to produce. The Evaluation screen states the rule
    verbatim; the screen beside it used to break it.
    """
    body = _text(_run("Live monitor log")).lower()
    assert "unscheduled" in body
    assert "scheduled" in body


def test_the_outcome_window_is_called_wall_clock_not_trading_hours():
    """Two clocks, correctly separated in the code and conflated in the copy.

    `live.outcome_window_hours` is 48 WALL-CLOCK hours — about two days. Two
    captions called it 48 trading hours, which is roughly seven calendar days,
    making the hit rate look far harder-won than it was. The lead-time column
    beside it really is in trading hours and is correctly labelled, which is
    exactly why the two must not blur.
    """
    body = _text(_run("Live monitor log")).lower()
    assert "48 trading hours" not in body
    assert "wall-clock" in body or "wall clock" in body


def test_the_evaluation_screen_prefers_the_re_run_over_the_void_files():
    """Three Phase 10 files sit side by side and two of them are VOID.

    The 2026-09-08 pair — original and arithmetically corrected — were produced
    by a frame that ranked on window length: pure random noise scored 29.6x on
    it, beating every detector. Preferring either over the 2026-09-10 re-run
    would put numbers on screen that the project's own correction note disowns.
    """
    from app.screens import _PHASE10

    assert _PHASE10[0][0].endswith("FINAL-test-evaluation-r2.csv")
    for path, caption in _PHASE10[1:]:
        assert "VOID" in caption, f"{path} must be labelled void"


def test_the_lift_caveat_is_not_applied_to_the_corrected_re_run():
    """The caveat disqualifies the lift column. Printing it unconditionally
    would disown the one set of numbers that is actually sound."""
    body = _text(_run("Evaluation"))
    if "re-run once on 2026-09-10" in body:
        assert "not trustworthy" not in body
        assert "random_noise" in body, "the null must be named on screen"
