"""End-to-end smoke test for the evaluation stack.

Every piece is tested on its own elsewhere. This covers what those cannot:
that the pieces compose, that the extremes survive, and that the plan's
"metric before model" requirement actually held.

That last one is usually an unverifiable promise about the order someone worked
in. Here it is checkable: the model packages are empty, and `src/eval/` imports
nothing from them.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pandas as pd
import pytest

from src.eval.contract import actions_from_scores, validate_predictions
from src.eval.metrics import (
    calibration_summary,
    detection_delay_summary,
    precision_at_alert_budget,
)
from src.eval.report import COLUMNS, report_table, t0_variant_gap
from src.eval.synthetic import make_synthetic_predictions

REPO = Path(__file__).resolve().parents[1]
DENSE = dict(n_positive=60, n_quiet=1500, n_tickers=8, span_days=180)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_synthetic_predictions(**DENSE, signal_strength=2.0, seed=5)


# --- the Done-when -------------------------------------------------------


def test_full_chain_runs_and_renders(frame) -> None:
    """contract -> budget -> threshold -> actions -> delay -> calibration -> table.

    Each stage consumes the previous stage's output, which is the thing
    per-piece tests cannot check.
    """
    validated = validate_predictions(frame)
    budget = precision_at_alert_budget(validated)
    decided = actions_from_scores(validated, budget.threshold)
    delay = detection_delay_summary(decided)
    table = report_table(frame)

    # the stages agree with each other
    assert delay.n_detections == budget.true_positives

    # and it renders — a table that computes but cannot be printed is not a report
    rendered = table.to_string(index=False)
    assert "precision" in rendered
    assert len(rendered.splitlines()) == len(table) + 1


# --- proving "metric before model" ---------------------------------------


def test_no_model_exists_yet() -> None:
    """Phase-bound, and deliberately so.

    The plan requires the evaluation code to be written before the first model,
    so the headline metric cannot be chosen after seeing which one flatters a
    result. Right now that is literally true. Phase 5 will fill these packages
    and this assertion will be retired — the git history is what preserves the
    claim afterwards.
    """
    for package in ("baselines", "rl", "pipeline"):
        modules = [p for p in (REPO / "src" / package).glob("*.py")
                   if p.name != "__init__.py"]
        assert modules == [], f"src/{package} is no longer empty: {modules}"


def test_eval_does_not_import_models_network_or_db() -> None:
    """Permanent architectural invariant, not a phase-bound one.

    `architecture-context.md` fixes the dependency direction: evaluation sits
    downstream of the models and never reaches back, and it never touches the
    network or the database. Keeps holding after Phase 5.
    """
    forbidden = {"src.baselines", "src.rl", "src.pipeline", "src.db",
                 "sqlite3", "requests", "yfinance", "finnhub"}
    for path in (REPO / "src" / "eval").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                assert name not in forbidden and root not in forbidden, (
                    f"{path.name} imports {name!r}"
                )


# --- the two extremes ----------------------------------------------------


def oracle(frame: pd.DataFrame) -> pd.DataFrame:
    """A model that knows the answer: 1.0 on positives, 0.0 on quiet windows."""
    out = frame.copy()
    out["score"] = out["t0_utc"].notna().astype("float64")
    return out


def test_perfect_oracle_scores_perfectly(frame) -> None:
    """Pins the top of the range.

    If the metrics cannot report a perfect model as perfect, nothing they say
    about an imperfect one can be trusted.
    """
    n_positive = int(validate_predictions(frame)
                     .groupby("window_id")["t0_utc"].first().notna().sum())
    r = precision_at_alert_budget(oracle(frame), max_alerts=n_positive)
    assert r.precision == 1.0
    assert r.recall == 1.0


def test_oracle_is_perfectly_calibrated(frame) -> None:
    cal = calibration_summary(oracle(frame))
    assert cal.brier == pytest.approx(0.0)
    assert cal.ece == pytest.approx(0.0)
    assert cal.brier_skill_score == pytest.approx(1.0)


def test_degenerate_policy_survives_every_metric(frame) -> None:
    """Pins the bottom of the range.

    Always-WAIT is a required baseline. It must flow through every metric
    without erroring, and be visibly degenerate at the end.
    """
    table = report_table(frame, max_alerts=0)
    assert table["degenerate"].all()

    delay = detection_delay_summary(frame)          # frames are all-WAIT
    assert delay.n_detections == 0
    assert math.isnan(delay.median_trading_hours)
    assert delay.n_missed == delay.n_positive

    budget = precision_at_alert_budget(frame, max_alerts=0)
    assert math.isnan(budget.precision)


# --- consistency and determinism -----------------------------------------


def test_numbers_are_internally_consistent(frame) -> None:
    table = report_table(frame)
    row = table[table["slice"] == "all"].iloc[0]
    assert row["n_missed"] == row["n_positive"] - round(row["recall"] * row["n_positive"])
    assert 0.0 <= row["precision"] <= 1.0
    assert 0.0 <= row["recall"] <= 1.0
    assert row["median_lead_trading_h"] <= row["median_lead_wall_h"]
    assert row["base_rate"] == pytest.approx(row["n_positive"] / row["n_windows"])


def test_chain_is_deterministic() -> None:
    """Same seed, identical table. A report that moves between runs cannot be
    defended in a viva."""
    a = report_table(make_synthetic_predictions(**DENSE, signal_strength=2.0, seed=5))
    b = report_table(make_synthetic_predictions(**DENSE, signal_strength=2.0, seed=5))
    pd.testing.assert_frame_equal(a, b)


def test_both_t0_variants_and_the_gap() -> None:
    """The plan requires both variants reported, with the gap between them."""
    filing = make_synthetic_predictions(**DENSE, signal_strength=2.0, seed=5)
    news = make_synthetic_predictions(**DENSE, signal_strength=1.2, seed=5)
    table = report_table({"filing": filing, "news_adjusted": news})
    assert set(table["t0_variant"]) == {"filing", "news_adjusted"}
    assert list(table.columns) == COLUMNS
    assert not math.isnan(t0_variant_gap(table))
