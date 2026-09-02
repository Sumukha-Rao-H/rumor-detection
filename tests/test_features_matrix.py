"""P4-11 — assembling the matrix every baseline will be trained on.

Two properties carry this file. The window's right edge must be STRICTLY before
t0, or `days_since_last_8k` announces the event (P4-10). And features must be
computed on each ticker's full history before slicing — computing them on a
48-bar slice would return NaN for every row of every event while looking
perfectly reasonable.
"""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.eval.contract import SCHEMA
from src.pipeline.features import build_matrix, write_matrix
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts


HOUR = 3600


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = {**cfg["paths"], "processed": str(tmp_path / "processed")}
    return cfg


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "mx.db")


def seed(conn, cfg, ticker="AAA", event_id="e1", n_bars=800, t0_offset=700,
         usable=1, items="8.01", start_days=200):
    """A ticker with plenty of history and one usable event inside it."""
    base = date_str_to_ts(cfg["study_window"]["start"]) + start_days * 86400
    iv = cfg["market"]["interval"]
    bench = cfg["market"]["benchmark"]
    rng = np.random.default_rng(4)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n_bars)))
    vols = rng.lognormal(12, 0.3, n_bars)
    db.upsert_bars(conn, [
        (ticker, base + i * HOUR, 0, 0, 0, float(closes[i]), float(vols[i]), iv)
        for i in range(n_bars)])
    db.upsert_bars(conn, [
        (bench, base + i * HOUR, 0, 0, 0, 100.0 + i * 0.01, 1e6, iv)
        for i in range(n_bars)])

    t0 = base + t0_offset * HOUR
    db.upsert_companies(conn, [{"cik": f"CIK{ticker}", "ticker": ticker,
                                "in_universe": 1}])
    db.upsert_filings(conn, [{
        "accession_no": event_id, "cik": f"CIK{ticker}", "ticker": ticker,
        "form": "8-K", "items": items, "acceptance_utc": t0,
        "filing_date_utc": t0}])
    db.upsert_events(conn, [{
        "event_id": event_id, "accession_no": event_id, "ticker": ticker,
        "items": items, "t0_filing_utc": t0, "t0_utc": t0,
        "t0_source": "filing", "is_scheduled": 0, "usable": usable}])
    return t0


# --------------------------------------------------------------------------
# the boundary
# --------------------------------------------------------------------------

def test_no_row_falls_at_or_after_t0(cfg, conn):
    """THE Done-when. At t0 `days_since_last_8k` is 0 — the event, announced."""
    t0 = seed(conn, cfg)
    m = build_matrix(cfg, conn)
    assert (m["ts_utc"] < m["t0_utc"]).all()
    assert (m["ts_utc"] == t0).sum() == 0


def test_the_last_row_is_the_bar_before_t0(cfg, conn):
    """The edge is tight, not merely safe — one bar earlier would lose a
    decision step from every episode."""
    t0 = seed(conn, cfg)
    m = build_matrix(cfg, conn)
    assert m["ts_utc"].max() == t0 - HOUR


def test_stricter_than_the_evaluation_contract(cfg, conn):
    """`contract.py` forbids only hours AFTER t0; the matrix excludes t0 too."""
    seed(conn, cfg)
    m = build_matrix(cfg, conn)
    assert (m["ts_utc"] <= m["t0_utc"]).all()      # the contract's rule
    assert (m["ts_utc"] < m["t0_utc"]).all()       # ours, stricter


# --------------------------------------------------------------------------
# window shape
# --------------------------------------------------------------------------

def test_a_window_has_at_most_horizon_rows(cfg, conn):
    seed(conn, cfg)
    m = build_matrix(cfg, conn)
    assert m.groupby("window_id").size().max() == cfg["decision"]["horizon_hours"]


def test_a_short_history_gives_a_short_window(cfg, conn):
    """Not padded: a fabricated bar is worse than a short episode."""
    seed(conn, cfg, n_bars=30, t0_offset=10)
    m = build_matrix(cfg, conn)
    assert len(m) == 10 < cfg["decision"]["horizon_hours"]


def test_window_id_is_the_event_id(cfg, conn):
    seed(conn, cfg, event_id="0000320193-26-000018")
    m = build_matrix(cfg, conn)
    assert set(m["window_id"]) == {"0000320193-26-000018"}


# --------------------------------------------------------------------------
# the trap that would have failed silently
# --------------------------------------------------------------------------

def test_features_are_computed_on_full_history_not_the_slice(cfg, conn):
    """The real reason features are built per ticker then sliced.

    `volume_z` needs a 480-bar baseline and `volatility` 120. Computing on a
    48-bar slice would return NaN for every row of every event — and would look
    entirely reasonable while doing it.
    """
    seed(conn, cfg, n_bars=800, t0_offset=700)
    m = build_matrix(cfg, conn)
    assert m["volume_z"].notna().all()
    assert m["volatility"].notna().all()


def test_an_event_early_in_the_history_has_nan_baseline_features(cfg, conn):
    """Reported as an explained NaN, not silently dropped."""
    seed(conn, cfg, n_bars=800, t0_offset=100)
    m = build_matrix(cfg, conn)
    assert m["volume_z"].isna().any()
    assert m["ret_1h"].notna().all()      # short-horizon features still work


# --------------------------------------------------------------------------
# schema and population
# --------------------------------------------------------------------------

def test_the_schema_matches_the_evaluation_contract(cfg, conn):
    """No rename may sit between the matrix and the metric."""
    seed(conn, cfg)
    m = build_matrix(cfg, conn)
    identifying = set(SCHEMA) - {"score", "action"}
    assert identifying <= set(m.columns)


def test_only_usable_events_are_built(cfg, conn):
    seed(conn, cfg, ticker="AAA", event_id="keep", usable=1)
    seed(conn, cfg, ticker="BBB", event_id="drop", usable=0)
    m = build_matrix(cfg, conn)
    assert set(m["window_id"]) == {"keep"}


def test_raises_when_no_usable_events_exist(cfg, conn):
    seed(conn, cfg, usable=0)
    with pytest.raises(SystemExit, match="no usable events"):
        build_matrix(cfg, conn)


def test_rerunning_overwrites_rather_than_appends(cfg, conn):
    seed(conn, cfg)
    first = write_matrix(cfg, conn)
    second = write_matrix(cfg, conn)
    assert len(first) == len(second)
    from pathlib import Path
    written = pd.read_parquet(Path(cfg["paths"]["processed"]) / "features.parquet")
    assert len(written) == len(first)


# --------------------------------------------------------------------------
# robustness — data gaps and edge configs that must not crash the whole run
# --------------------------------------------------------------------------

def test_a_ticker_with_zero_bars_is_skipped_not_crashed(cfg, conn):
    """A usable event whose ticker has no bars for the configured interval is
    a plausible data gap (a collector miss, a delisted name) — the run must
    skip it, not raise a KeyError while building an empty frame.
    """
    seed(conn, cfg, ticker="AAA", event_id="keep")

    base = date_str_to_ts(cfg["study_window"]["start"]) + 200 * 86400
    t0 = base + 700 * HOUR
    db.upsert_companies(conn, [{"cik": "CIKBBB", "ticker": "BBB",
                               "in_universe": 1}])
    db.upsert_filings(conn, [{
        "accession_no": "drop", "cik": "CIKBBB", "ticker": "BBB",
        "form": "8-K", "items": "8.01", "acceptance_utc": t0,
        "filing_date_utc": t0}])
    db.upsert_events(conn, [{
        "event_id": "drop", "accession_no": "drop", "ticker": "BBB",
        "items": "8.01", "t0_filing_utc": t0, "t0_utc": t0,
        "t0_source": "filing", "is_scheduled": 0, "usable": 1}])
    # BBB has zero rows in `bars` — no bars were ever inserted for it.

    m = build_matrix(cfg, conn)
    assert set(m["window_id"]) == {"keep"}


def test_every_window_empty_raises_the_clear_systemexit(cfg, conn):
    """t0 at the ticker's very first bar leaves nothing before it. When that
    is true of every event in the run, `pd.concat([])` must not be the
    message the caller sees — the intended `SystemExit` must be.
    """
    seed(conn, cfg, t0_offset=0)
    with pytest.raises(SystemExit, match="feature matrix is empty"):
        build_matrix(cfg, conn)


def test_empty_scheduled_codes_does_not_break_the_earnings_query(cfg, conn):
    """`items.scheduled: []` is a valid, if unusual, config — the earnings SQL
    must not degrade to the syntax error `AND ()`.
    """
    seed(conn, cfg)
    empty_scheduled = {**cfg, "items": {**cfg["items"], "scheduled": []}}
    m = build_matrix(empty_scheduled, conn)
    assert m["days_since_last_earnings"].isna().all()
