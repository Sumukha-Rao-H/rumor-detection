"""P5-03 — quiet windows must be indistinguishable from positives but for the label.

`sampling.py` states the requirement and the reason: *"Quiet windows are built
in the same shape as positives ... Any structural difference would be
something a model could learn instead of the market."* These tests are that
sentence made enforceable — same columns, same bar count, same strictly-before
boundary, and label columns null together.
"""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.pipeline.features import build_matrix, build_quiet_matrix
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path, cfg):
    """A ticker with enough history for volume_z to be defined, plus SPY."""
    c = db.get_conn(tmp_path / "quiet.db")
    iv = cfg["market"]["interval"]
    start = date_str_to_ts("2025-09-01")
    n = 900                                   # > volume_zscore_window_h + slack

    for ticker in ("AAA", cfg["market"]["benchmark"]):
        bars = []
        for i in range(n):
            ts = start + i * HOUR
            bars.append((ticker, ts, 100.0, 101.0, 99.0,
                         100.0 + (i % 7) * 0.1, 1_000_000 + (i % 13) * 1000, iv))
        db.upsert_bars(c, bars)

    db.upsert_companies(c, [{"cik": "C1", "ticker": "AAA", "in_universe": 1}])
    db.upsert_filings(c, [{
        "accession_no": "A-1", "cik": "C1", "ticker": "AAA", "form": "8-K",
        "items": "8.01", "acceptance_utc": start + 800 * HOUR,
        "filing_date_utc": start + 800 * HOUR}])
    db.upsert_events(c, [{
        "event_id": "E1", "accession_no": "A-1", "ticker": "AAA",
        "items": "8.01", "t0_filing_utc": start + 800 * HOUR,
        "t0_utc": start + 800 * HOUR, "t0_source": "filing",
        "is_scheduled": 0, "usable": 1, "exclude_reason": None}])
    return c


def test_quiet_windows_have_the_positive_shape(cfg, conn):
    """Same columns, same bar count. Anything else is learnable structure."""
    positives = build_matrix(cfg, conn)
    anchor = date_str_to_ts("2025-09-01") + 700 * HOUR
    quiet = build_quiet_matrix(cfg, conn, [("AAA", anchor)])

    assert list(quiet.columns) == list(positives.columns)
    assert len(quiet) == cfg["decision"]["horizon_hours"]
    assert len(quiet) == len(positives[positives.window_id == "E1"])


def test_quiet_windows_are_strictly_before_their_anchor(cfg, conn):
    """The same `side="left"` boundary positives use. A bar AT the anchor would
    be the negative equivalent of scoring at t0."""
    anchor = date_str_to_ts("2025-09-01") + 700 * HOUR
    quiet = build_quiet_matrix(cfg, conn, [("AAA", anchor)])
    assert (quiet["ts_utc"] < anchor).all()


def test_label_columns_are_null_together(cfg, conn):
    """The contract requires t0, is_scheduled and item_code to be null exactly
    on quiet windows — null t0 IS the negative label."""
    anchor = date_str_to_ts("2025-09-01") + 700 * HOUR
    quiet = build_quiet_matrix(cfg, conn, [("AAA", anchor)])

    assert quiet["t0_utc"].isna().all()
    assert quiet["is_scheduled"].isna().all()
    assert quiet["item_code"].isna().all()


def test_window_ids_cannot_collide_with_event_ids(cfg, conn):
    """Positives and negatives are concatenated before evaluation; an id clash
    would silently merge a positive and a negative into one window."""
    positives = build_matrix(cfg, conn)
    anchor = date_str_to_ts("2025-09-01") + 700 * HOUR
    quiet = build_quiet_matrix(cfg, conn, [("AAA", anchor)])

    assert quiet["window_id"].str.startswith("quiet:").all()
    assert set(quiet["window_id"]) & set(positives["window_id"]) == set()


def test_several_anchors_produce_several_distinct_windows(cfg, conn):
    base = date_str_to_ts("2025-09-01")
    pairs = [("AAA", base + 600 * HOUR), ("AAA", base + 700 * HOUR)]
    quiet = build_quiet_matrix(cfg, conn, pairs)
    assert quiet["window_id"].nunique() == 2


def test_a_ticker_without_bars_is_skipped_not_fatal(cfg, conn, capsys):
    """A shortfall must be reported: unmentioned, a 3:1 sample quietly stops
    being 3:1 while still calling itself that."""
    base = date_str_to_ts("2025-09-01")
    quiet = build_quiet_matrix(cfg, conn, [("AAA", base + 700 * HOUR),
                                           ("NOPE", base + 700 * HOUR)])
    assert quiet["window_id"].nunique() == 1
    assert "skipped" in capsys.readouterr().out


def test_an_anchor_before_the_history_is_skipped(cfg, conn, capsys):
    quiet = build_quiet_matrix(cfg, conn,
                               [("AAA", date_str_to_ts("2020-01-01"))])
    assert quiet.empty
    assert "skipped" in capsys.readouterr().out


def test_no_pairs_returns_empty_rather_than_raising(cfg, conn):
    """Drawing zero negatives is a legitimate configuration, unlike an events
    table with no usable rows."""
    assert build_quiet_matrix(cfg, conn, []).empty


def test_volume_z_is_defined_on_quiet_windows(cfg, conn):
    """The whole point of computing features over full history then slicing:
    a per-window computation would return NaN for every row and look fine."""
    anchor = date_str_to_ts("2025-09-01") + 700 * HOUR
    quiet = build_quiet_matrix(cfg, conn, [("AAA", anchor)])
    assert np.isfinite(quiet["volume_z"]).any()
