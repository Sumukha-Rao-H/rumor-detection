"""P4-13 — leakage checked against the composed pipeline, not just the parts.

P2-08 and P4-06…P4-10 test each builder alone. Three failure modes survive
that: a leak entering in how the five are *composed*, an off-by-one in how
windows are *sliced* out of full history, and one ticker's features being built
from another ticker's bars — which produces perfectly ordered timestamps and
completely wrong values, and which no forward-looking check can ever see.

The central test tampers with the future and demands the matrix come back
byte-identical. That covers slicing, composition and boundary arithmetic at
once, without knowing anything about how any feature is implemented.
"""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.pipeline import features as F
from src.pipeline.features import build_matrix, ticker_features
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts
from tests.test_leakage import assert_no_lookahead, find_lookahead


HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "leak.db")


def seed(conn, cfg, tickers=("AAA", "BBB"), n_bars=900, t0_bar=800):
    """A realistic multi-ticker database with one usable event each.

    Deliberately different price levels per ticker: if features were ever built
    from the wrong ticker's bars, the values would be obviously wrong rather
    than plausibly wrong.
    """
    base = date_str_to_ts(cfg["study_window"]["start"]) + 120 * 86400
    iv = cfg["market"]["interval"]
    bench = cfg["market"]["benchmark"]
    rng = np.random.default_rng(17)

    db.upsert_bars(conn, [
        (bench, base + i * HOUR, 0, 0, 0, 400.0 + i * 0.01, 5e6, iv)
        for i in range(n_bars)])

    t0s = {}
    for k, t in enumerate(tickers):
        level = 100.0 * (k + 1) * 10          # 1000, 2000 — unmistakable
        closes = level * np.exp(np.cumsum(rng.normal(0, 0.004, n_bars)))
        vols = rng.lognormal(12 + k, 0.3, n_bars)
        db.upsert_bars(conn, [
            (t, base + i * HOUR, 0, 0, 0, float(closes[i]), float(vols[i]), iv)
            for i in range(n_bars)])

        t0 = base + t0_bar * HOUR
        t0s[t] = t0
        db.upsert_companies(conn, [{"cik": f"CIK{t}", "ticker": t,
                                    "in_universe": 1}])
        db.upsert_filings(conn, [{
            "accession_no": f"{t}-e", "cik": f"CIK{t}", "ticker": t,
            "form": "8-K", "items": "8.01", "acceptance_utc": t0,
            "filing_date_utc": t0}])
        db.upsert_events(conn, [{
            "event_id": f"{t}-e", "accession_no": f"{t}-e", "ticker": t,
            "items": "8.01", "t0_filing_utc": t0, "t0_utc": t0,
            "t0_source": "filing", "is_scheduled": 0, "usable": 1}])
    return t0s


def tamper(conn, cfg, from_ts, price=3.0, volume=7.0):
    """Multiply every bar at or after `from_ts`, for every ticker."""
    conn.execute(
        "UPDATE bars SET close = close * ?, volume = volume * ? "
        "WHERE ts_utc >= ? AND interval = ?",
        (price, volume, from_ts, cfg["market"]["interval"]))
    conn.commit()


def frame_of(conn, cfg, ticker):
    rows = conn.execute(
        "SELECT ts_utc, close, volume FROM bars WHERE ticker = ? AND "
        "interval = ? ORDER BY ts_utc",
        (ticker, cfg["market"]["interval"])).fetchall()
    return pd.DataFrame([dict(r) for r in rows]).set_index("ts_utc")


# --------------------------------------------------------------------------
# the composition
# --------------------------------------------------------------------------

def test_composed_features_pass_the_detector(cfg, conn):
    """All five builders together, not one at a time."""
    seed(conn, cfg)
    bench = frame_of(conn, cfg, cfg["market"]["benchmark"])
    filings = np.array([date_str_to_ts(cfg["study_window"]["start"])],
                       dtype=np.int64)
    assert_no_lookahead(
        lambda f: ticker_features(f, bench, filings, filings, cfg),
        frame_of(conn, cfg, "AAA"))


def test_a_centred_window_in_the_composition_is_caught(cfg, monkeypatch):
    """THE second Done-when, and the guard on the guard.

    One of the five builders is swapped for a centred-window version. A passing
    suite means nothing unless a broken pipeline actually fails.
    """
    def leaky(frame, cfg=None):
        return pd.DataFrame(
            {"volatility": frame["close"].rolling(9, center=True).std()},
            index=frame.index)

    monkeypatch.setattr(F, "realised_volatility", leaky)
    rng = np.random.default_rng(2)
    idx = pd.Index([1_762_000_000 + i * HOUR for i in range(400)], name="ts_utc")
    frame = pd.DataFrame({"close": 100 + np.cumsum(rng.normal(0, .5, 400)),
                          "volume": rng.lognormal(12, .3, 400)}, index=idx)
    bench = frame.copy()
    empty = np.array([], dtype=np.int64)

    assert find_lookahead(
        lambda f: F.ticker_features(f, bench, empty, empty, cfg=load_config()),
        frame)


# --------------------------------------------------------------------------
# the assembled matrix — the Done-when
# --------------------------------------------------------------------------

def test_the_matrix_is_unchanged_by_tampering_with_the_future(cfg, conn):
    """THE test. If any row depends on a bar at or after its own t0, this fails.

    Covers slicing, composition and boundary arithmetic together, and knows
    nothing about how any feature is implemented.
    """
    t0s = seed(conn, cfg)
    before = build_matrix(cfg, conn)

    tamper(conn, cfg, min(t0s.values()))
    after = build_matrix(cfg, conn)

    pd.testing.assert_frame_equal(before, after)


def test_tampering_with_volume_after_t0_changes_nothing(cfg, conn):
    """`volume_z` reads volume alone; a price-only check would miss it."""
    t0s = seed(conn, cfg)
    before = build_matrix(cfg, conn)
    tamper(conn, cfg, min(t0s.values()), price=1.0, volume=50.0)
    pd.testing.assert_frame_equal(before, build_matrix(cfg, conn))


def test_tampering_before_t0_does_change_the_matrix(cfg, conn):
    """Proves the test above can fail. Without this, an inert check would look
    like a clean bill of health."""
    t0s = seed(conn, cfg)
    before = build_matrix(cfg, conn)
    tamper(conn, cfg, min(t0s.values()) - 40 * HOUR)
    after = build_matrix(cfg, conn)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(before, after)


# --------------------------------------------------------------------------
# the failure no forward-looking check can see
# --------------------------------------------------------------------------

def test_features_are_not_contaminated_across_tickers(cfg, conn):
    """A frame built from the wrong ticker gives ordered timestamps and wrong
    values — invisible to any leakage detector, so it needs its own test.

    The fixture prices AAA near 1000 and BBB near 2000, so a swap is stark.
    """
    seed(conn, cfg)
    m = build_matrix(cfg, conn)
    bench = frame_of(conn, cfg, cfg["market"]["benchmark"])
    empty = np.array([], dtype=np.int64)

    for ticker in ("AAA", "BBB"):
        own = ticker_features(frame_of(conn, cfg, ticker), bench, empty,
                              empty, cfg)
        rows = m[m["ticker"] == ticker]
        expected = own.loc[rows["ts_utc"].to_numpy(), "ret_1h"].to_numpy()
        np.testing.assert_allclose(rows["ret_1h"].to_numpy(), expected,
                                   equal_nan=True)


def test_no_matrix_row_is_at_or_after_its_t0(cfg, conn):
    """The boundary once more, end to end rather than in isolation."""
    seed(conn, cfg)
    m = build_matrix(cfg, conn)
    assert (m["ts_utc"] < m["t0_utc"]).all()
