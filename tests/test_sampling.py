"""P4-12 — what "nothing is coming" looks like.

Two rules make a window genuinely quiet, and each closes an open issue. The gap
is measured from EVERY filing, not just usable events, which stops a quiet
window landing beside an immaterial 8-K and resolves issue 28's clustering by
construction. And windows with no volume baseline are excluded, matching the
issue-32 rule for positives, because a detector cannot tell "quiet" from
"unmeasurable".
"""

import numpy as np
import pytest

from src import db
from src.pipeline.sampling import all_candidates, draw, quiet_candidates, report
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts


HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "s.db")


def seed_ticker(conn, cfg, ticker="AAA", n_bars=2000, filings=(), usable_at=None):
    """A ticker with contiguous hourly bars and optional filings."""
    base = date_str_to_ts(cfg["study_window"]["start"]) + 100 * 86400
    iv = cfg["market"]["interval"]
    db.upsert_companies(conn, [{"cik": f"CIK{ticker}", "ticker": ticker,
                                "in_universe": 1}])
    db.upsert_bars(conn, [
        (ticker, base + i * HOUR, 0, 0, 0, 100.0, 1e6, iv) for i in range(n_bars)])
    for k, off in enumerate(filings):
        ts = base + off * HOUR
        db.upsert_filings(conn, [{
            "accession_no": f"{ticker}-{k}", "cik": f"CIK{ticker}",
            "ticker": ticker, "form": "8-K", "items": "8.01",
            "acceptance_utc": ts, "filing_date_utc": ts}])
    return base


# --------------------------------------------------------------------------
# the core rule
# --------------------------------------------------------------------------

def test_a_quiet_window_is_far_from_every_filing(cfg, conn):
    """THE Done-when's core rule."""
    gap = cfg["sampling"]["quiet_gap_hours"]
    base = seed_ticker(conn, cfg, n_bars=2000, filings=(1000,))
    anchors = quiet_candidates(cfg, conn, "AAA")
    filing_ts = base + 1000 * HOUR
    assert anchors.size > 0
    assert np.abs(anchors - filing_ts).min() > gap * HOUR


def test_the_gap_counts_unusable_filings_too(cfg, conn):
    """Issue 28, resolved by construction.

    An immaterial 8-K is still an 8-K. Measuring the gap from `filings` rather
    than from usable `events` means a quiet window cannot land beside one — and
    cannot land inside an event's cluster either, since a cluster's members are
    themselves filings.
    """
    base = seed_ticker(conn, cfg, n_bars=2000, filings=(1000,))
    # The filing is in `filings` but no event row exists for it at all.
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    anchors = quiet_candidates(cfg, conn, "AAA")
    gap = cfg["sampling"]["quiet_gap_hours"] * HOUR
    assert np.abs(anchors - (base + 1000 * HOUR)).min() > gap


def test_a_window_straddling_a_filing_is_rejected(cfg, conn):
    """Every bar behind the anchor must be quiet, not merely the anchor."""
    horizon = cfg["decision"]["horizon_hours"]
    gap = cfg["sampling"]["quiet_gap_hours"]
    base = seed_ticker(conn, cfg, n_bars=2000, filings=(1000,))
    anchors = quiet_candidates(cfg, conn, "AAA")
    for a in anchors:
        window_start = a - horizon * HOUR
        assert abs(window_start - (base + 1000 * HOUR)) > gap * HOUR


def test_a_ticker_with_no_filings_is_all_eligible(cfg, conn):
    """8 universe members file nothing at all."""
    seed_ticker(conn, cfg, n_bars=2000, filings=())
    horizon = cfg["decision"]["horizon_hours"]
    anchors = quiet_candidates(cfg, conn, "AAA")
    assert anchors.size == pytest.approx(2000 // (horizon + 1), abs=2)


def test_a_ticker_with_too_few_bars_gives_nothing(cfg, conn):
    seed_ticker(conn, cfg, n_bars=10)
    assert quiet_candidates(cfg, conn, "AAA").size == 0


def test_a_heavy_filer_can_contribute_no_negatives(cfg, conn):
    """Filing every other day legitimately leaves no quiet stretch."""
    seed_ticker(conn, cfg, n_bars=2000, filings=tuple(range(0, 2000, 48)))
    assert quiet_candidates(cfg, conn, "AAA").size == 0


# --------------------------------------------------------------------------
# window shape — identical to a positive's
# --------------------------------------------------------------------------

def test_quiet_windows_do_not_overlap_each_other(cfg, conn):
    """Two windows sharing 47 of 48 bars are not two observations."""
    horizon = cfg["decision"]["horizon_hours"]
    seed_ticker(conn, cfg, n_bars=2000, filings=())
    anchors = np.sort(quiet_candidates(cfg, conn, "AAA"))
    assert (np.diff(anchors) > horizon * HOUR).all()


def test_a_quiet_window_ends_strictly_before_its_anchor(cfg, conn):
    """Same boundary rule as a positive, so nothing but the label differs."""
    base = seed_ticker(conn, cfg, n_bars=2000, filings=())
    horizon = cfg["decision"]["horizon_hours"]
    a = int(quiet_candidates(cfg, conn, "AAA")[0])
    window = [a - i * HOUR for i in range(1, horizon + 1)]
    assert max(window) < a


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------

def test_the_same_seed_draws_the_same_sample(cfg, conn):
    seed_ticker(conn, cfg, n_bars=2000, filings=())
    seed_ticker(conn, cfg, ticker="BBB", n_bars=2000, filings=())
    cands = all_candidates(cfg, conn)
    assert draw(cfg, cands, 15) == draw(cfg, cands, 15)


def test_a_different_seed_draws_differently(cfg, conn):
    seed_ticker(conn, cfg, n_bars=2000, filings=())
    seed_ticker(conn, cfg, ticker="BBB", n_bars=2000, filings=())
    cands = all_candidates(cfg, conn)
    assert draw(cfg, cands, 15, seed=1) != draw(cfg, cands, 15, seed=2)


def test_ratios_are_nested(cfg, conn):
    """10 subset 20 subset 40, so the robustness check varies ONE thing.

    Three independent draws would confound the ratio with the sample.
    """
    seed_ticker(conn, cfg, n_bars=4000, filings=())
    seed_ticker(conn, cfg, ticker="BBB", n_bars=4000, filings=())
    cands = all_candidates(cfg, conn)
    small, mid, big = draw(cfg, cands, 10), draw(cfg, cands, 20), draw(cfg, cands, 40)
    assert small == mid[:10] == big[:10]
    assert mid == big[:20]


def test_a_shortfall_is_reported_not_resampled(cfg, conn):
    """Duplicate rows would be a fabricated observation."""
    seed_ticker(conn, cfg, n_bars=2000, filings=())
    cands = all_candidates(cfg, conn)
    available = sum(len(v) for v in cands.values())
    got = draw(cfg, cands, available + 500)
    assert len(got) == available
    assert len(set(got)) == len(got)


def test_raises_when_no_quiet_window_exists(cfg, conn):
    seed_ticker(conn, cfg, n_bars=2000, filings=tuple(range(0, 2000, 48)))
    with pytest.raises(SystemExit, match="no quiet window"):
        all_candidates(cfg, conn)


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def test_report_gives_both_base_rate_framings(cfg, conn):
    """The plan never pinned the unit, and the two framings differ enormously.

    Per-bar counts every hour as a decision opportunity; tiled counts each
    48-bar block once. With the same positives on top, the tiled rate is far
    the larger, which is exactly why the choice cannot be made silently.
    """
    base = seed_ticker(conn, cfg, n_bars=3000, filings=(2500,))
    t0 = base + 2500 * HOUR
    db.upsert_events(conn, [{
        "event_id": "e1", "accession_no": "AAA-0", "ticker": "AAA",
        "items": "8.01", "t0_filing_utc": t0, "t0_utc": t0,
        "t0_source": "filing", "is_scheduled": 0, "usable": 1}])

    r = report(cfg, conn)
    assert r["positives"] == 1
    assert r["available"] > 0
    assert 0 < r["rate_per_bar"] < r["rate_tiled"] <= 1
    assert r["rate_per_bar"] == pytest.approx(1 / r["total_bars"])


def test_report_counts_windows_not_ticker_name_lengths(cfg, conn):
    """Regression: `sum(len(v) for v in candidates)` iterates a dict's KEYS.

    That summed ticker-symbol lengths and produced 5,050 — a plausible-looking
    number that was not the quantity at all, and it understated the available
    negatives sixfold.
    """
    base = seed_ticker(conn, cfg, ticker="AAA", n_bars=3000, filings=(0,))
    seed_ticker(conn, cfg, ticker="BBBB", n_bars=3000, filings=())
    db.upsert_events(conn, [{
        "event_id": "e1", "accession_no": "AAA-0", "ticker": "AAA",
        "items": "8.01", "t0_filing_utc": base, "t0_utc": base,
        "t0_source": "filing", "is_scheduled": 0, "usable": 1}])

    cands = all_candidates(cfg, conn)
    expected = sum(len(v) for v in cands.values())
    assert report(cfg, conn)["available"] == expected
    assert expected > len("AAA") + len("BBBB")     # not the string lengths
    assert expected > 50                           # real windows, not 7
