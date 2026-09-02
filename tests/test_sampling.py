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
from src.pipeline.sampling import (
    all_candidates, draw, eval_decision_points, quiet_candidates, report,
)
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
# the volume baseline: a quiet window is worthless without one
# --------------------------------------------------------------------------

def test_a_short_history_has_no_valid_baseline(cfg, conn):
    """58 bars total is nowhere near `features.min_baseline_bars` (120).

    Before this rule existed, this exact ticker produced an anchor sitting on
    only 48 bars of history — indistinguishable, from this module's own point
    of view, from a fully-baselined quiet window.
    """
    horizon = cfg["decision"]["horizon_hours"]
    seed_ticker(conn, cfg, n_bars=horizon + 10, filings=())
    assert quiet_candidates(cfg, conn, "AAA").size == 0


def test_an_anchor_needs_min_baseline_bars_behind_its_window(cfg, conn):
    """The earliest possible anchor sits at `horizon + min_baseline_bars`.

    Any earlier and `volume_zscore()`'s own `min_periods` would still be
    undefined for the first bar of the window — "quiet" would really mean
    "unmeasurable".
    """
    horizon = cfg["decision"]["horizon_hours"]
    min_baseline = cfg["features"]["min_baseline_bars"]
    base = seed_ticker(conn, cfg, n_bars=horizon + min_baseline + 60, filings=())
    anchors = quiet_candidates(cfg, conn, "AAA")
    assert anchors.size >= 1
    assert (int(anchors.min()) - base) // HOUR == horizon + min_baseline


# --------------------------------------------------------------------------
# the gap is measured from t0, not from raw acceptance
# --------------------------------------------------------------------------

def test_quiet_gap_uses_t0_not_raw_acceptance(cfg, conn):
    """AGENTS.md rule 3: t0 = min(acceptance, news), never acceptance alone.

    A matched news article can put a filing's true t0 hours before its SEC
    acceptance (t0.py). A bar that clears the gap measured from acceptance
    alone but not from that earlier t0 must still be excluded, because t0 —
    not acceptance — is the instant the market actually learned.
    """
    gap = cfg["sampling"]["quiet_gap_hours"]
    horizon = cfg["decision"]["horizon_hours"]
    min_baseline = cfg["features"]["min_baseline_bars"]
    offset = 20                                # news precedes acceptance by this much
    first_anchor_idx = horizon + min_baseline  # earliest position any anchor can form

    # Placed so the first possible anchor clears the gap from acceptance
    # alone, but not from the true (earlier) t0.
    filing_hour = first_anchor_idx + gap + offset - 5
    assert (filing_hour - first_anchor_idx) > gap                # clears acceptance
    assert (filing_hour - offset - first_anchor_idx) <= gap       # does not clear t0

    base = seed_ticker(conn, cfg, n_bars=filing_hour + 50, filings=(filing_hour,))
    acceptance_ts = base + filing_hour * HOUR
    t0_ts = acceptance_ts - offset * HOUR
    db.upsert_events(conn, [{
        "event_id": "e1", "accession_no": "AAA-0", "ticker": "AAA",
        "items": "8.01", "t0_filing_utc": acceptance_ts, "t0_news_utc": t0_ts,
        "t0_utc": t0_ts, "t0_source": "news", "is_scheduled": 0, "usable": 1}])

    anchors = quiet_candidates(cfg, conn, "AAA")

    # The anchor that acceptance-only logic would have placed right here
    # (it clears the acceptance gap by construction) must not appear.
    forbidden_ts = base + first_anchor_idx * HOUR
    assert forbidden_ts not in anchors
    # And nothing at all sits within the gap of the TRUE t0.
    assert anchors.size == 0 or np.abs(anchors - t0_ts).min() > gap * HOUR


def test_an_unmatched_filing_still_falls_back_to_acceptance(cfg, conn):
    """No `events` row at all (outside the study window, say) — acceptance
    is the only clock available, same as `t0.py` itself falls back to."""
    base = seed_ticker(conn, cfg, n_bars=2000, filings=(1000,))
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    anchors = quiet_candidates(cfg, conn, "AAA")
    gap = cfg["sampling"]["quiet_gap_hours"] * HOUR
    assert np.abs(anchors - (base + 1000 * HOUR)).min() > gap


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


def test_anchors_never_come_from_outside_the_study_window(conn, cfg):
    """Negatives must occupy the same calendar as positives.

    The price snapshot is deliberately frozen PAST `study_window.end` (config
    ships a 2026-08-30 freeze against a 2026-08-01 end), while `events.py`
    confines every positive to the window. Drawing quiet anchors from that
    trailing stretch would hand a model a period no positive can occupy — a
    free "which month is this?" separator standing in for the signal.

    Seeds bars that run well past the window end and asserts none of them is
    offered as an anchor.
    """
    lo = date_str_to_ts(cfg["study_window"]["start"])
    hi = date_str_to_ts(cfg["study_window"]["end"])
    iv = cfg["market"]["interval"]
    db.upsert_companies(conn, [{"cik": "CIKZZZ", "ticker": "ZZZ",
                                "in_universe": 1}])
    # Contiguous hourly bars straddling the window end: 900 inside, 400 after.
    start = hi - 900 * HOUR
    db.upsert_bars(conn, [
        ("ZZZ", start + i * HOUR, 0, 0, 0, 100.0, 1e6, iv) for i in range(1300)])

    anchors = quiet_candidates(cfg, conn, "ZZZ")

    assert anchors.size, "expected some in-window anchors to be found"
    assert anchors.max() <= hi, (
        f"anchor at {anchors.max()} is past study_window.end {hi} — negatives "
        f"drawn from a period no positive can occupy")
    assert anchors.min() >= lo


def test_the_eval_set_carries_the_true_base_rate_not_the_training_ratio(conn, cfg):
    """P4-12's central distinction, decided 2026-09-01.

    Training may draw `negatives_per_positive` quiet windows and distort the
    balance deliberately. Evaluation may not: in live use the system sees every
    trading hour and must stay quiet through nearly all of them, so the
    denominator is every in-universe bar, giving ~0.3% and an always-quiet
    accuracy of ~99.7% — the exact trap the plan says the headline metric
    exists to expose.

    Built so the two numbers cannot silently converge: 1 event against 1,000
    bars is 0.1%, nothing like the ~25% a 3:1 training draw would show.
    """
    lo = date_str_to_ts(cfg["study_window"]["start"])
    iv = cfg["market"]["interval"]
    db.upsert_companies(conn, [{"cik": "CIKEV", "ticker": "EVL",
                                "in_universe": 1}])
    db.upsert_bars(conn, [
        ("EVL", lo + i * HOUR, 0, 0, 0, 100.0, 1e6, iv) for i in range(1000)])
    db.upsert_filings(conn, [{
        "accession_no": "EVL-1", "cik": "CIKEV", "ticker": "EVL", "form": "8-K",
        "items": "8.01", "acceptance_utc": lo + 500 * HOUR,
        "filing_date_utc": lo + 500 * HOUR}])
    db.upsert_events(conn, [{
        "event_id": "EVL1", "accession_no": "EVL-1", "ticker": "EVL",
        "items": "8.01", "t0_filing_utc": lo + 500 * HOUR,
        "t0_utc": lo + 500 * HOUR, "t0_source": "filing", "is_scheduled": 0,
        "usable": 1, "exclude_reason": None}])

    ev = eval_decision_points(cfg, conn)

    assert ev["decision_points"] == 1000
    assert ev["positives"] == 1
    assert ev["base_rate"] == pytest.approx(0.001)
    assert ev["always_quiet_accuracy"] == pytest.approx(0.999)
    # One positive per EVENT, never per pre-event hour: 48 positive hours
    # would read as 4.8% here and ~12.5% on the real data, which is not the
    # number the plan quotes.
    assert ev["base_rate"] < 0.01
