"""P4-01 — finding the press release that preceded the filing.

The Done-when is `test_finds_the_release_before_an_earnings_filing`, built to
the real shape this project rests on: an earnings release on the wire at
~16:05 ET, the 8-K accepted at ~20:30 UTC. If the matcher cannot recover that
gap, the whole t0 correction is decorative.

The rest pin the window's two edges, the tier ceiling, and the publication-time
rule — each of which, if wrong, silently moves every lead-time number.
"""

import pytest

from src import db
from src.pipeline.t0 import (
    Match, gap_percentiles, lookback_window, match_all, match_filing,
)
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, iso_utc_to_ts


HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "t0.db")


def add_news_at(conn, ticker, published_utc, tier=2, name="Benzinga",
                title="t"):
    """Insert an article at an exact epoch, for tests keyed off the lookback."""
    db.upsert_news(conn, [{
        "url": f"http://x/{ticker}/{published_utc}", "ticker": ticker,
        "title": title, "source_name": name, "source_domain": "x.com",
        "source_tier": tier, "published_utc": published_utc, "api": "finnhub",
    }])
    return published_utc


def add_news(conn, ticker, published_iso, tier=2, name="Benzinga",
             title="t", url=None, seen_utc=None):
    ts = iso_utc_to_ts(published_iso)
    db.upsert_news(conn, [{
        "url": url or f"http://x/{ticker}/{ts}", "ticker": ticker, "title": title,
        "source_name": name, "source_domain": "x.com", "source_tier": tier,
        "published_utc": ts, "seen_utc": seen_utc, "api": "finnhub",
    }])
    return ts


# --------------------------------------------------------------------------
# the Done-when
# --------------------------------------------------------------------------

def test_finds_the_release_before_an_earnings_filing(cfg, conn):
    """The exact shape the t0 correction exists for.

    Earnings cross the wire at 16:05 ET (20:05 UTC in summer) and the 8-K is
    accepted at 20:30 UTC. Using acceptance alone would count those 25 minutes
    as "advance warning" when the market already knew.
    """
    acceptance = iso_utc_to_ts("2026-02-25T20:30:00Z")
    release = add_news(conn, "AAPL", "2026-02-25T20:05:00Z")

    m = match_filing(cfg, conn, "AAPL", acceptance)
    assert m.source == "news"
    assert m.news_utc == release
    assert m.t0_utc == release
    assert m.t0_utc < acceptance
    assert m.gap_hours == pytest.approx(25 / 60, abs=1e-6)


# --------------------------------------------------------------------------
# the window's edges
# --------------------------------------------------------------------------

def test_lookback_window_ends_at_acceptance(cfg):
    """It must never reach past the filing — that would be look-ahead."""
    acc = iso_utc_to_ts("2026-02-25T20:30:00Z")
    lo, hi = lookback_window(cfg, acc)
    assert hi == acc
    assert lo == acc - cfg["news"]["t0_lookback_hours"] * HOUR


def test_takes_the_earliest_article_not_the_latest(cfg, conn):
    """t0 is a minimum: the first public moment, not the most recent one.

    Both articles are placed relative to the configured lookback, so the test
    keeps meaning what it says if the window is retuned again.
    """
    acc = iso_utc_to_ts("2026-02-25T20:30:00Z")
    look = cfg["news"]["t0_lookback_hours"] * HOUR
    first = add_news_at(conn, "AAPL", acc - look + 60)      # just inside
    add_news_at(conn, "AAPL", acc - 60)                     # just before filing
    assert match_filing(cfg, conn, "AAPL", acc).t0_utc == first


def test_ignores_articles_published_after_acceptance(cfg, conn):
    """A later article cannot lower a minimum, and reaching forward is leakage."""
    acc = iso_utc_to_ts("2026-02-25T20:30:00Z")
    add_news(conn, "AAPL", "2026-02-25T21:00:00Z")
    m = match_filing(cfg, conn, "AAPL", acc)
    assert m.source == "filing" and m.t0_utc == acc


def test_ignores_articles_before_the_lookback_opens(cfg, conn):
    """The window has a floor.

    This is the whole point of issue 27: without a floor, unrelated earlier
    coverage becomes t0 and the label turns arbitrary.
    """
    acc = iso_utc_to_ts("2026-02-25T20:30:00Z")
    look = cfg["news"]["t0_lookback_hours"] * HOUR
    add_news_at(conn, "AAPL", acc - look - HOUR, title="an hour too early")
    inside = add_news_at(conn, "AAPL", acc - look + 60)
    assert match_filing(cfg, conn, "AAPL", acc).t0_utc == inside


def test_the_lookback_floor_is_inclusive(cfg, conn):
    """An article exactly `t0_lookback_hours` before acceptance still counts.

    Pinned because it is the one boundary a reader is likely to assume the
    other way, and it decides t0 for any event sitting on it.
    """
    acc = iso_utc_to_ts("2026-02-25T20:30:00Z")
    look = cfg["news"]["t0_lookback_hours"] * HOUR
    edge = add_news_at(conn, "AAPL", acc - look)            # exactly on the floor
    assert match_filing(cfg, conn, "AAPL", acc).t0_utc == edge


def test_an_article_exactly_at_acceptance_is_included(cfg, conn):
    acc = iso_utc_to_ts("2026-02-25T20:30:00Z")
    add_news(conn, "AAPL", "2026-02-25T20:30:00Z")
    assert match_filing(cfg, conn, "AAPL", acc).t0_utc == acc


def test_returns_filing_time_when_no_article_exists(cfg, conn):
    acc = iso_utc_to_ts("2026-02-25T20:30:00Z")
    m = match_filing(cfg, conn, "AAPL", acc)
    assert m.source == "filing" and m.news_utc is None and m.t0_utc == acc


def test_only_matches_the_same_ticker(cfg, conn):
    acc = iso_utc_to_ts("2026-02-25T20:30:00Z")
    add_news(conn, "MSFT", "2026-02-25T20:05:00Z")
    assert match_filing(cfg, conn, "AAPL", acc).source == "filing"


# --------------------------------------------------------------------------
# tiers — the sensitivity analysis
# --------------------------------------------------------------------------

def test_tier1_only_ignores_a_tier2_article(cfg, conn):
    """Running at tier 1 and tier 2 and comparing IS the sensitivity check.

    On the real data no tier-1 article exists at all, so a tier-1 t0 falls back
    to filing time everywhere — the project's headline limitation, measured.
    """
    acc = iso_utc_to_ts("2026-02-25T20:30:00Z")
    add_news(conn, "AAPL", "2026-02-25T20:05:00Z", tier=2)
    assert match_filing(cfg, conn, "AAPL", acc, max_tier=2).source == "news"
    assert match_filing(cfg, conn, "AAPL", acc, max_tier=1).source == "filing"


def test_a_tier1_article_is_found_at_both_ceilings(cfg, conn):
    acc = iso_utc_to_ts("2026-02-25T20:30:00Z")
    wire = add_news(conn, "AAPL", "2026-02-25T20:05:00Z", tier=1, name="Reuters")
    assert match_filing(cfg, conn, "AAPL", acc, max_tier=1).t0_utc == wire
    assert match_filing(cfg, conn, "AAPL", acc, max_tier=2).t0_utc == wire


def test_untiered_publishers_are_excluded(cfg, conn):
    """A publisher not on either whitelist is not evidence the news was public."""
    acc = iso_utc_to_ts("2026-02-25T20:30:00Z")
    add_news(conn, "AAPL", "2026-02-25T20:05:00Z", tier=None, name="SomeBlog")
    assert match_filing(cfg, conn, "AAPL", acc, max_tier=2).source == "filing"


def test_uses_publication_time_not_crawl_time(cfg, conn):
    """P1-14's distinction. Crawl time lags publication and would move t0."""
    acc = iso_utc_to_ts("2026-02-25T20:30:00Z")
    db.upsert_news(conn, [{
        "url": "http://x/gdelt", "ticker": "AAPL", "title": "t",
        "source_name": "Benzinga", "source_tier": 2,
        "published_utc": None,                       # GDELT row: crawl time only
        "seen_utc": iso_utc_to_ts("2026-02-25T20:05:00Z"), "api": "gdelt",
    }])
    assert match_filing(cfg, conn, "AAPL", acc).source == "filing"


# --------------------------------------------------------------------------
# the run over everything, and the diagnostic
# --------------------------------------------------------------------------

def seed_universe_filing(conn, ticker, acceptance_iso):
    db.upsert_companies(conn, [{"cik": f"CIK{ticker}", "ticker": ticker,
                                "in_universe": 1}])
    ts = iso_utc_to_ts(acceptance_iso)
    db.upsert_filings(conn, [{
        "accession_no": f"{ticker}-{ts}", "cik": f"CIK{ticker}", "ticker": ticker,
        "form": "8-K", "items": "2.02", "acceptance_utc": ts,
        "filing_date_utc": ts,
    }])
    return ts


def test_match_all_covers_every_in_window_filing(cfg, conn):
    a1 = seed_universe_filing(conn, "AAA", "2026-02-25T20:30:00Z")
    seed_universe_filing(conn, "BBB", "2026-03-10T20:30:00Z")
    add_news(conn, "AAA", "2026-02-25T20:05:00Z")
    ms = match_all(cfg, conn)
    assert len(ms) == 2
    assert {m.ticker: m.source for m in ms} == {"AAA": "news", "BBB": "filing"}


def test_match_all_skips_tickers_outside_the_universe(cfg, conn):
    seed_universe_filing(conn, "AAA", "2026-02-25T20:30:00Z")
    db.upsert_companies(conn, [{"cik": "CIKOUT", "ticker": "OUT",
                                "in_universe": 0}])
    db.upsert_filings(conn, [{
        "accession_no": "OUT-1", "cik": "CIKOUT", "ticker": "OUT", "form": "8-K",
        "items": "2.02", "acceptance_utc": iso_utc_to_ts("2026-02-25T20:30:00Z"),
    }])
    assert {m.ticker for m in match_all(cfg, conn)} == {"AAA"}


def test_match_all_raises_when_there_is_nothing_to_match(cfg, conn):
    with pytest.raises(SystemExit, match="no in-window filings"):
        match_all(cfg, conn)


def test_gap_percentiles_measure_only_news_matches(cfg):
    """The diagnostic behind issue 27 — filing-time fallbacks are not gaps."""
    acc = 1_800_000_000
    ms = [Match("A", acc, acc - 2 * HOUR, acc - 2 * HOUR, "news"),
          Match("B", acc, acc - 18 * HOUR, acc - 18 * HOUR, "news"),
          Match("C", acc, None, acc, "filing")]
    pct = gap_percentiles(ms)
    assert pct["within_2h"] == pytest.approx(0.5)
    assert pct["beyond_12h"] == pytest.approx(0.5)
    assert pct["mean"] == pytest.approx(10.0)


def test_gap_percentiles_empty_when_nothing_matched(cfg):
    """Tier 1 on the real data produces exactly this."""
    assert gap_percentiles([Match("A", 1, None, 1, "filing")]) == {}
