"""News collector tests — response parsing, no network."""

import pytest

from src.collectors.news import (
    domain_of, finnhub_items_to_rows, gdelt_articles_to_rows, publisher_stem,
    tier_of,
)
from src.utils.config import load_config


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


def test_domain_of_strips_www():
    assert domain_of("https://www.reuters.com/business/x") == "reuters.com"
    assert domain_of("https://apnews.com/article/y") == "apnews.com"


def test_gdelt_rows(cfg):
    articles = [
        {"url": "https://www.reuters.com/a", "title": "Tesla acquires X",
         "domain": "reuters.com", "seendate": "20250611T120000Z"},
        {"url": "https://cnbc.com/b", "title": "no seendate"},  # dropped
        {"seendate": "20250611T120000Z"},                       # no url, dropped
    ]
    rows = gdelt_articles_to_rows(cfg, articles, "TSLA")
    assert len(rows) == 1
    row = rows[0]
    assert (row["ticker"], row["source_domain"], row["api"]) == (
        "TSLA", "reuters.com", "gdelt")
    assert row["source_name"] is None, "GDELT identifies publishers by domain"
    assert row["seen_utc"] == 1749643200      # crawl time, 2025-06-11 12:00 UTC
    assert row["published_utc"] is None, (
        "GDELT reports when its crawler SAW the article, not when it was "
        "published — recording a crawl time as publication would corrupt t0")


def test_finnhub_rows(cfg):
    items = [
        {"url": "https://finance.site/a", "headline": "TSLA guidance cut",
         "datetime": 1749643200, "source": "SomeWire"},
        {"headline": "no url", "datetime": 1},  # dropped
    ]
    rows = finnhub_items_to_rows(cfg, items, "TSLA")
    assert len(rows) == 1
    row = rows[0]
    assert row["api"] == "finnhub"
    assert row["published_utc"] == 1749643200, "`datetime` IS publication time"
    assert row["seen_utc"] is None, "this API reports no crawl time"
    assert row["source_name"] == "SomeWire", "the publisher comes from `source`"
    assert row["source_domain"] is None, "a name is not a domain — do not guess"


def test_finnhub_publisher_is_not_taken_from_the_url(cfg):
    """Regression guard for issue 1 in the register.

    Every Finnhub `url` is a redirect wrapper on finnhub.io. Deriving the
    publisher from it labelled all 107 rows of the first real pull `finnhub.io`
    and made the t0 whitelist match nothing.
    """
    items = [{"url": "https://finnhub.io/api/news?id=abc123",
              "headline": "x", "datetime": 1749643200, "source": "Benzinga"}]
    row = finnhub_items_to_rows(cfg, items, "TSLA")[0]
    assert row["source_name"] == "Benzinga"
    assert "finnhub.io" not in str(row["source_domain"])


def test_finnhub_missing_source_is_none_not_empty(cfg):
    items = [{"url": "https://x/a", "headline": "x", "datetime": 1, "source": "  "}]
    assert finnhub_items_to_rows(cfg, items, "TSLA")[0]["source_name"] is None


def test_default_gdelt_query_uses_the_sec_company_name(tmp_path):
    """Company names come from SEC company_tickers_exchange.json, stored in
    `companies` — not from a hand-maintained CSV."""
    from src import db
    from src.collectors.news import default_gdelt_query

    conn = db.get_conn(tmp_path / "t.db")
    db.upsert_companies(conn, [{
        "cik": "0000320193", "ticker": "AAPL", "name": "Apple Inc.",
        "exchange": "Nasdaq", "sic": "3571", "in_universe": 1,
        "adv_usd": 1e10, "last_price": 200.0, "universe_as_of": 0,
    }])
    assert default_gdelt_query(conn, "AAPL") == '"Apple Inc."'
    assert default_gdelt_query(conn, "ZZZZ") == "ZZZZ"  # bare-ticker fallback
    conn.close()


# --------------------------------------------------------------------------
# P1-13b — tiered whitelist
#
# Finnhub's free tier carries no wire services (measured: 1,080 articles over
# 5 tickers, zero from Reuters/PR Newswire/Business Wire). A wire-only
# whitelist would leave the t0 correction applying to almost nothing, so
# credibility is tiered and Phase 4 reports both tiers with the gap.
# --------------------------------------------------------------------------


def test_stem_matches_a_domain_against_a_name():
    """The two APIs identify publishers differently. One normalised stem lets a
    single whitelist cover both without a hand-maintained mapping table."""
    assert publisher_stem("reuters.com") == publisher_stem("Reuters")
    assert publisher_stem("prnewswire.com") == publisher_stem("PR Newswire")
    assert publisher_stem("seekingalpha.com") == publisher_stem("SeekingAlpha")
    assert publisher_stem(None) == "" and publisher_stem("  ") == ""


def test_tier_is_a_property_of_the_publisher_not_the_api(cfg):
    """A wire is tier 1 whether it arrived as a GDELT domain or a Finnhub name."""
    assert tier_of(cfg, "reuters.com", None) == 1
    assert tier_of(cfg, None, "Reuters") == 1


def test_tier2_is_the_fast_republishers(cfg):
    assert tier_of(cfg, None, "Benzinga") == 2
    assert tier_of(cfg, None, "CNBC") == 2


def test_unknown_publisher_gets_no_tier(cfg):
    """ChartMill is a stock screener, not evidence that news became public."""
    assert tier_of(cfg, None, "ChartMill") is None
    assert tier_of(cfg, None, None) is None


def test_the_redirect_domain_is_not_credible(cfg):
    """The pre-P1-13a rows are all stamped finnhub.io. They must not count."""
    assert tier_of(cfg, "finnhub.io", None) is None


def test_rows_carry_their_tier(cfg):
    gdelt = gdelt_articles_to_rows(cfg, [{
        "url": "https://reuters.com/a", "title": "t",
        "domain": "reuters.com", "seendate": "20250611T120000Z"}], "TSLA")
    assert gdelt[0]["source_tier"] == 1

    finn = finnhub_items_to_rows(cfg, [
        {"url": "https://finnhub.io/api/news?id=1", "headline": "a",
         "datetime": 1, "source": "Benzinga"},
        {"url": "https://finnhub.io/api/news?id=2", "headline": "b",
         "datetime": 2, "source": "ChartMill"},
    ], "TSLA")
    assert [r["source_tier"] for r in finn] == [2, None]


# --------------------------------------------------------------------------
# P1-14 — publication time is not crawl time
# --------------------------------------------------------------------------


def test_gdelt_never_populates_published_utc(cfg):
    """The task's Done-when, stated literally.

    GDELT's `seendate` is when its crawler found the article. Treating that as
    publication would push t0 later by an unknown amount for every GDELT-sourced
    event, and nothing downstream would flag it.
    """
    rows = gdelt_articles_to_rows(cfg, [
        {"url": "https://reuters.com/a", "title": "a", "domain": "reuters.com",
         "seendate": "20250611T120000Z"},
        {"url": "https://apnews.com/b", "title": "b", "domain": "apnews.com",
         "seendate": "20250611T130000Z"},
    ], "TSLA")
    assert len(rows) == 2
    assert all(r["published_utc"] is None for r in rows)
    assert all(r["seen_utc"] is not None for r in rows)


def test_finnhub_never_populates_seen_utc(cfg):
    """The mirror image: Finnhub reports publication, not a crawl time."""
    rows = finnhub_items_to_rows(cfg, [
        {"url": "https://x/1", "headline": "a", "datetime": 100, "source": "CNBC"},
    ], "TSLA")
    assert rows[0]["published_utc"] == 100
    assert rows[0]["seen_utc"] is None


def test_fetched_utc_is_always_recorded(cfg):
    """Provenance: when WE pulled the row, distinct from both other times."""
    from src.utils.timeutils import utc_now_ts

    now = utc_now_ts()
    finn = finnhub_items_to_rows(cfg, [
        {"url": "https://x/1", "headline": "a", "datetime": 100, "source": "CNBC"}], "T")
    gdelt = gdelt_articles_to_rows(cfg, [
        {"url": "https://y/1", "title": "b", "domain": "reuters.com",
         "seendate": "20250611T120000Z"}], "T")
    for row in finn + gdelt:
        assert abs(row["fetched_utc"] - now) < 5
