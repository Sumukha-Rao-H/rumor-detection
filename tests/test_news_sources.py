"""Publisher attribution for Finnhub rows (plan §6.4 whitelist matching)."""

from src.collectors.news import finnhub_items_to_rows, finnhub_source_domain

ALIASES = {"Reuters": "reuters.com", "Yahoo": "finance.yahoo.com"}


def test_a_known_publisher_maps_to_its_domain():
    assert finnhub_source_domain("Reuters", ALIASES) == "reuters.com"


def test_publisher_matching_ignores_case():
    assert finnhub_source_domain("reuters", ALIASES) == "reuters.com"


def test_an_unknown_publisher_is_kept_but_unmatchable():
    """It must stay visible for review, not inherit a credible domain."""
    assert finnhub_source_domain("Seeking Alpha", ALIASES) == "seekingalpha"


def test_a_missing_publisher_is_empty():
    assert finnhub_source_domain(None, ALIASES) == ""


def test_rows_are_attributed_to_the_publisher_not_the_redirect():
    """Finnhub's url is finnhub.io for every article; the source field is not."""
    items = [{"url": "https://finnhub.io/api/news?id=abc", "datetime": 100,
              "headline": "Acme acquired", "source": "Reuters"}]
    (row,) = finnhub_items_to_rows(items, "AAA", ALIASES)
    assert row[3] == "reuters.com"


def test_rows_fall_back_to_the_url_when_there_is_no_source():
    items = [{"url": "https://www.cnbc.com/x", "datetime": 100,
              "headline": "h", "source": ""}]
    (row,) = finnhub_items_to_rows(items, "AAA", ALIASES)
    assert row[3] == "cnbc.com"
