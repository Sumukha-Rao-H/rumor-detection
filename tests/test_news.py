"""News collector tests — response parsing, no network."""

from src.collectors.news import (
    domain_of, finnhub_items_to_rows, gdelt_articles_to_rows,
)


def test_domain_of_strips_www():
    assert domain_of("https://www.reuters.com/business/x") == "reuters.com"
    assert domain_of("https://apnews.com/article/y") == "apnews.com"


def test_gdelt_rows():
    articles = [
        {"url": "https://www.reuters.com/a", "title": "Tesla acquires X",
         "domain": "reuters.com", "seendate": "20250611T120000Z"},
        {"url": "https://cnbc.com/b", "title": "no seendate"},  # dropped
        {"seendate": "20250611T120000Z"},                       # no url, dropped
    ]
    rows = gdelt_articles_to_rows(articles, "TSLA")
    assert len(rows) == 1
    url, ticker, title, domain, seen, api = rows[0]
    assert (ticker, domain, api) == ("TSLA", "reuters.com", "gdelt")
    assert seen == 1749643200  # 2025-06-11 12:00:00 UTC


def test_finnhub_rows():
    items = [
        {"url": "https://finance.site/a", "headline": "TSLA guidance cut",
         "datetime": 1749643200, "source": "SomeWire"},
        {"headline": "no url", "datetime": 1},  # dropped
    ]
    rows = finnhub_items_to_rows(items, "TSLA")
    assert len(rows) == 1
    assert rows[0][4] == 1749643200 and rows[0][5] == "finnhub"
