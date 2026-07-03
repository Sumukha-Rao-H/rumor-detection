"""Ticker extraction tests (plan §6.1 rules)."""

import pytest

from src.pipeline.tickers import TickerExtractor
from src.utils.config import load_config


@pytest.fixture(scope="module")
def extractor():
    return TickerExtractor.from_config(load_config())


def test_cashtag_match(extractor):
    assert extractor.extract("Huge news coming for $TSLA tomorrow") == ["TSLA"]


def test_bare_token_requires_universe(extractor):
    assert extractor.extract("I think NVDA will announce a split") == ["NVDA"]
    # Random uppercase token not in universe: no match
    assert extractor.extract("my ZZZZ position") == []


def test_blacklist_never_matches(extractor):
    # DD, YOLO, CEO are real tickers/acronyms but blacklisted common words
    assert extractor.extract("This DD is pure YOLO says the CEO") == []
    # Even as cashtags, blacklisted symbols are dropped
    assert extractor.extract("$DD to the moon") == []


def test_company_name_match(extractor):
    assert extractor.extract("Tesla is acquiring a lithium mine") == ["TSLA"]
    assert extractor.extract("bank of america under investigation") == ["BAC"]


def test_portfolio_spam_discarded(extractor):
    text = "My portfolio: $AAPL $MSFT $NVDA $AMZN all in"
    assert extractor.extract(text) == []


def test_empty_and_none(extractor):
    assert extractor.extract("") == []
    assert extractor.extract(None) == []
