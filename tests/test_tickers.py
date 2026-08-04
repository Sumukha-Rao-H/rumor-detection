"""Ticker extraction tests (plan §6.1 rules)."""

import pytest

from src.pipeline.tickers import TickerExtractor, load_matchable_names
from src.utils.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def extractor(cfg):
    return TickerExtractor.from_config(cfg)


def test_cashtag_match(extractor):
    assert extractor.extract("Huge news coming for $TSLA tomorrow") == ["TSLA"]


def test_cashtag_requires_a_real_listed_symbol(extractor):
    """Pump posts invent tickers; only US-listed symbols count."""
    assert extractor.extract("$ZQXJ about to run 500%") == []


def test_bare_token_requires_core_universe(extractor):
    assert extractor.extract("I think NVDA will announce a split") == ["NVDA"]
    # Random uppercase token not in the universe: no match
    assert extractor.extract("my ZZZZ position") == []


def test_blacklist_never_matches(extractor):
    # DD, YOLO, CEO are real tickers/acronyms but blacklisted common words
    assert extractor.extract("This DD is pure YOLO says the CEO") == []
    # Even as cashtags, blacklisted symbols are dropped
    assert extractor.extract("$DD to the moon") == []


def test_company_name_match(extractor):
    assert extractor.extract("Tesla is acquiring a lithium mine") == ["TSLA"]
    assert extractor.extract("bank of america under investigation") == ["BAC"]


def test_generic_company_names_do_not_match(extractor):
    """The name_match flag must keep ordinary words from becoming tickers."""
    assert extractor.extract("my price target for this is 400") == []
    assert extractor.extract("saw this on reddit earlier today") == []
    assert extractor.extract("it is just a shell company") == []


def test_portfolio_spam_discarded(extractor):
    text = "My portfolio: $AAPL $MSFT $NVDA $AMZN all in"
    assert extractor.extract(text) == []


def test_empty_and_none(extractor):
    assert extractor.extract("") == []
    assert extractor.extract(None) == []


def test_matchable_names_respects_the_flag(tmp_path):
    path = tmp_path / "u.csv"
    path.write_text("ticker,name,name_match\nTGT,Target,0\nNVDA,Nvidia,1\n")
    assert load_matchable_names(path) == {"nvidia": "NVDA"}


def test_matchable_names_without_the_column_keeps_multiword_only(tmp_path):
    """Legacy CSVs have no name_match; only distinctive names are trusted."""
    path = tmp_path / "u.csv"
    path.write_text("ticker,name\nTGT,Target\nBAC,Bank of America\n")
    assert load_matchable_names(path) == {"bank of america": "BAC"}


def test_listed_only_symbols_are_cashtag_only():
    """A symbol outside the core universe matches as $CASHTAG but not bare."""
    ex = TickerExtractor(
        universe={"NVDA": "Nvidia"},
        blacklist=[],
        listed={"NVDA", "ZEO"},
        matchable_names={"nvidia": "NVDA"},
    )
    assert ex.extract("$ZEO is moving") == ["ZEO"]
    assert ex.extract("ZEO is moving") == []
