"""Universe builder + relink tests (plan §6.1 empirical prune)."""

import pytest

from src import db
from src.pipeline.build_universe import (
    CorpusStats,
    build_core_universe,
    clean_company_name,
)
from src.pipeline.relink import delete_orphans, relink
from src.pipeline.tickers import TickerExtractor


@pytest.mark.parametrize("raw,expected", [
    ("Apple Inc. - Common Stock", "Apple"),
    ("Agilent Technologies, Inc. Common Stock", "Agilent Technologies"),
    ("Alphabet Inc. Class A Common Stock", "Alphabet"),
    ("Novo Nordisk A/S - American Depositary Shares", "Novo Nordisk A/S"),
    ("Bank of America Corporation", "Bank of America"),
    ("- Warrants", ""),
])
def test_clean_company_name(raw, expected):
    assert clean_company_name(raw) == expected


def _cfg(**over):
    tickers = {
        "blacklist": ["USD"],
        "core_size": 10,
        "core_min_cashtags": 2,
        "name_match_min_support": 3,
        "name_match_min_precision": 0.15,
    }
    tickers.update(over)
    return {"tickers": tickers}


def _stats(texts, candidates):
    stats = CorpusStats(candidates, 0, 10**11)
    for text in texts:
        stats.offer_text(text)
    return stats


def test_core_universe_ranks_by_cashtags_not_bare_tokens():
    """Bare uppercase counts are dominated by acronyms; $CASHTAG is intentional."""
    listed = [
        {"ticker": "NVDA", "name": "Nvidia", "etf": 0},
        {"ticker": "ET", "name": "Energy Transfer", "etf": 0},
    ]
    texts = ["$NVDA up", "$NVDA again", "earnings at 4pm ET", "call at 9am ET",
             "ET is the timezone", "$ET once"]
    core = build_core_universe(listed, _stats(texts, {}), _cfg(), {})
    assert [r["ticker"] for r in core] == ["NVDA"]  # ET: 1 cashtag < min 2


def test_blacklisted_symbols_never_enter_the_core():
    listed = [{"ticker": "USD", "name": "ProShares Ultra Semiconductors", "etf": 1}]
    core = build_core_universe(listed, _stats(["$USD $USD $USD"], {}), _cfg(), {})
    assert core == []


def test_frequent_single_word_name_needs_cooccurrence():
    """'price target' must not become TGT; 'Nvidia' must stay NVDA."""
    listed = [
        {"ticker": "TGT", "name": "Target", "etf": 0},
        {"ticker": "NVDA", "name": "Nvidia", "etf": 0},
    ]
    candidates = {"target": "TGT", "nvidia": "NVDA"}
    texts = (
        ["my price target is 400", "target hit", "raising my target", "target again"]
        + ["$TGT calls", "$TGT puts", "$NVDA earnings", "$NVDA calls"]
        + ["nvidia and NVDA", "nvidia NVDA again", "nvidia rally NVDA", "nvidia"]
    )
    core = {r["ticker"]: r for r in build_core_universe(
        listed, _stats(texts, candidates), _cfg(), {})}
    assert core["TGT"]["name_match"] == 0
    assert core["NVDA"]["name_match"] == 1


def test_multiword_name_is_trusted_without_cooccurrence():
    """'Bank of America' cannot collide with English, so precision is moot."""
    listed = [{"ticker": "BAC", "name": "Bank of America", "etf": 0}]
    candidates = {"bank of america": "BAC"}
    texts = ["$BAC calls", "$BAC puts"] + ["bank of america is fine"] * 5
    core = build_core_universe(listed, _stats(texts, candidates), _cfg(), {})
    assert core[0]["name_match"] == 1
    assert core[0]["name_precision"] == 0.0


def test_relink_rebuilds_links_and_reports_orphans(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    posts = [
        {"id": "keep", "title": "$NVDA to the moon", "selftext": "",
         "created_utc": 1, "source": "arctic"},
        {"id": "orphan", "title": "my price target is 400", "selftext": "",
         "created_utc": 2, "source": "arctic"},
    ]
    db.upsert_posts(conn, posts)
    # Stale links from an older, looser universe.
    db.link_post_tickers(conn, [("keep", "NVDA"), ("orphan", "TGT")])

    extractor = TickerExtractor(universe={"NVDA": "Nvidia"}, blacklist=[],
                                listed={"NVDA", "TGT"},
                                matchable_names={"nvidia": "NVDA"})
    stats = relink(conn, extractor)
    assert stats == {"posts": 2, "links_before": 2, "links_after": 1, "orphans": 1}
    assert [r[0] for r in conn.execute("SELECT ticker FROM post_tickers")] == ["NVDA"]

    assert delete_orphans(conn) == 1
    assert [r[0] for r in conn.execute("SELECT id FROM posts")] == ["keep"]


def test_relink_dry_run_changes_nothing(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    db.upsert_posts(conn, [{"id": "orphan", "title": "nothing here",
                            "created_utc": 1, "source": "arctic"}])
    db.link_post_tickers(conn, [("orphan", "TGT")])
    extractor = TickerExtractor(universe={}, blacklist=[], listed=set())

    stats = relink(conn, extractor, dry_run=True)
    assert stats["orphans"] == 1
    assert conn.execute("SELECT COUNT(*) FROM post_tickers").fetchone()[0] == 1
