"""P4-01/P4-02 — the reports, which are the only thing anyone actually reads.

`stored_gap_report` prints the task's Done-when number, `print_report` prints
the match rate the whole t0 claim rests on, and `print_sample` is how issue 27
was spotted in the first place. All three shipped untested: the gap report was
verified by re-implementing its SQL in a test, which checks SQLite rather than
the reporter.

What is pinned here is what a reader would be misled by if it broke: numbers
read back from the table instead of recomputed, buckets that live inside the
lookback window, the two very different reasons an event keeps filing time,
and a sample that shows the article that actually matched.
"""

import pytest

from src import db
from src.pipeline import t0 as T
from src.pipeline.t0 import (
    print_report, print_sample, stored_gap_report, write_events,
)
from src.utils.config import load_config
from src.utils.timeutils import iso_utc_to_ts


HOUR = 3600
ACCEPTANCE = "2026-02-25T20:30:00Z"


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "report.db")


def seed(conn, ticker="AAPL", accession=None, acceptance_iso=ACCEPTANCE,
         in_universe=1):
    db.upsert_companies(conn, [{"cik": f"CIK{ticker}", "ticker": ticker,
                                "in_universe": in_universe}])
    ts = iso_utc_to_ts(acceptance_iso)
    db.upsert_filings(conn, [{
        "accession_no": accession or f"{ticker}-{ts}", "cik": f"CIK{ticker}",
        "ticker": ticker, "form": "8-K", "items": "2.02",
        "acceptance_utc": ts, "filing_date_utc": ts,
    }])
    return ts


def add_article(conn, ticker, at_utc, tier=2, title="t", name="Benzinga",
                url=None):
    db.upsert_news(conn, [{
        "url": url or f"http://x/{ticker}/{at_utc}/{tier}", "ticker": ticker,
        "title": title, "source_name": name, "source_domain": "x.com",
        "source_tier": tier, "published_utc": at_utc, "api": "finnhub",
    }])
    return at_utc


def with_lookback(cfg, hours):
    return {**cfg, "news": {**cfg["news"], "t0_lookback_hours": hours}}


# --------------------------------------------------------------------------
# stored_gap_report — the Done-when number
# --------------------------------------------------------------------------

def test_gap_report_prints_the_match_rate_and_the_median(cfg, conn, capsys):
    """One matched event at 45 minutes, one falling back to filing time."""
    acc = seed(conn, "AAA")
    add_article(conn, "AAA", acc - 45 * 60)
    seed(conn, "BBB")
    write_events(cfg, conn)

    stored_gap_report(cfg, conn)
    out = capsys.readouterr().out
    assert "events              : 2" in out
    assert "t0 from news        : 1  (50.0%)" in out
    assert "t0 from filing time : 1" in out
    assert "median :    45.0 min" in out


def test_gap_report_reads_the_table_rather_than_recomputing(cfg, conn, capsys):
    """The stored number must be the reported number. Deleting the article
    afterwards cannot move it — if it does, the report is recomputing and the
    figure in the write-up describes a table nobody has."""
    acc = seed(conn, "AAA")
    add_article(conn, "AAA", acc - 45 * 60)
    write_events(cfg, conn)
    conn.execute("DELETE FROM news")
    conn.commit()

    stored_gap_report(cfg, conn)
    out = capsys.readouterr().out
    assert "t0 from news        : 1  (100.0%)" in out
    assert "median :    45.0 min" in out


def test_gap_report_points_at_the_command_that_stores_events(cfg, conn, capsys):
    """The old text said "run without --report first", which is not a command
    that stores anything."""
    stored_gap_report(cfg, conn)
    out = capsys.readouterr().out
    assert "--build" in out
    assert "without --report" not in out


def test_gap_report_prints_the_tier_and_lookback_that_built_the_rows(
        cfg, conn, capsys):
    """Provenance. "0 matched" at tier 1 and "0 matched" from a broken news
    table are the same row in the table and must not be the same report."""
    acc = seed(conn, "AAA")
    add_article(conn, "AAA", acc - 45 * 60)
    write_events(with_lookback(cfg, 3), conn, max_tier=1)

    stored_gap_report(cfg, conn)
    out = capsys.readouterr().out
    assert "at tier 1, 3h lookback" in out
    assert "No event took its t0 from news at this tier." in out


def test_gap_report_admits_when_the_rows_predate_provenance(cfg, conn, capsys):
    """Rows already in the live DB carry no stamp; saying so beats implying
    the current config produced them."""
    conn.execute(
        "INSERT INTO events (event_id, ticker, t0_filing_utc, t0_news_utc, "
        "t0_utc, t0_source) VALUES ('e1', 'AAA', 100, 100, 100, 'news')")
    conn.commit()

    stored_gap_report(cfg, conn)
    assert "unknown" in capsys.readouterr().out


# --------------------------------------------------------------------------
# print_report — the match rate, and why the rest fell back
# --------------------------------------------------------------------------

def test_report_splits_the_two_reasons_for_keeping_filing_time(cfg, conn,
                                                               capsys):
    """A ticker with no news at all is a coverage hole — and the shape a lost
    article takes here, since `news` is keyed on url alone. Reporting it as an
    ordinary "no article in the window" hides the defect inside a rate."""
    acc = seed(conn, "AAA")
    add_article(conn, "AAA", acc - 30 * 60)                  # matched
    covered = seed(conn, "BBB")
    add_article(conn, "BBB", covered - 40 * HOUR)            # covered, too old
    seed(conn, "CCC")                                        # nothing at all

    print_report(cfg, conn)
    out = capsys.readouterr().out
    assert "matched to news  : 1  (33.3%)" in out
    assert "fall back to filing time: 2" in out
    assert "no article in the 3h lookback : 1" in out
    assert "no news stored for the ticker     : 1" in out


def test_report_buckets_live_inside_the_lookback(cfg, conn, capsys):
    """Bug 4: `beyond 6h`/`beyond 12h` were 0% by construction under a 3h
    window, and the warning still talked about a 24h one."""
    acc = seed(conn, "AAA")
    add_article(conn, "AAA", acc - 30 * 60)

    print_report(cfg, conn)
    out = capsys.readouterr().out
    assert "within  0.75h of acceptance: 100%" in out
    assert "within  1.50h of acceptance" in out
    assert "more than  6h" not in out
    assert "more than 12h" not in out
    assert "24h window" not in out


def test_report_warns_when_the_median_sits_at_the_windows_outer_edge(
        cfg, conn, capsys):
    """Issue 27's signature: widen the window and the earliest article stops
    being the release. The threshold moves with the window instead of being a
    literal 6, so the warning can still fire."""
    wide = with_lookback(cfg, 24)
    for t in ("AAA", "BBB", "CCC"):
        acc = seed(conn, t)
        add_article(conn, t, acc - 20 * HOUR)

    print_report(wide, conn)
    out = capsys.readouterr().out
    assert "⚠" in out and "issue 27" in out


def test_report_does_not_warn_when_matches_sit_near_acceptance(cfg, conn,
                                                               capsys):
    for t in ("AAA", "BBB", "CCC"):
        acc = seed(conn, t)
        add_article(conn, t, acc - 20 * 60)

    print_report(cfg, conn)
    assert "⚠" not in capsys.readouterr().out


def test_report_states_the_limitation_when_nothing_matched(cfg, conn, capsys):
    """Tier 1 on the real data prints exactly this, and it is a finding."""
    acc = seed(conn, "AAA")
    add_article(conn, "AAA", acc - 30 * 60, tier=2)

    print_report(cfg, conn, max_tier=1)
    out = capsys.readouterr().out
    assert "No article matched at this tier" in out
    assert "matched to news  : 0  (0.0%)" in out


# --------------------------------------------------------------------------
# print_sample — the headlines
# --------------------------------------------------------------------------

def test_sample_shows_the_article_that_actually_matched(cfg, conn, capsys):
    """Bug 6: the lookup dropped the tier ceiling, so a same-second untiered
    blog post could be printed as "the release" — in the one tool whose whole
    job is judging whether the match is the release."""
    acc = seed(conn, "AAA")
    at = acc - 30 * 60
    add_article(conn, "AAA", at, tier=None, title="BLOG POST", name="SomeBlog",
                url="http://a/blog")
    add_article(conn, "AAA", at, tier=2, title="REAL RELEASE", name="Benzinga",
                url="http://z/wire")

    print_sample(cfg, conn, 5)
    out = capsys.readouterr().out
    assert "REAL RELEASE" in out and "BLOG POST" not in out
    assert "Benzinga" in out


def test_sample_is_deterministic(cfg, conn, capsys):
    """Two articles for the same ticker at the same second: without an
    ORDER BY, SQLite may return either, and the same command prints a
    different headline on a second run."""
    acc = seed(conn, "AAA")
    at = acc - 30 * 60
    add_article(conn, "AAA", at, title="first", url="http://a/1")
    add_article(conn, "AAA", at, title="second", url="http://a/2")

    print_sample(cfg, conn, 5)
    first = capsys.readouterr().out
    print_sample(cfg, conn, 5)
    assert capsys.readouterr().out == first


def test_sample_of_zero_prints_no_filings(cfg, conn, capsys):
    acc = seed(conn, "AAA")
    add_article(conn, "AAA", acc - 30 * 60)

    print_sample(cfg, conn, 0)
    out = capsys.readouterr().out
    assert "=== 0 matched filings ===" in out
    assert "AAA" not in out


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------

@pytest.fixture
def cli(monkeypatch, cfg, conn):
    """Run `main()` with a given argv against the temp DB."""
    calls = {}
    monkeypatch.setattr(T, "load_config", lambda: cfg)
    monkeypatch.setattr(T.db, "get_conn", lambda path: conn)
    for name in ("write_events", "stored_gap_report", "print_report",
                 "print_sample"):
        monkeypatch.setattr(
            T, name,
            lambda *a, name=name, **kw: calls.setdefault(name, kw))

    def run(*argv):
        monkeypatch.setattr("sys.argv", ["t0", *argv])
        T.main()
        return calls
    return run


def test_build_refuses_a_tier(monkeypatch, cfg):
    """Bug 2. `--tier 1` is advertised as the sensitivity check, but there is
    one `t0_utc` column: adding `--build` does not store a second variant, it
    overwrites the study with the uncorrected t0 — 8,659 corrections gone."""
    monkeypatch.setattr("sys.argv", ["t0", "--build", "--tier", "1"])
    monkeypatch.setattr(T.db, "get_conn", lambda path: pytest.fail(
        "the DB must not even be opened for a refused build"))
    with pytest.raises(SystemExit, match="cannot be combined with --build"):
        T.main()


def test_report_still_accepts_a_tier(cli):
    assert cli("--report", "--tier", "1")["print_report"] == {"max_tier": 1}


def test_tier_any_reaches_the_untiered_ceiling(cli):
    """`max_tier=None` is supported everywhere below the CLI; before this the
    CLI's choices made it unreachable."""
    assert cli("--report", "--tier", "any")["print_report"] == {"max_tier": None}


def test_sample_of_zero_is_not_treated_as_no_sample(cli):
    """`--sample 0` is falsy: it used to fall through and print the full
    report instead."""
    calls = cli("--sample", "0")
    assert "print_sample" in calls and "print_report" not in calls


def test_build_reports_the_stored_gap_afterwards(cli):
    calls = cli("--build")
    assert calls["write_events"] == {"max_tier": 2}
    assert "stored_gap_report" in calls
