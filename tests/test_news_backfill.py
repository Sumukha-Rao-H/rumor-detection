"""P4-00 — the in-window news backfill, and the resume a 15-hour run needs.

The Done-when is `test_targets_cover_every_filing_lookback`: every filing must
have its `t0_lookback_hours` fully inside a fetched window, or the miss shows
up later as "no news found" rather than as an error. The rest protects the
things that make a long run survivable — state per (ticker, window), a quiet
week recorded as a permanent answer, and a guard that checks attempts before
it cries wolf.
"""

import pytest

from src import db
from src.collectors import news
from src.collectors.news import (
    FETCH_SOURCE, backfill_targets, collect_targets, full_coverage_targets,
    window_grid,
)
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts


HOUR = 3600


@pytest.fixture(scope="module")
def cfg():
    cfg = load_config()
    cfg["news"] = {**cfg["news"], "finnhub_min_interval_s": 0.0,
                   "gdelt_min_interval_s": 0.0}
    return cfg


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "bf.db")


def add_filing(conn, ticker, acceptance_utc, accession=None, form="8-K"):
    db.upsert_filings(conn, [{
        "accession_no": accession or f"{ticker}-{acceptance_utc}",
        "cik": "0000000001", "ticker": ticker, "form": form, "items": "1.01",
        "acceptance_utc": acceptance_utc, "filing_date_utc": acceptance_utc,
    }])


def covers(targets, ticker, ts):
    return any(t == ticker and s <= ts < e for t, _, s, e in targets)


# --------------------------------------------------------------------------
# which weeks get fetched
# --------------------------------------------------------------------------

def test_targets_cover_every_filing_lookback(cfg, conn):
    """THE Done-when: acceptance AND the full lookback before it are covered."""
    lookback = cfg["news"]["t0_lookback_hours"] * HOUR
    grid = window_grid(cfg)
    stamps = [grid[0][0] + 5 * HOUR, grid[3][0] + 100 * HOUR, grid[-1][1] - HOUR]
    for i, ts in enumerate(stamps):
        add_filing(conn, "AAPL", ts, accession=f"a-{i}")

    targets = backfill_targets(cfg, conn)
    for ts in stamps:
        assert covers(targets, "AAPL", ts), ts
        assert covers(targets, "AAPL", max(ts - lookback, grid[0][0])), ts


def test_filing_at_a_window_edge_pulls_the_previous_window(cfg, conn):
    """A filing 1 h into a week needs the week before it for the 24 h lookback."""
    grid = window_grid(cfg)
    add_filing(conn, "AAPL", grid[5][0] + HOUR)
    idx = {i for _, i, _, _ in backfill_targets(cfg, conn)}
    assert idx == {4, 5}


def test_targets_skip_weeks_with_no_filing(cfg, conn):
    """The whole point of the 89 h -> 15 h saving."""
    grid = window_grid(cfg)
    add_filing(conn, "AAPL", grid[10][0] + 80 * HOUR)
    targets = backfill_targets(cfg, conn)
    assert len(targets) == 1 and len(grid) > 40


def test_targets_are_deduped(cfg, conn):
    """Ten filings in one week is one call, not ten."""
    grid = window_grid(cfg)
    for i in range(10):
        add_filing(conn, "AAPL", grid[7][0] + (50 + i) * HOUR, accession=f"d-{i}")
    assert len(backfill_targets(cfg, conn)) == 1


def test_targets_ignore_filings_outside_the_study_window(cfg, conn):
    add_filing(conn, "AAPL", date_str_to_ts(cfg["study_window"]["start"]) - 40 * 86400)
    add_filing(conn, "MSFT", date_str_to_ts(cfg["study_window"]["end"]) + 40 * 86400)
    assert backfill_targets(cfg, conn) == []


def test_targets_include_amendments_and_ignore_other_forms(cfg, conn):
    grid = window_grid(cfg)
    add_filing(conn, "AAPL", grid[9][0] + 80 * HOUR, form="8-K/A")
    add_filing(conn, "MSFT", grid[9][0] + 80 * HOUR, form="10-Q")
    assert {t for t, _, _, _ in backfill_targets(cfg, conn)} == {"AAPL"}


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------

class FakeCollect:
    """Stands in for `collect`; can be told to blow up on the Nth call."""

    def __init__(self, per_call=3, fail_after=None, error=None):
        self.per_call = per_call
        self.fail_after = fail_after
        self.error = error or KeyboardInterrupt("^C")
        self.calls = []

    def __call__(self, cfg, conn, ticker, query, start, end, apis, **kw):
        self.calls.append((ticker, start))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise self.error
        n = self.per_call(ticker) if callable(self.per_call) else self.per_call
        for i in range(n):
            db.upsert_news(conn, [{"url": f"http://x/{ticker}/{start}/{i}",
                                   "ticker": ticker, "title": "t",
                                   "published_utc": start, "api": "finnhub"}])
        return n


def seeded(cfg, conn, tickers=("AAPL", "MSFT"), weeks=(3, 9)):
    grid = window_grid(cfg)
    for t in tickers:
        for w in weeks:
            add_filing(conn, t, grid[w][0] + 80 * HOUR, accession=f"{t}-{w}")
    return backfill_targets(cfg, conn)


def test_resume_skips_completed_pairs(cfg, conn, monkeypatch):
    """A 15-hour run must not restart from the top."""
    targets = seeded(cfg, conn)
    monkeypatch.setattr(news, "collect", FakeCollect(fail_after=2))
    with pytest.raises(KeyboardInterrupt):
        collect_targets(cfg, conn, targets, ["finnhub"])

    second = FakeCollect()
    monkeypatch.setattr(news, "collect", second)
    collect_targets(cfg, conn, targets, ["finnhub"], resume=True)
    assert len(second.calls) == len(targets) - 2
    assert len(db.completed_keys(conn, FETCH_SOURCE)) == len(targets)


def test_resume_does_not_retry_a_quiet_week(cfg, conn, monkeypatch):
    """Inverted from P3-01 on purpose: a quiet week is a permanent answer.

    4,094 of 6,054 companies had no news at all in P2-09's sample week, so
    marking these retryable would make every resume re-fetch tens of thousands
    of weeks that will never have anything in them.
    """
    targets = seeded(cfg, conn)
    monkeypatch.setattr(news, "collect",
                        FakeCollect(per_call=lambda t: 0 if t == "MSFT" else 3))
    collect_targets(cfg, conn, targets, ["finnhub"])

    states = dict(conn.execute(
        "SELECT key, status FROM fetch_state WHERE source = ?", (FETCH_SOURCE,)))
    assert set(states.values()) == {"ok"}
    second = FakeCollect()
    monkeypatch.setattr(news, "collect", second)
    collect_targets(cfg, conn, targets, ["finnhub"], resume=True)
    assert second.calls == []


def test_resume_retries_a_failed_pair(cfg, conn, monkeypatch):
    targets = seeded(cfg, conn)
    monkeypatch.setattr(news, "collect",
                        FakeCollect(fail_after=1, error=RuntimeError("503")))
    collect_targets(cfg, conn, targets, ["finnhub"])
    failed = {k for k, in conn.execute(
        "SELECT key FROM fetch_state WHERE status = 'failed'")}
    assert failed

    second = FakeCollect()
    monkeypatch.setattr(news, "collect", second)
    collect_targets(cfg, conn, targets, ["finnhub"], resume=True)
    retried = {f"{t}@{i}" for t, i, s, _ in targets
               if (t, s) in second.calls}
    assert retried == failed


def test_state_is_committed_per_pair_not_at_the_end(cfg, conn, monkeypatch):
    targets = seeded(cfg, conn)
    monkeypatch.setattr(news, "collect", FakeCollect(fail_after=2))
    with pytest.raises(KeyboardInterrupt):
        collect_targets(cfg, conn, targets, ["finnhub"])
    assert len(db.completed_keys(conn, FETCH_SOURCE)) == 2


def test_news_is_written_before_state(cfg, conn, monkeypatch):
    """A crash between the two re-fetches one pair; the reverse loses rows."""
    seen = {}
    real = db.set_fetch_state

    def spy(c, source, key, status, **kw):
        seen[key] = c.execute("SELECT COUNT(*) FROM news").fetchone()[0]
        return real(c, source, key, status, **kw)

    targets = seeded(cfg, conn)
    monkeypatch.setattr(news.db, "set_fetch_state", spy)
    monkeypatch.setattr(news, "collect", FakeCollect())
    collect_targets(cfg, conn, targets, ["finnhub"])
    assert sorted(seen.values()) == [3, 6, 9, 12]


def test_keyboard_interrupt_stops_the_run(cfg, conn, monkeypatch):
    targets = seeded(cfg, conn)
    monkeypatch.setattr(news, "collect", FakeCollect(fail_after=0))
    with pytest.raises(KeyboardInterrupt):
        collect_targets(cfg, conn, targets, ["finnhub"])


# --------------------------------------------------------------------------
# the zero-record guard
# --------------------------------------------------------------------------

def test_guard_fires_when_every_attempted_pair_parsed_zero(cfg, conn,
                                                           monkeypatch):
    targets = seeded(cfg, conn)
    monkeypatch.setattr(news, "collect", FakeCollect(per_call=0))
    with pytest.raises(SystemExit, match="ZERO records"):
        collect_targets(cfg, conn, targets, ["finnhub"])


def test_a_run_that_parsed_nothing_leaves_its_pairs_resumable(cfg, conn,
                                                              monkeypatch):
    """The failure mode the guard exists for, survived in the worst way.

    A quiet week is recorded as a permanent `ok` on purpose — re-fetching tens
    of thousands of them on every resume would be absurd. But that `ok` used to
    be committed as each pair happened, BEFORE the run-level guard ran. So an
    expired Finnhub key answering HTTP 200 with `[]` for everything marked all
    N pairs `ok` and then raised loudly; the `--resume` the CLI recommends then
    skipped all N, attempted nothing, tripped no guard and exited 0 with `news`
    still empty — and the state permanently claimed those weeks were collected,
    so no later run would ever fetch them. Every filing in them silently loses
    its t0 correction, which is the project's headline contribution.
    """
    targets = seeded(cfg, conn)
    monkeypatch.setattr(news, "collect", FakeCollect(per_call=0))
    with pytest.raises(SystemExit, match="ZERO records"):
        collect_targets(cfg, conn, targets, ["finnhub"])

    # Nothing was banked, so the endpoint coming back is all it takes.
    assert db.completed_keys(conn, FETCH_SOURCE) == set()
    monkeypatch.setattr(news, "collect", FakeCollect())
    assert collect_targets(cfg, conn, targets, ["finnhub"], resume=True) > 0


def test_a_genuinely_quiet_week_is_still_banked_when_the_run_was_real(
        cfg, conn, monkeypatch):
    """The other half: buffering must not cost the permanent-answer rule.

    If the run as a whole parsed something, a pair that parsed nothing really
    was a quiet week, and a resume must not come back for it."""
    targets = seeded(cfg, conn)
    calls = []

    def one_loud_pair_then_silence(cfg_, conn_, ticker, query, start, end,
                                   apis, **kw):
        calls.append(ticker)
        if len(calls) > 1:
            return 0
        db.upsert_news(conn_, [{"url": "http://x/1", "ticker": ticker,
                                "title": "t", "published_utc": start,
                                "api": "finnhub"}])
        return 1

    monkeypatch.setattr(news, "collect", one_loud_pair_then_silence)
    collect_targets(cfg, conn, targets, ["finnhub"])
    assert db.completed_keys(conn, FETCH_SOURCE) == {f"{t[0]}@{t[1]}"
                                                    for t in targets}


def test_guard_silent_when_everything_was_skipped(cfg, conn, monkeypatch):
    """A completed backfill re-run attempted nothing, so it failed at nothing."""
    targets = seeded(cfg, conn)
    monkeypatch.setattr(news, "collect", FakeCollect())
    collect_targets(cfg, conn, targets, ["finnhub"])
    monkeypatch.setattr(news, "collect", FakeCollect(per_call=0))
    assert collect_targets(cfg, conn, targets, ["finnhub"], resume=True) == 0


# --------------------------------------------------------------------------
# P4-00b — continuous coverage for the Phase 8 ablation
# --------------------------------------------------------------------------

def add_universe_company(conn, ticker, in_universe=1):
    db.upsert_companies(conn, [{"cik": f"CIK{ticker}", "ticker": ticker,
                                "name": ticker, "in_universe": in_universe}])


def test_full_coverage_is_every_universe_ticker_times_every_week(cfg, conn):
    """THE Done-when: no week is skipped, so a zero count means 'nothing
    published' rather than 'nobody asked'."""
    for t in ("AAA", "BBB", "CCC"):
        add_universe_company(conn, t)
    weeks = len(window_grid(cfg))
    targets = full_coverage_targets(cfg, conn)
    assert len(targets) == 3 * weeks
    assert {t for t, _, _, _ in targets} == {"AAA", "BBB", "CCC"}
    assert {i for _, i, _, _ in targets} == set(range(weeks))


def test_full_coverage_excludes_non_universe_tickers(cfg, conn):
    """No feature is ever computed for a company outside the universe."""
    add_universe_company(conn, "IN")
    add_universe_company(conn, "OUT", in_universe=0)
    assert {t for t, _, _, _ in full_coverage_targets(cfg, conn)} == {"IN"}


def test_full_coverage_includes_tickers_with_no_filings(cfg, conn):
    """8 real universe members file nothing in the window — their quiet weeks
    are exactly the negatives Phase 8 needs."""
    add_universe_company(conn, "QUIET")
    assert backfill_targets(cfg, conn) == []          # no filings, so no backfill
    assert len(full_coverage_targets(cfg, conn)) == len(window_grid(cfg))


def test_full_coverage_reuses_the_backfill_window_grid(cfg, conn):
    """Shared `fetch_state` keys must mean the same spans in both modes, or
    resume would skip a pair that covered a different stretch of time."""
    add_universe_company(conn, "AAA")
    grid = window_grid(cfg)
    add_filing(conn, "AAA", grid[5][0] + 80 * HOUR)
    bf = {(i, s, e) for _, i, s, e in backfill_targets(cfg, conn)}
    fc = {(i, s, e) for _, i, s, e in full_coverage_targets(cfg, conn)}
    assert bf <= fc


def test_resume_skips_pairs_already_done_by_the_backfill(cfg, conn, monkeypatch):
    """The 16,467 pairs P4-00 already collected must not be re-fetched."""
    add_universe_company(conn, "AAA")
    grid = window_grid(cfg)
    add_filing(conn, "AAA", grid[5][0] + 80 * HOUR)
    monkeypatch.setattr(news, "collect", FakeCollect())
    collect_targets(cfg, conn, backfill_targets(cfg, conn), ["finnhub"])
    already = len(db.completed_keys(conn, FETCH_SOURCE))
    assert already > 0

    second = FakeCollect()
    monkeypatch.setattr(news, "collect", second)
    collect_targets(cfg, conn, full_coverage_targets(cfg, conn), ["finnhub"],
                    resume=True)
    assert len(second.calls) == len(window_grid(cfg)) - already
    assert len(db.completed_keys(conn, FETCH_SOURCE)) == len(window_grid(cfg))
