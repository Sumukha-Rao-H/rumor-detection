"""P3-02 — the liquidity filter, and the survivorship trap it exists to avoid.

The test that matters most is
`test_a_company_delisted_mid_window_still_qualifies`: a universe built from
today's data silently drops every company acquired or delisted during the
window, which is exactly the dramatic population this project detects. Every
measurement here must come from bars at or before the window start, and none
from after it.
"""

import pytest

from src import db
from src.pipeline.universe import (
    Candidate, apply_filter, as_of_ts, classify, gather_candidates,
    select_universe,
)
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts


DAY = 86400


@pytest.fixture
def cfg():
    cfg = load_config()
    cfg["universe"] = {**cfg["universe"], "max_tickers": 10}
    return cfg


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "u.db")


def add_company(conn, ticker, cik=None, successor=None):
    db.upsert_companies(conn, [{"cik": cik or f"CIK{ticker}", "ticker": ticker,
                                "name": ticker, "successor_cik": successor}])


def add_filing(conn, ticker, acceptance_utc, form="8-K", accession=None):
    db.upsert_filings(conn, [{
        "accession_no": accession or f"{ticker}-{acceptance_utc}-{form}",
        "cik": f"CIK{ticker}", "ticker": ticker, "form": form, "items": "1.01",
        "acceptance_utc": acceptance_utc, "filing_date_utc": acceptance_utc,
    }])


def prior_8k(cfg, conn, ticker, form="8-K"):
    """An 8-K a month before the window — what makes a real filer eligible."""
    add_filing(conn, ticker, as_of_ts(cfg) - 30 * DAY, form=form)


def add_bars(conn, ticker, start_ts, days, close=10.0, volume=1_000_000,
             interval="1d"):
    """`days` consecutive daily bars, one per calendar day (weekends included).

    Calendar days rather than trading days on purpose: these tests are about
    the date arithmetic in the filter, not about the exchange calendar.
    """
    db.upsert_bars(conn, [
        (ticker, start_ts + i * DAY, close, close, close, close, volume,
         interval)
        for i in range(days)
    ])


def liquid(cfg, conn, ticker, files_8k=True, **kw):
    """A company that comfortably clears every threshold."""
    add_company(conn, ticker)
    if files_8k:
        prior_8k(cfg, conn, ticker)
    start = as_of_ts(cfg) - (cfg["universe"]["min_history_days"] + 30) * DAY
    add_bars(conn, ticker, start, kw.pop("days", 500), **kw)


# --------------------------------------------------------------------------
# the as-of rule — the whole point of the task
# --------------------------------------------------------------------------

def test_a_company_delisted_mid_window_still_qualifies(cfg, conn):
    """THE test. Liquid at the window start, dead six months later — it stays.

    Being acquired or delisted IS the market-moving event this project detects.
    A filter that drops those companies removes the most interesting rows in
    the study and leaves no trace that it did.
    """
    cutoff = as_of_ts(cfg)
    add_company(conn, "GONE")
    prior_8k(cfg, conn, "GONE")
    add_bars(conn, "GONE", cutoff - 500 * DAY, 500)   # last bar at the cutoff
    survivors, _ = select_universe(cfg, conn)
    assert [c.ticker for c in survivors] == ["GONE"]


def test_bars_after_the_window_start_are_ignored(cfg, conn):
    """A post-start volume explosion must not change the measurement."""
    cutoff = as_of_ts(cfg)
    liquid(cfg, conn, "AAA")
    before = gather_candidates(cfg, conn)["AAA"]
    add_bars(conn, "AAA", cutoff + DAY, 200, close=999.0, volume=10**9)
    after = gather_candidates(cfg, conn)["AAA"]
    assert before == after


def test_min_price_uses_the_last_close_before_the_window(cfg, conn):
    """Not today's price: a stock can be $2 now and have been $50 at the start."""
    cutoff = as_of_ts(cfg)
    liquid(cfg, conn, "AAA", close=50.0)
    add_bars(conn, "AAA", cutoff + DAY, 100, close=1.0)   # crashes later
    assert classify(cfg, gather_candidates(cfg, conn)["AAA"]) is None


# --------------------------------------------------------------------------
# each knob
# --------------------------------------------------------------------------

def test_adv_is_close_times_volume_averaged(cfg, conn):
    liquid(cfg, conn, "AAA", close=20.0, volume=2_000_000)
    assert gather_candidates(cfg, conn)["AAA"].adv_usd == pytest.approx(4e7)


def test_min_adv_excludes_a_thin_stock(cfg, conn):
    liquid(cfg, conn, "THIN", close=10.0, volume=400)
    assert classify(cfg, gather_candidates(cfg, conn)["THIN"]) == "below_min_adv"


def test_min_price_excludes_a_penny_stock(cfg, conn):
    liquid(cfg, conn, "PENNY", close=1.0, volume=50_000_000)
    assert classify(cfg, gather_candidates(cfg, conn)["PENNY"]) == "below_min_price"


def test_min_history_excludes_a_recent_ipo(cfg, conn):
    add_company(conn, "IPO")
    prior_8k(cfg, conn, "IPO")
    add_bars(conn, "IPO", as_of_ts(cfg) - 60 * DAY, 60)
    assert classify(cfg, gather_candidates(cfg, conn)["IPO"]) == "short_history"


def test_a_company_with_no_bars_is_excluded(cfg, conn):
    add_company(conn, "SVA")
    prior_8k(cfg, conn, "SVA")
    assert classify(cfg, gather_candidates(cfg, conn)["SVA"]) == "no_bars"


def test_history_boundary_tolerates_a_weekend(cfg, conn):
    """The trap that returned zero companies during P3-01.

    `start - 365d` was a Sunday and the next day a holiday, so the earliest
    possible bar is days later and a strict comparison excludes everyone.
    """
    cutoff = as_of_ts(cfg)
    tol = cfg["market"]["coverage_tolerance_days"]
    first = cutoff - cfg["universe"]["min_history_days"] * DAY + (tol - 1) * DAY
    add_company(conn, "AAA")
    prior_8k(cfg, conn, "AAA")
    add_bars(conn, "AAA", first, 400)
    assert classify(cfg, gather_candidates(cfg, conn)["AAA"]) is None


# --------------------------------------------------------------------------
# the cap
# --------------------------------------------------------------------------

def test_max_tickers_keeps_the_most_liquid(cfg, conn):
    for i in range(15):
        liquid(cfg, conn, f"T{i:02d}", volume=1_000_000 * (i + 1))
    survivors, reasons = select_universe(cfg, conn)
    assert len(survivors) == 10
    assert survivors[0].ticker == "T14"          # most liquid first
    assert reasons["over_max_tickers"] == 5
    assert "T00" not in {c.ticker for c in survivors}


def test_max_tickers_cut_is_deterministic_on_ties(cfg, conn):
    for i in range(15):
        liquid(cfg, conn, f"T{i:02d}")           # identical ADV throughout
    first = [c.ticker for c in select_universe(cfg, conn)[0]]
    second = [c.ticker for c in select_universe(cfg, conn)[0]]
    assert first == second == sorted(first)


# --------------------------------------------------------------------------
# writing the flags
# --------------------------------------------------------------------------

def test_apply_filter_populates_in_universe_and_adv(cfg, conn):
    """The Done-when, literally."""
    liquid(cfg, conn, "AAA", close=20.0, volume=2_000_000)
    liquid(cfg, conn, "THIN", close=10.0, volume=400)
    apply_filter(cfg, conn)

    row = conn.execute("SELECT in_universe, adv_usd, last_price, universe_as_of "
                       "FROM companies WHERE ticker = 'AAA'").fetchone()
    assert row["in_universe"] == 1
    assert row["adv_usd"] == pytest.approx(4e7)
    assert row["last_price"] == pytest.approx(20.0)
    assert row["universe_as_of"] == as_of_ts(cfg)
    assert db.universe_tickers(conn) == ["AAA"]


def test_predecessor_rows_are_not_flagged(cfg, conn):
    """A reorganised company must not be counted twice."""
    liquid(cfg, conn, "XOM")
    add_company(conn, "XOM", cik="OLDXOM", successor="CIKXOM")
    apply_filter(cfg, conn)
    flags = [r[0] for r in conn.execute(
        "SELECT in_universe FROM companies WHERE ticker = 'XOM' "
        "ORDER BY successor_cik IS NULL DESC")]
    assert flags == [1, 0]


def test_rerun_demotes_a_company_that_no_longer_qualifies(cfg, conn):
    """Flags are rebuilt, not accumulated."""
    liquid(cfg, conn, "AAA")
    liquid(cfg, conn, "BBB")
    apply_filter(cfg, conn)
    assert set(db.universe_tickers(conn)) == {"AAA", "BBB"}

    conn.execute("DELETE FROM bars WHERE ticker = 'BBB'")
    conn.commit()
    apply_filter(cfg, conn)
    assert db.universe_tickers(conn) == ["AAA"]


def test_raises_when_nothing_survives(cfg, conn):
    liquid(cfg, conn, "THIN", close=10.0, volume=400)
    with pytest.raises(SystemExit, match="ZERO companies"):
        apply_filter(cfg, conn)


def test_raises_on_as_of_other_than_start(cfg):
    cfg = {**cfg, "universe": {**cfg["universe"], "as_of": "today"}}
    with pytest.raises(ValueError, match="only 'start' is supported"):
        as_of_ts(cfg)


def test_every_candidate_is_either_kept_or_given_one_reason(cfg, conn):
    """The report has to add up, or the surviving count is not defensible."""
    liquid(cfg, conn, "AAA")
    liquid(cfg, conn, "THIN", volume=400)
    liquid(cfg, conn, "PENNY", close=1.0, volume=50_000_000)
    add_company(conn, "IPO")
    prior_8k(cfg, conn, "IPO")
    add_bars(conn, "IPO", as_of_ts(cfg) - 60 * DAY, 60)
    add_company(conn, "NOBARS")
    prior_8k(cfg, conn, "NOBARS")

    survivors, reasons = select_universe(cfg, conn)
    assert len(survivors) + sum(reasons.values()) == len(db.candidate_tickers(conn))
    assert set(reasons) == {"below_min_adv", "below_min_price", "short_history",
                            "no_bars"}


# --------------------------------------------------------------------------
# P3-02b — entities that cannot file an 8-K
# --------------------------------------------------------------------------

def test_an_etf_with_no_prior_8k_is_excluded(cfg, conn):
    """SPY, QQQ and the foreign private issuers can never produce an event."""
    liquid(cfg, conn, "SPY", files_8k=False, volume=50_000_000)
    assert classify(cfg, gather_candidates(cfg, conn)["SPY"]) == "no_prior_8k"


def test_a_company_with_a_prior_8k_is_kept(cfg, conn):
    liquid(cfg, conn, "AAPL")
    assert classify(cfg, gather_candidates(cfg, conn)["AAPL"]) is None


def test_only_filings_before_the_window_count(cfg, conn):
    """No look-ahead: an in-window 8-K must not rescue a company.

    Keying on in-window filings would build the universe out of the outcome and
    hand every surviving member a guaranteed positive.
    """
    liquid(cfg, conn, "NEW", files_8k=False)
    add_filing(conn, "NEW", as_of_ts(cfg) + 30 * DAY)      # during the window
    assert classify(cfg, gather_candidates(cfg, conn)["NEW"]) == "no_prior_8k"


def test_an_amendment_counts_as_a_prior_filing(cfg, conn):
    """8-K/A is a real filing with its own acceptance time (P2-04)."""
    liquid(cfg, conn, "AMD8", files_8k=False)
    prior_8k(cfg, conn, "AMD8", form="8-K/A")
    assert classify(cfg, gather_candidates(cfg, conn)["AMD8"]) is None


def test_a_non_8k_form_does_not_qualify(cfg, conn):
    """A 10-Q filer that never files 8-Ks is still outside the answer key."""
    liquid(cfg, conn, "TENQ", files_8k=False)
    prior_8k(cfg, conn, "TENQ", form="10-Q")
    assert classify(cfg, gather_candidates(cfg, conn)["TENQ"]) == "no_prior_8k"


def test_the_rule_can_be_switched_off(cfg, conn):
    """'What if you keep them?' should cost one line to answer."""
    liquid(cfg, conn, "SPY", files_8k=False)
    off = {**cfg, "universe": {**cfg["universe"], "require_prior_8k": False}}
    assert classify(off, gather_candidates(off, conn)["SPY"]) is None


def test_excluded_slots_are_backfilled_to_the_cap(cfg, conn):
    """The user's decision, literally: real filers take the freed slots."""
    for i in range(5):                      # most liquid, but cannot file
        liquid(cfg, conn, f"ETF{i}", files_8k=False, volume=90_000_000)
    for i in range(12):                     # real filers, less liquid
        liquid(cfg, conn, f"CO{i:02d}", volume=1_000_000 * (i + 1))

    survivors, reasons = select_universe(cfg, conn)
    assert len(survivors) == cfg["universe"]["max_tickers"] == 10
    assert not any(t.ticker.startswith("ETF") for t in survivors)
    assert reasons["no_prior_8k"] == 5
    assert reasons["over_max_tickers"] == 2   # 12 real filers, 10 slots


def test_no_prior_8k_is_reported_before_liquidity_reasons(cfg, conn):
    """An entity outside the answer key is excluded for that, not for being thin."""
    liquid(cfg, conn, "THINETF", files_8k=False, volume=400)
    assert classify(cfg, gather_candidates(cfg, conn)["THINETF"]) == "no_prior_8k"
