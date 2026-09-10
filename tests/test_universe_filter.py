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

    The scenario is built literally: GONE's bars run up to the window start,
    continue 180 days into the window and then stop. A filter keyed on recent
    trading must judge it by the last bar at or before the cutoff, never by the
    last bar it has. ALIVE is the control — both must survive together, or the
    test would also pass on a filter that keeps nothing but noise.
    """
    cutoff = as_of_ts(cfg)
    add_company(conn, "GONE")
    prior_8k(cfg, conn, "GONE")
    add_bars(conn, "GONE", cutoff - 400 * DAY, 400 + 180)   # stops mid-window
    liquid(cfg, conn, "ALIVE")
    survivors, reasons = select_universe(cfg, conn)
    assert sorted(c.ticker for c in survivors) == ["ALIVE", "GONE"]
    assert reasons == {}


def test_a_company_that_stopped_trading_before_the_window_is_excluded(cfg, conn):
    """The mirror image, and the one the as-of rule must NOT let through.

    Delisted six months BEFORE the start, it never trades during the window at
    all. Its ADV is a mean over the days it was alive, so it can still outrank
    a real company and take a capped slot.
    """
    cutoff = as_of_ts(cfg)
    add_company(conn, "DEAD")
    prior_8k(cfg, conn, "DEAD")
    add_bars(conn, "DEAD", cutoff - 400 * DAY, 220,        # last bar 181d back
             close=20.0, volume=12_000_000)
    assert classify(cfg, gather_candidates(cfg, conn)["DEAD"]) \
        == "not_trading_at_as_of"


def test_a_mid_window_ipo_has_no_bars_to_measure(cfg, conn):
    """The look-ahead rule's mirror: bars only after the start measure nothing."""
    cutoff = as_of_ts(cfg)
    add_company(conn, "IPO2")
    prior_8k(cfg, conn, "IPO2")
    add_bars(conn, "IPO2", cutoff + DAY, 300)
    assert classify(cfg, gather_candidates(cfg, conn)["IPO2"]) == "no_bars"


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
# exactly at each threshold
#
# Every threshold is documented as inclusive — "price >= $5", "ADV >= $5M" —
# and inclusive is what the report prints. Without a case sitting exactly on
# the line, `<` and `<=` are indistinguishable and the printed claim is not
# checked by anything. Each pair below is a value on the line (must qualify)
# and its nearest neighbour on the wrong side (must not).
# --------------------------------------------------------------------------

def test_a_price_exactly_at_the_minimum_qualifies(cfg, conn):
    liquid(cfg, conn, "EDGE", close=cfg["universe"]["min_price_usd"],
           volume=2_000_000)
    assert classify(cfg, gather_candidates(cfg, conn)["EDGE"]) is None


def test_a_price_a_cent_below_the_minimum_is_excluded(cfg, conn):
    liquid(cfg, conn, "EDGE", close=cfg["universe"]["min_price_usd"] - 0.01,
           volume=2_000_000)
    assert classify(cfg, gather_candidates(cfg, conn)["EDGE"]) == "below_min_price"


def test_an_adv_exactly_at_the_minimum_qualifies(cfg, conn):
    """close * volume lands on min_adv_usd to the cent, on every bar."""
    liquid(cfg, conn, "EDGE", close=10.0,
           volume=cfg["universe"]["min_adv_usd"] // 10)
    cand = gather_candidates(cfg, conn)["EDGE"]
    assert cand.adv_usd == cfg["universe"]["min_adv_usd"]
    assert classify(cfg, cand) is None


def test_an_adv_a_hair_below_the_minimum_is_excluded(cfg, conn):
    liquid(cfg, conn, "EDGE", close=10.0,
           volume=cfg["universe"]["min_adv_usd"] // 10 - 1)
    assert classify(cfg, gather_candidates(cfg, conn)["EDGE"]) == "below_min_adv"


def history_boundary(cfg):
    """The earliest first bar that still counts as a full year of history."""
    return (as_of_ts(cfg) - cfg["universe"]["min_history_days"] * DAY
            + cfg["market"]["coverage_tolerance_days"] * DAY)


def test_history_exactly_at_the_boundary_qualifies(cfg, conn):
    cutoff, first = as_of_ts(cfg), history_boundary(cfg)
    add_company(conn, "EDGE")
    prior_8k(cfg, conn, "EDGE")
    add_bars(conn, "EDGE", first, (cutoff - first) // DAY + 1)
    assert classify(cfg, gather_candidates(cfg, conn)["EDGE"]) is None


def test_history_one_day_short_of_the_boundary_is_excluded(cfg, conn):
    cutoff, first = as_of_ts(cfg), history_boundary(cfg) + DAY
    add_company(conn, "EDGE")
    prior_8k(cfg, conn, "EDGE")
    add_bars(conn, "EDGE", first, (cutoff - first) // DAY + 1)
    assert classify(cfg, gather_candidates(cfg, conn)["EDGE"]) == "short_history"


def sparse(cfg, conn, ticker, n_bars, **kw):
    """Enough history to clear `min_history_days`, but only `n_bars` recent."""
    cutoff = as_of_ts(cfg)
    add_company(conn, ticker)
    prior_8k(cfg, conn, ticker)
    add_bars(conn, ticker, cutoff - 400 * DAY, 1, **kw)   # outside the lookback
    add_bars(conn, ticker, cutoff - (n_bars - 1) * DAY, n_bars, **kw)


def test_exactly_the_minimum_bar_count_qualifies(cfg, conn):
    sparse(cfg, conn, "EDGE", cfg["universe"]["min_bars_in_lookback"],
           close=10.0, volume=1_000_000)
    assert classify(cfg, gather_candidates(cfg, conn)["EDGE"]) is None


def test_one_bar_short_of_the_minimum_is_excluded(cfg, conn):
    """An ADV averaged over a handful of days is not an average daily volume."""
    sparse(cfg, conn, "EDGE", cfg["universe"]["min_bars_in_lookback"] - 1,
           close=10.0, volume=1_000_000)
    assert classify(cfg, gather_candidates(cfg, conn)["EDGE"]) == "too_few_bars"


def test_a_last_bar_exactly_at_the_staleness_limit_qualifies(cfg, conn):
    cutoff = as_of_ts(cfg)
    stale = cfg["universe"]["max_bar_staleness_days"]
    add_company(conn, "EDGE")
    prior_8k(cfg, conn, "EDGE")
    add_bars(conn, "EDGE", cutoff - 400 * DAY, 400 - stale + 1)
    assert classify(cfg, gather_candidates(cfg, conn)["EDGE"]) is None


def test_a_last_bar_one_day_past_the_staleness_limit_is_excluded(cfg, conn):
    cutoff = as_of_ts(cfg)
    stale = cfg["universe"]["max_bar_staleness_days"]
    add_company(conn, "EDGE")
    prior_8k(cfg, conn, "EDGE")
    add_bars(conn, "EDGE", cutoff - 400 * DAY, 400 - stale)
    assert classify(cfg, gather_candidates(cfg, conn)["EDGE"]) \
        == "not_trading_at_as_of"


def test_bars_with_no_price_data_get_their_own_reason(cfg, conn):
    """A broken collector cycle must not read as "trades too thinly".

    NULL closes never reach the real `bars` table today, and a NaN written from
    Python arrives as NULL. Either way the aggregate is NULL, not small, and
    the report should say so rather than blaming the company.
    """
    cutoff = as_of_ts(cfg)
    add_company(conn, "NULLS")
    prior_8k(cfg, conn, "NULLS")
    add_bars(conn, "NULLS", cutoff - 400 * DAY, 1)         # history, real bar
    db.upsert_bars(conn, [("NULLS", cutoff - i * DAY, None, None, None, None,
                           None, "1d") for i in range(300)])
    assert classify(cfg, gather_candidates(cfg, conn)["NULLS"]) == "no_usable_bars"


def test_a_sparse_or_dead_name_never_outranks_a_continuously_traded_one(cfg, conn):
    """The cap is a scarce resource, and a mean over four days can win it.

    SPARSE traded four days all year and ranks first on ADV; DEAD stopped six
    months before the window. Both used to take slots from ALIVE, which traded
    every session.
    """
    cfg = {**cfg, "universe": {**cfg["universe"], "max_tickers": 2}}
    cutoff = as_of_ts(cfg)
    sparse(cfg, conn, "SPARSE", 4, close=50.0, volume=20_000_000)
    add_company(conn, "DEAD")
    prior_8k(cfg, conn, "DEAD")
    add_bars(conn, "DEAD", cutoff - 400 * DAY, 220, close=20.0,
             volume=12_000_000)
    liquid(cfg, conn, "ALIVE", close=10.0, volume=1_200_000)

    survivors, reasons = select_universe(cfg, conn)
    assert [c.ticker for c in survivors] == ["ALIVE"]
    assert reasons == {"too_few_bars": 1, "not_trading_at_as_of": 1}


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


def test_a_survivor_that_cannot_be_flagged_fails_loudly_and_writes_nothing(
        cfg, conn):
    """Two guarantees in one run: the count is checked, and the write is atomic.

    GHOST clears every threshold but carries no primary row — every row with
    its ticker is a predecessor — so the UPDATE matches nothing and it would
    vanish from the study with only a log line one short. And because the clear
    and the set share a transaction, the failure leaves the previous run's
    flags exactly as they were: the alternative is `in_universe = 0` on every
    row, which every later stage reads as an empty universe and calls success.
    """
    liquid(cfg, conn, "AAA")
    apply_filter(cfg, conn)
    assert db.universe_tickers(conn) == ["AAA"]

    add_company(conn, "GHOST", cik="OLDGHOST", successor="CIKGHOST")
    prior_8k(cfg, conn, "GHOST")
    add_bars(conn, "GHOST", as_of_ts(cfg) - 395 * DAY, 500)

    with pytest.raises(SystemExit, match="write mismatch") as err:
        apply_filter(cfg, conn)
    assert "GHOST" in str(err.value)
    assert db.universe_tickers(conn) == ["AAA"]   # nothing cleared, nothing set


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


def test_a_decade_old_8k_does_not_make_a_trust_a_filer(cfg, conn):
    """QQQ, literally: two administrative filings in 2014 and nothing since.

    "Ever filed an 8-K since 1994" is not the rule the config describes. A
    grantor trust that files sponsor paperwork once a decade can never produce
    the kind of event this project detects, yet it is liquid enough to take a
    capped slot from a real filer — the #3 slot by ADV, in the built universe.
    """
    liquid(cfg, conn, "QQQ", files_8k=False, volume=50_000_000)
    add_filing(conn, "QQQ", as_of_ts(cfg) - 4000 * DAY)
    assert classify(cfg, gather_candidates(cfg, conn)["QQQ"]) == "no_prior_8k"


def test_an_8k_exactly_at_the_recency_boundary_still_counts(cfg, conn):
    liquid(cfg, conn, "OLD", files_8k=False)
    add_filing(conn, "OLD",
               as_of_ts(cfg) - cfg["universe"]["prior_8k_lookback_days"] * DAY)
    assert classify(cfg, gather_candidates(cfg, conn)["OLD"]) is None


def test_an_8k_a_day_older_than_the_boundary_does_not(cfg, conn):
    liquid(cfg, conn, "OLD", files_8k=False)
    add_filing(conn, "OLD", as_of_ts(cfg)
               - (cfg["universe"]["prior_8k_lookback_days"] + 1) * DAY)
    assert classify(cfg, gather_candidates(cfg, conn)["OLD"]) == "no_prior_8k"


def test_a_candidate_must_carry_its_filing_evidence(cfg):
    """Fail closed: no default may let an entity qualify on silence.

    `files_8k` decides membership. A default of True means any Candidate built
    without filing evidence — a fixture, a future caller — silently passes the
    one rule that keeps entities outside the answer key out of the study.
    """
    with pytest.raises(TypeError):
        Candidate("ZZZ", 0, 0, 50.0, 1e8, 300)


# --------------------------------------------------------------------------
# The stored universe can drift away from the config that describes it
# --------------------------------------------------------------------------

def test_drift_is_reported_when_the_config_no_longer_selects_the_stored_set(cfg,
                                                                           conn):
    """`companies.in_universe` is written once and read by every later stage.
    Nothing re-derives it, so a config change after the write leaves code and
    data disagreeing in silence — which is what happened when
    `prior_8k_lookback_days` was added and the filter was never re-run."""
    from src.pipeline.universe import check_drift

    liquid(cfg, conn, "AAA")
    liquid(cfg, conn, "BBB")
    apply_filter(cfg, conn)
    assert check_drift(cfg, conn)["drift"] == 0

    # A company qualifies now that did not when the flags were written.
    liquid(cfg, conn, "CCC")
    d = check_drift(cfg, conn)
    assert d["drift"] > 0
    assert "CCC" in d["computed_only"]


def test_drift_check_writes_nothing(cfg, conn):
    """It is the mode you run to find out where you stand, so it must not
    quietly move you somewhere else."""
    from src.pipeline.universe import check_drift

    liquid(cfg, conn, "AAA")
    apply_filter(cfg, conn)
    before = {r[0] for r in conn.execute(
        "SELECT ticker FROM companies WHERE in_universe = 1")}
    liquid(cfg, conn, "CCC")
    check_drift(cfg, conn)
    after = {r[0] for r in conn.execute(
        "SELECT ticker FROM companies WHERE in_universe = 1")}
    assert before == after
