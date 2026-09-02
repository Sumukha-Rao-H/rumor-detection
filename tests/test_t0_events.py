"""P4-02 — the analysis unit and its clock.

`events` carries three timestamps side by side, and they must never collapse
into one. That is a frozen decision: keeping them apart is what lets the whole
evaluation be re-run under either t0 variant by pointing at a different column,
and what keeps the acceptance-minus-news gap inspectable — the gap being itself
a result, not an intermediate.

The re-run tests matter more than they look. The lookback changed 24h -> 3h on
the day this was written, so "rebuilding corrects rows rather than duplicating
them" is a property the project already depends on — and "corrects" has to mean
the t0 columns ONLY. `db.upsert_events` writes every column of every row it is
handed, so a rebuild that forgets the P4-03/P4-04 columns NULLs them and resets
`usable` to 0, which empties the feature matrix without a word.
"""

import pytest

from src import db
from src.pipeline.t0 import (
    build_events, build_provenance, event_id_for, prune_events, write_events,
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
    return db.get_conn(tmp_path / "ev.db")


def seed(conn, ticker="AAPL", accession="0000320193-26-000018",
         acceptance_iso=ACCEPTANCE, items="2.02", in_universe=1, form="8-K"):
    db.upsert_companies(conn, [{"cik": f"CIK{ticker}", "ticker": ticker,
                                "in_universe": in_universe}])
    ts = iso_utc_to_ts(acceptance_iso)
    db.upsert_filings(conn, [{
        "accession_no": accession, "cik": f"CIK{ticker}", "ticker": ticker,
        "form": form, "items": items, "acceptance_utc": ts,
        "filing_date_utc": ts,
    }])
    return ts


def add_article(conn, ticker, at_utc, tier=2):
    db.upsert_news(conn, [{
        "url": f"http://x/{ticker}/{at_utc}", "ticker": ticker, "title": "t",
        "source_name": "Benzinga", "source_domain": "x.com", "source_tier": tier,
        "published_utc": at_utc, "api": "finnhub",
    }])
    return at_utc


def only_event(conn):
    return conn.execute("SELECT * FROM events").fetchone()


# --------------------------------------------------------------------------
# the three clocks
# --------------------------------------------------------------------------

def test_all_three_timestamps_are_stored_separately(cfg, conn):
    """THE frozen decision. Collapsing them would push a variant switch into
    every metric and discard the gap, which is itself a reported result."""
    acc = seed(conn)
    news = add_article(conn, "AAPL", acc - 45 * 60)
    write_events(cfg, conn)

    e = only_event(conn)
    assert e["t0_filing_utc"] == acc
    assert e["t0_news_utc"] == news
    assert e["t0_utc"] == news
    assert e["t0_source"] == "news"
    assert e["t0_filing_utc"] != e["t0_news_utc"]      # genuinely distinct


def test_t0_is_the_minimum_of_the_two(cfg, conn):
    acc = seed(conn)
    news = add_article(conn, "AAPL", acc - 30 * 60)
    write_events(cfg, conn)
    e = only_event(conn)
    assert e["t0_utc"] == min(e["t0_filing_utc"], e["t0_news_utc"])


def test_unmatched_filing_falls_back_to_acceptance(cfg, conn):
    """Not a failure — the uncorrected baseline every prior paper uses.

    Both variants must cover the same population, or they describe two
    different studies.
    """
    acc = seed(conn)
    write_events(cfg, conn)
    e = only_event(conn)
    assert e["t0_news_utc"] is None
    assert e["t0_utc"] == acc == e["t0_filing_utc"]
    assert e["t0_source"] == "filing"


def test_an_article_exactly_at_acceptance_still_counts_as_news(cfg, conn):
    """t0 is unchanged, but an article did match — the count stays honest."""
    acc = seed(conn)
    add_article(conn, "AAPL", acc)
    write_events(cfg, conn)
    e = only_event(conn)
    assert e["t0_utc"] == acc and e["t0_source"] == "news"


# --------------------------------------------------------------------------
# identity and re-running
# --------------------------------------------------------------------------

def test_event_id_is_the_accession_without_punctuation(cfg, conn):
    seed(conn, accession="0000320193-26-000018")
    write_events(cfg, conn)
    assert only_event(conn)["event_id"] == "000032019326000018"
    assert event_id_for("0000320193-26-000018") == "000032019326000018"


def test_rebuilding_updates_rather_than_duplicates(cfg, conn):
    """Re-running must correct the study, not create a second copy of it."""
    acc = seed(conn)
    add_article(conn, "AAPL", acc - 30 * 60)
    write_events(cfg, conn)
    written, new = write_events(cfg, conn)
    assert written == 1 and new == 0
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_rebuilding_after_a_lookback_change_moves_t0(cfg, conn):
    """The exact situation this project hit: the window narrowed 24h -> 3h, so
    events built earlier carried a stale t0 and had to be corrected in place."""
    acc = seed(conn)
    old_match = add_article(conn, "AAPL", acc - 10 * HOUR)

    wide = {**cfg, "news": {**cfg["news"], "t0_lookback_hours": 24}}
    write_events(wide, conn)
    assert only_event(conn)["t0_utc"] == old_match

    narrow = {**cfg, "news": {**cfg["news"], "t0_lookback_hours": 3}}
    write_events(narrow, conn)
    e = only_event(conn)
    assert e["t0_utc"] == acc and e["t0_source"] == "filing"
    assert e["t0_news_utc"] is None
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


# --------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------

def test_amendments_get_their_own_event(cfg, conn):
    """8-K/A is a real filing with its own acceptance time (P2-04)."""
    seed(conn, accession="a-1", form="8-K")
    seed(conn, accession="a-2", form="8-K/A",
         acceptance_iso="2026-03-01T20:30:00Z")
    assert len(build_events(cfg, conn)) == 2


def test_filings_outside_the_universe_are_not_built(cfg, conn):
    seed(conn, ticker="IN", accession="in-1")
    seed(conn, ticker="OUT", accession="out-1", in_universe=0)
    assert {r["ticker"] for r in build_events(cfg, conn)} == {"IN"}


def test_a_non_8k_form_is_not_an_event(cfg, conn):
    seed(conn, accession="q-1", form="10-Q")
    with pytest.raises(SystemExit, match="no events"):
        build_events(cfg, conn)


def test_filtering_columns_are_left_for_later_phases(cfg, conn):
    """P4-03 does item filtering, P4-04 materiality. Not this task's business."""
    seed(conn)
    write_events(cfg, conn)
    e = only_event(conn)
    assert e["is_material"] is None and e["exclude_reason"] is None
    assert e["usable"] == 0


def test_raises_when_no_event_can_be_built(cfg, conn):
    with pytest.raises(SystemExit, match="no events could be built"):
        build_events(cfg, conn)


# --------------------------------------------------------------------------
# the gap — the Done-when's number
# --------------------------------------------------------------------------

def test_gap_is_recoverable_from_the_stored_columns(cfg, conn):
    """The reported gap must come from what was stored, not a recomputation."""
    acc = seed(conn, accession="g-1")
    add_article(conn, "AAPL", acc - 45 * 60)
    seed(conn, ticker="BBB", accession="g-2")        # unmatched
    write_events(cfg, conn)

    rows = conn.execute(
        "SELECT t0_filing_utc, t0_news_utc, t0_source FROM events").fetchall()
    gaps = [(r["t0_filing_utc"] - r["t0_news_utc"]) / HOUR
            for r in rows if r["t0_source"] == "news"]
    assert len(gaps) == 1 and gaps[0] == pytest.approx(0.75)


# --------------------------------------------------------------------------
# a rebuild must not damage what other phases stored
# --------------------------------------------------------------------------

def classify(conn, event_id="000032019326000018", **cols):
    """Stand in for P4-03/P4-04, which write these columns with UPDATE."""
    cols = cols or {"is_scheduled": 1, "abs_return": 0.07, "is_material": 1,
                    "usable": 1, "exclude_reason": None}
    sets = ", ".join(f"{c} = ?" for c in cols)
    conn.execute(f"UPDATE events SET {sets} WHERE event_id = ?",
                 (*cols.values(), event_id))
    conn.commit()


def test_rebuilding_preserves_the_filter_columns(cfg, conn):
    """THE destructive bug. `--build` is documented as safe to re-run after a
    config change; before this it reset every classified event to
    `usable = 0, is_scheduled = NULL`, and `features.py` selects
    `WHERE usable = 1` — so the whole feature matrix came back empty."""
    acc = seed(conn)
    add_article(conn, "AAPL", acc - 30 * 60)
    write_events(cfg, conn)
    classify(conn)

    write_events(cfg, conn)                      # the rebuild

    e = only_event(conn)
    assert e["usable"] == 1
    assert e["is_scheduled"] == 1
    assert e["abs_return"] == pytest.approx(0.07)
    assert e["is_material"] == 1


def test_a_rebuild_that_moves_t0_still_preserves_them(cfg, conn):
    """The realistic case: the columns survive a rebuild that genuinely
    changes t0, not only a no-op re-run."""
    acc = seed(conn)
    old_match = add_article(conn, "AAPL", acc - 10 * HOUR)
    wide = {**cfg, "news": {**cfg["news"], "t0_lookback_hours": 24}}
    write_events(wide, conn)
    assert only_event(conn)["t0_utc"] == old_match
    classify(conn, usable=1, exclude_reason=None, is_scheduled=0)

    write_events({**cfg, "news": {**cfg["news"], "t0_lookback_hours": 3}}, conn)

    e = only_event(conn)
    assert e["t0_utc"] == acc and e["t0_source"] == "filing"   # t0 corrected
    assert e["usable"] == 1 and e["is_scheduled"] == 0         # verdict kept


def test_an_exclusion_verdict_survives_a_rebuild(cfg, conn):
    """A rejected event must stay rejected, with its reason intact."""
    seed(conn)
    write_events(cfg, conn)
    classify(conn, usable=0, exclude_reason="immaterial")
    write_events(cfg, conn)
    e = only_event(conn)
    assert e["usable"] == 0 and e["exclude_reason"] == "immaterial"


def test_a_new_event_still_starts_unusable(cfg, conn):
    """Preserving must not mean inventing: an event nobody has classified yet
    is `usable = 0`, explicitly, not NULL."""
    seed(conn, accession="a-1")
    write_events(cfg, conn)
    classify(conn, event_id=event_id_for("a-1"), usable=1)

    seed(conn, ticker="BBB", accession="b-1")     # arrives later
    write_events(cfg, conn)

    rows = {r["event_id"]: r for r in conn.execute("SELECT * FROM events")}
    assert rows[event_id_for("a-1")]["usable"] == 1
    assert rows[event_id_for("b-1")]["usable"] == 0
    assert rows[event_id_for("b-1")]["is_scheduled"] is None


# --------------------------------------------------------------------------
# a rebuild must also remove what no longer belongs
# --------------------------------------------------------------------------

def test_events_outside_a_narrowed_window_are_pruned(cfg, conn):
    """`study_window.start` really did move a year forward. Without pruning,
    events built under the old window sit in `events` forever with their old
    `usable` flag and inflate every rate read back from the table."""
    seed(conn, accession="old-1", acceptance_iso="2025-10-01T20:30:00Z")
    seed(conn, ticker="BBB", accession="new-1",
         acceptance_iso="2026-02-25T20:30:00Z")
    write_events(cfg, conn)
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2

    later = {**cfg, "study_window": {**cfg["study_window"],
                                     "start": "2026-01-01"}}
    write_events(later, conn)

    assert [r["accession_no"] for r in conn.execute(
        "SELECT accession_no FROM events")] == ["new-1"]


def test_events_dropped_from_the_universe_are_pruned(cfg, conn):
    """Same for a company that no longer passes the liquidity filter."""
    seed(conn, ticker="AAA", accession="a-1")
    seed(conn, ticker="OUT", accession="o-1")
    write_events(cfg, conn)
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2

    conn.execute("UPDATE companies SET in_universe = 0 WHERE ticker = 'OUT'")
    conn.commit()
    write_events(cfg, conn)

    assert [r["ticker"] for r in conn.execute("SELECT ticker FROM events")] \
        == ["AAA"]


def test_prune_reports_how_many_it_removed(cfg, conn):
    """A silent deletion is worse than none — the count is the audit trail."""
    seed(conn, accession="a-1")
    seed(conn, ticker="BBB", accession="b-1")
    write_events(cfg, conn)
    assert prune_events(conn, {event_id_for("a-1")}) == 1
    assert prune_events(conn, {event_id_for("a-1")}) == 0      # idempotent


def test_a_build_that_drops_events_says_so(cfg, conn, caplog):
    """The study getting smaller is a finding, not a detail of a log line."""
    seed(conn, accession="old-1", acceptance_iso="2025-10-01T20:30:00Z")
    seed(conn, ticker="BBB", accession="new-1")
    write_events(cfg, conn)

    later = {**cfg, "study_window": {**cfg["study_window"],
                                     "start": "2026-01-01"}}
    with caplog.at_level("WARNING"):
        write_events(later, conn)
    assert "1 stored events dropped" in caplog.text


def test_pruning_keeps_every_current_event(cfg, conn):
    """The dangerous direction: pruning must never touch a live event."""
    acc = seed(conn)
    add_article(conn, "AAPL", acc - 30 * 60)
    written, _ = write_events(cfg, conn)
    assert written == 1
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


# --------------------------------------------------------------------------
# provenance — which config produced the stored rows
# --------------------------------------------------------------------------

def test_a_build_records_the_tier_and_lookback_it_used(cfg, conn):
    """"0 matched" at tier 1 and "0 matched" from a broken news table look
    identical in the table. The stamp is how a later reader tells them apart."""
    acc = seed(conn)
    add_article(conn, "AAPL", acc - 30 * 60)
    write_events(cfg, conn, max_tier=1)

    prov = build_provenance(conn)
    assert prov["max_tier"] == "1"
    assert prov["lookback_hours"] == str(cfg["news"]["t0_lookback_hours"])
    assert int(prov["built_utc"]) > 0


def test_provenance_is_absent_before_any_build(cfg, conn):
    assert build_provenance(conn) == {"built_utc": None, "max_tier": None,
                                      "lookback_hours": None}


def test_provenance_records_an_unrestricted_tier(cfg, conn):
    seed(conn)
    write_events(cfg, conn, max_tier=None)
    assert build_provenance(conn)["max_tier"] == "any"


def test_a_rebuild_overwrites_the_stamp(cfg, conn):
    """Provenance describes the rows as they stand, not the first run."""
    seed(conn)
    write_events(cfg, conn, max_tier=1)
    write_events(cfg, conn, max_tier=2)
    assert build_provenance(conn)["max_tier"] == "2"


def test_a_build_with_no_news_match_warns_loudly(cfg, conn, caplog):
    """Rule 6. Storing the uncorrected baseline is a decision, not a detail —
    the exact outcome of the old `--build --tier 1`, which said nothing."""
    acc = seed(conn)
    add_article(conn, "AAPL", acc - 30 * 60, tier=2)
    with caplog.at_level("WARNING"):
        write_events(cfg, conn, max_tier=1)
    assert "NOT ONE event took its t0 from news" in caplog.text
    assert only_event(conn)["t0_source"] == "filing"


def test_a_build_that_matches_news_does_not_warn(cfg, conn, caplog):
    acc = seed(conn)
    add_article(conn, "AAPL", acc - 30 * 60)
    with caplog.at_level("WARNING"):
        write_events(cfg, conn)
    assert "NOT ONE event" not in caplog.text
