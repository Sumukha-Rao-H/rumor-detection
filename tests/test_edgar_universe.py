"""Universe build tests — parsing and primary-ticker selection. No network.

The one that earns its keep is `test_share_classes_do_not_create_extra_rows`.
`companies` is keyed by CIK and 895 real CIKs carry several tickers, so a naive
loop leaves JPMorgan labelled `VYLD` — a structured note. The filings would
still be right and every price and headline joined to them would be wrong,
with no error anywhere.
"""

import copy

import pytest

from src import db
from src.collectors.edgar import build_universe, company_rows, pick_primary_ticker
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts


PAYLOAD = {
    "fields": ["cik", "name", "ticker", "exchange"],
    "data": [
        [1045810, "NVIDIA CORP", "NVDA", "Nasdaq"],
        [1652044, "Alphabet Inc.", "GOOGL", "Nasdaq"],
        [1652044, "Alphabet Inc.", "GOOG", "Nasdaq"],
        [1652044, "Alphabet Inc.", "GOOGM", "Nasdaq"],
        [1652044, "Alphabet Inc.", "GOOGN", "Nasdaq"],
        [19617, "JPMORGAN CHASE & CO", "JPM", "NYSE"],
        [19617, "JPMORGAN CHASE & CO", "JPM-PC", "NYSE"],
        [19617, "JPMORGAN CHASE & CO", "VYLD", "NYSE"],
        [1067983, "BERKSHIRE HATHAWAY INC", "BRK-B", "NYSE"],
        [1067983, "BERKSHIRE HATHAWAY INC", "BRK-A", "NYSE"],
        [999001, "Some OTC Shell", "SHEL", "OTC"],
        [999002, "Cboe Listed Thing", "CBOX", "CBOE"],
        [999003, "No Exchange Given", "NONE", None],
    ],
}


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture
def conn(tmp_path):
    c = db.get_conn(tmp_path / "test.db")
    yield c
    c.close()


class FakeClient:
    """Stands in for EdgarClient — returns a payload, records force flags."""

    def __init__(self, payload):
        self.payload = payload
        self.forced = []

    def company_tickers_url(self) -> str:
        return "https://www.sec.gov/files/company_tickers_exchange.json"

    def get_json(self, url, force=False):
        self.forced.append(force)
        return self.payload


# -- selection -------------------------------------------------------------

def test_primary_ticker_prefers_the_unhyphenated_listing():
    assert pick_primary_ticker(["JPM", "JPM-PC", "VYLD"]) == "JPM"
    assert pick_primary_ticker(["ORCL-PD", "ORCL"]) == "ORCL"


def test_primary_ticker_falls_back_to_file_order():
    """Genuine dual-class commons are all hyphenated; keep the first, not none."""
    assert pick_primary_ticker(["BRK-B", "BRK-A"]) == "BRK-B"
    assert pick_primary_ticker(["CMS-PB"]) == "CMS-PB"


def test_share_classes_do_not_create_extra_rows(cfg):
    rows = company_rows(cfg, PAYLOAD)
    alphabet = [r for r in rows if r["cik"] == "0001652044"]
    assert len(alphabet) == 1
    assert alphabet[0]["ticker"] == "GOOGL"


def test_the_structured_note_does_not_become_the_company(cfg):
    """The defect this task exists to prevent: last-ticker-wins labels JPM `VYLD`."""
    rows = company_rows(cfg, PAYLOAD)
    jpm = next(r for r in rows if r["cik"] == "0000019617")
    assert jpm["ticker"] == "JPM"


# -- parsing ---------------------------------------------------------------

def test_only_configured_exchanges_are_kept(cfg):
    rows = company_rows(cfg, PAYLOAD)
    exchanges = {r["exchange"] for r in rows}
    assert exchanges <= set(cfg["universe"]["exchanges"])
    assert "OTC" not in exchanges and "CBOE" not in exchanges
    assert all(r["exchange"] is not None for r in rows)


def test_cik_is_zero_padded_to_ten_digits_as_a_string(cfg):
    rows = company_rows(cfg, PAYLOAD)
    nvda = next(r for r in rows if r["ticker"] == "NVDA")
    assert nvda["cik"] == "0001045810", (
        "the submissions URL is CIK##########.json — an int would not join")
    assert all(isinstance(r["cik"], str) and len(r["cik"]) == 10 for r in rows)


def test_one_row_per_cik(cfg):
    rows = company_rows(cfg, PAYLOAD)
    ciks = [r["cik"] for r in rows]
    assert len(ciks) == len(set(ciks))


def test_universe_as_of_is_the_window_start_not_today(cfg):
    rows = company_rows(cfg, PAYLOAD)
    expected = date_str_to_ts(cfg["study_window"]["start"])
    assert {r["universe_as_of"] for r in rows} == {expected}, (
        "a universe dated today has already dropped every company that was "
        "acquired or delisted — exactly the events this study is about")


def test_the_liquidity_flag_is_left_undecided(cfg):
    """P2-02 has no opinion about liquidity; the Phase 3 filter sets it."""
    rows = company_rows(cfg, PAYLOAD)
    assert all(r["in_universe"] is None for r in rows)


# -- the build -------------------------------------------------------------

def test_rerun_adds_no_duplicates(cfg, conn):
    """The Done-when."""
    client = FakeClient(PAYLOAD)
    first = build_universe(cfg, conn, client=client)
    count_1 = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    build_universe(cfg, conn, client=client)
    count_2 = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    assert count_1 == first == count_2


def test_rebuilding_the_universe_keeps_in_universe_flags(cfg, conn):
    """Regression guard: upsert_companies used to clobber the liquidity flag.

    Phase 3 sets `in_universe`; a later universe rebuild passing NULL must not
    silently empty it, or `market.py --universe` would download nothing.
    """
    client = FakeClient(PAYLOAD)
    build_universe(cfg, conn, client=client)
    conn.execute("UPDATE companies SET in_universe = 1 WHERE ticker = 'NVDA'")
    conn.commit()
    build_universe(cfg, conn, client=client)
    assert db.universe_tickers(conn) == ["NVDA"]


def test_empty_payload_raises(cfg, conn):
    empty = {"fields": PAYLOAD["fields"], "data": []}
    with pytest.raises(RuntimeError, match="ZERO companies"):
        build_universe(cfg, conn, client=FakeClient(empty))


def test_everything_filtered_out_also_raises(cfg, conn):
    """A 200 that parses to nothing usable is the same failure as an empty one."""
    otc_only = {"fields": PAYLOAD["fields"],
                "data": [[999001, "Shell", "SHEL", "OTC"]]}
    with pytest.raises(RuntimeError, match="ZERO companies"):
        build_universe(cfg, conn, client=FakeClient(otc_only))
