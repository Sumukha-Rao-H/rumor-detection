"""The two operator scripts — `scripts/browse.py` and `scripts/build_slim_db.py`.

Neither had a test. `build_slim_db.py` in particular seeds the scheduled
monitor in CI, so a silent failure there produces a plausible-looking database
and every alert computed from the wrong columns.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from src import db
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

REPO_ROOT = Path(__file__).resolve().parents[1]
HOUR = 3600
BASE = date_str_to_ts("2026-08-10")


def _load(name: str):
    """Import a script by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        f"_script_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_db(tmp_path):
    """A miniature stand-in for the real 1 GB database."""
    path = tmp_path / "source.db"
    conn = db.get_conn(path)
    db.upsert_companies(conn, [
        {"cik": "C1", "ticker": "AAA", "in_universe": 1},
        {"cik": "C2", "ticker": "BBB", "in_universe": 0},
    ])
    db.upsert_bars(conn, [("AAA", BASE + i * HOUR, 1.0, 2.0, 0.5, 1.5, 100, "60m")
                          for i in range(48)]
                   + [("BBB", BASE + i * HOUR, 1.0, 2.0, 0.5, 1.5, 100, "60m")
                      for i in range(48)])
    db.upsert_filings(conn, [{"accession_no": "0001", "cik": "C1",
                              "ticker": "AAA", "form": "8-K", "items": "8.01",
                              "acceptance_utc": BASE, "filing_date_utc": BASE}])
    db.upsert_events(conn, [{"event_id": "E1", "accession_no": "0001",
                             "ticker": "AAA", "items": "8.01",
                             "t0_filing_utc": BASE, "t0_utc": BASE,
                             "t0_source": "filing", "is_scheduled": 0,
                             "usable": 1, "exclude_reason": None}])
    conn.commit()
    conn.close()
    return path


def test_the_bootstrap_copy_names_its_columns_rather_than_selecting_star(
        source_db, tmp_path):
    """`SELECT *` across two databases matches by POSITION. The destination is
    built fresh from SCHEMA while the source has had columns appended by
    ALTER TABLE, so the two orders agree only by accident of migration
    history. If one ever diverged, a positional copy would succeed silently
    with values in the wrong columns — no error, and every alert wrong."""
    script = _load("build_slim_db")
    out = db.get_conn(tmp_path / "dest.db")
    sql = script._copy_sql(out, "companies")
    assert "SELECT *" not in sql
    for column in ("cik", "ticker", "in_universe"):
        assert f'"{column}"' in sql
    out.close()


def test_the_bootstrap_carries_the_universe_bars_filings_and_events(
        source_db, tmp_path):
    script = _load("build_slim_db")
    cfg = load_config()
    cfg["paths"]["db"] = str(source_db)
    cfg["market"]["interval"] = "60m"
    cfg["market"]["benchmark"] = "BBB"

    stats = script.build(cfg, str(tmp_path / "slim.db"), days=365,
                         filing_days=400)

    assert stats["companies"] == 2
    assert stats["filings"] == 1
    assert stats["events"] == 1
    assert stats["bars"] == 96          # the in-universe ticker and the benchmark

    conn = db.get_conn(tmp_path / "slim.db")
    row = conn.execute("SELECT * FROM companies WHERE ticker = 'AAA'").fetchone()
    assert row["cik"] == "C1" and row["in_universe"] == 1   # not shifted
    conn.close()


def test_the_bootstrap_leaves_out_a_ticker_that_is_neither_universe_nor_benchmark(
        source_db, tmp_path):
    """The point of the slim database is that it is slim."""
    script = _load("build_slim_db")
    cfg = load_config()
    cfg["paths"]["db"] = str(source_db)
    cfg["market"]["interval"] = "60m"
    cfg["market"]["benchmark"] = "SPY"        # BBB is now neither

    script.build(cfg, str(tmp_path / "slim.db"), days=365, filing_days=400)
    conn = db.get_conn(tmp_path / "slim.db")
    assert {r[0] for r in conn.execute("SELECT DISTINCT ticker FROM bars")} \
        == {"AAA"}
    conn.close()


def test_browse_resolves_its_files_against_the_repo_not_the_working_directory():
    """Run from anywhere but the repo root, `--list` used to report every
    generated file as "(not generated yet)" — the inventory tool answering the
    one question it exists for, confidently and wrongly, while the SQLite half
    of the same listing kept working."""
    script = _load("browse")
    for name, (path, _kind) in script.FILES.items():
        assert Path(path).is_absolute(), f"{name} is still a relative path"
        assert str(path).startswith(str(REPO_ROOT)), name


def test_browse_can_reach_the_final_test_numbers():
    """Phase 10's table is the most important artefact in the project, and the
    tool that claims to browse every generated dataset could not open it."""
    script = _load("browse")
    assert "phase10-final" in script.FILES
    assert "phase10-final" in script.WHAT_IT_IS
    for name in script.FILES:
        assert name in script.WHAT_IT_IS, f"{name} has no description"
