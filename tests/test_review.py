"""Human review round-trip tests (plan §6.4 stage 3). No network."""

import csv
import json
from pathlib import Path

import pytest

from src import db
from src.pipeline.review import (
    export_sheet, import_sheet, parse_decision, sheet_rows, status,
)

DAY = 86400
T0 = 1_700_000_000 // DAY * DAY + 12 * 3600


def _cfg(tmp_path):
    return {
        "news": {"event_pre_hours": 24,
                 "whitelist": ["reuters.com", "sec.gov"],
                 "secondary_sources": ["benzinga.com"]},
        "event": {"label_horizon_hours": 72},
        "labeling": {"prompt_version": "label-v3", "max_headlines": 40,
                     "headline_chars": 200},
        "review": {"evidence_lines": 6},
        "paths": {"review": str(tmp_path)},
    }


def _event(conn, event_id="A-1", ticker="AAA", t0=T0, claim="AAA is being acquired"):
    db.upsert_events(conn, [{
        "event_id": event_id, "ticker": ticker, "t0_utc": t0,
        "claim_summary": claim, "claim_type": "merger",
        "post_ids": json.dumps(["p"]), "n_posts": 1,
        "subreddits": json.dumps(["stocks"]),
    }])
    conn.execute("UPDATE events SET n_rumor_posts = 1, claim_summary = ?"
                 " WHERE event_id = ?", (claim, event_id))
    conn.commit()


def _proposal(conn, event_id="A-1", verdict="UNVERIFIED", rule="moved",
              confidence=0.9, t_official=None):
    db.upsert_label_proposals(conn, [{
        "event_id": event_id, "verdict": verdict, "llm_verdict": verdict,
        "rule": rule, "confidence": confidence, "t_official_utc": t_official,
        "prompt_version": "label-v3",
    }])


# --- the sheet --------------------------------------------------------------

def test_the_sheet_carries_the_claim_and_its_evidence(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn)
    _proposal(conn)
    db.upsert_news(conn, [("u1", "AAA", "AAA agrees to be acquired",
                           "reuters.com", T0 + 3600, "gdelt")])
    (row,) = sheet_rows(conn, _cfg(tmp_path), "moved")
    assert row["claim_summary"] == "AAA is being acquired"
    assert "reuters.com" in row["evidence"]
    assert row["label"] == ""          # the human fills this


def test_credible_evidence_is_marked_for_the_reviewer(tmp_path):
    """'Reuters confirms' vs 'Benzinga repeats' is the judgement being asked."""
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn)
    _proposal(conn)
    db.upsert_news(conn, [
        ("u1", "AAA", "wire copy", "reuters.com", T0 + 3600, "gdelt"),
        ("u2", "AAA", "aggregated", "benzinga.com", T0 + 7200, "finnhub"),
    ])
    (row,) = sheet_rows(conn, _cfg(tmp_path), "moved")
    assert "*" in row["evidence"].split("reuters.com")[0]


def test_buckets_select_only_their_own_rule(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA")
    _event(conn, "B-1", "BBB")
    _proposal(conn, "A-1", rule="moved")
    _proposal(conn, "B-1", rule="quiet", verdict="FALSE")
    assert [r["event_id"] for r in sheet_rows(conn, _cfg(tmp_path), "moved")] == ["A-1"]
    assert [r["event_id"] for r in sheet_rows(conn, _cfg(tmp_path), "quiet")] == ["B-1"]
    assert len(sheet_rows(conn, _cfg(tmp_path), "all")) == 2


def test_an_unknown_bucket_is_refused(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    with pytest.raises(SystemExit):
        sheet_rows(conn, _cfg(tmp_path), "nonsense")


def test_already_reviewed_events_drop_out_of_the_sheet(tmp_path):
    """A resumed review must not re-ask what a human already answered."""
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn)
    _proposal(conn)
    conn.execute("UPDATE events SET human_reviewed = 1")
    conn.commit()
    assert sheet_rows(conn, _cfg(tmp_path), "moved") == []


# --- parsing a reviewer's answer -------------------------------------------

def test_the_accepted_verdicts_map_to_labels():
    assert parse_decision({"event_id": "A", "label": "TRUE"})[1] == 1
    assert parse_decision({"event_id": "A", "label": "FALSE"})[1] == 0
    assert parse_decision({"event_id": "A", "label": "SKIP"})[1] is None
    assert parse_decision({"event_id": "A", "label": ""})[1] is None


def test_verdicts_are_case_and_space_insensitive():
    assert parse_decision({"event_id": "A", "label": " true "})[1] == 1


def test_a_typo_is_refused_rather_than_guessed():
    """Silently reading 'ture' as FALSE is the noise review exists to stop."""
    with pytest.raises(ValueError, match="unrecognised"):
        parse_decision({"event_id": "A", "label": "ture"})


def test_a_row_without_an_event_id_is_refused():
    with pytest.raises(ValueError):
        parse_decision({"label": "TRUE"})


def test_a_reviewer_supplied_date_is_read_as_utc():
    from src.utils.timeutils import date_str_to_ts
    _, _, ts = parse_decision({"event_id": "A", "label": "TRUE",
                               "t_official": "2025-03-04"})
    assert ts == date_str_to_ts("2025-03-04")


def test_a_bad_timestamp_is_refused():
    with pytest.raises(ValueError, match="t_official"):
        parse_decision({"event_id": "A", "label": "TRUE", "t_official": "soon"})


# --- the round trip ---------------------------------------------------------

def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "label",
                                                    "t_official"])
        writer.writeheader()
        writer.writerows(rows)


def test_a_reviewed_sheet_writes_labels_and_marks_them_human(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn)
    _proposal(conn)
    path = tmp_path / "sheet.csv"
    _write(path, [{"event_id": "A-1", "label": "TRUE", "t_official": ""}])

    stats = import_sheet(conn, _cfg(tmp_path), path)
    assert stats["applied"] == 1
    row = conn.execute("SELECT * FROM events WHERE event_id='A-1'").fetchone()
    assert row["label"] == 1 and row["human_reviewed"] == 1
    assert row["label_source"] == "human"


def test_a_blank_label_leaves_the_event_untouched(tmp_path):
    """Reviewing 200 rows over two sittings must not label the unfinished ones."""
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn)
    _proposal(conn)
    path = tmp_path / "sheet.csv"
    _write(path, [{"event_id": "A-1", "label": "", "t_official": ""}])

    stats = import_sheet(conn, _cfg(tmp_path), path)
    assert stats["applied"] == 0 and stats["left_blank"] == 1
    row = conn.execute("SELECT * FROM events WHERE event_id='A-1'").fetchone()
    assert row["label"] is None and row["human_reviewed"] == 0


def test_an_unspecified_t_official_falls_back_to_the_proposal(tmp_path):
    """The reviewer is correcting the verdict, not the clock."""
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn)
    _proposal(conn, t_official=T0 + 5 * 3600)
    path = tmp_path / "sheet.csv"
    _write(path, [{"event_id": "A-1", "label": "TRUE", "t_official": ""}])

    import_sheet(conn, _cfg(tmp_path), path)
    assert conn.execute("SELECT t_official_utc FROM events").fetchone()[0] == \
        T0 + 5 * 3600


def test_a_reviewer_can_override_t_official(tmp_path):
    from src.utils.timeutils import date_str_to_ts

    conn = db.get_conn(tmp_path / "t.db")
    _event(conn)
    _proposal(conn, t_official=T0)
    path = tmp_path / "sheet.csv"
    _write(path, [{"event_id": "A-1", "label": "TRUE",
                   "t_official": "2025-03-04"}])

    import_sheet(conn, _cfg(tmp_path), path)
    assert conn.execute("SELECT t_official_utc FROM events").fetchone()[0] == \
        date_str_to_ts("2025-03-04")


def test_an_unknown_event_id_stops_the_import(tmp_path):
    """A mangled sheet must not half-apply."""
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn)
    _proposal(conn)
    path = tmp_path / "sheet.csv"
    _write(path, [{"event_id": "GHOST", "label": "TRUE", "t_official": ""}])
    with pytest.raises(SystemExit):
        import_sheet(conn, _cfg(tmp_path), path)


def test_export_then_import_round_trips(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA")
    _event(conn, "B-1", "BBB", t0=T0 + DAY)
    _proposal(conn, "A-1")
    _proposal(conn, "B-1")
    path = Path(tmp_path) / "moved.csv"
    assert export_sheet(conn, _cfg(tmp_path), path, "moved") == 2

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0]["label"] = "TRUE"
    rows[1]["label"] = "FALSE"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    import_sheet(conn, _cfg(tmp_path), path)
    labels = dict(conn.execute("SELECT event_id, label FROM events"))
    assert labels == {"A-1": 1, "B-1": 0}


def test_status_counts_progress_per_bucket(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "A-1", "AAA")
    _event(conn, "B-1", "BBB")
    _proposal(conn, "A-1", rule="moved")
    _proposal(conn, "B-1", rule="moved")
    conn.execute("UPDATE events SET human_reviewed = 1 WHERE event_id='A-1'")
    conn.commit()
    assert status(conn, _cfg(tmp_path))["moved"] == (1, 2)
