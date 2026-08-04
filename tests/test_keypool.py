"""API key rotation tests (src/utils/keypool.py). No network is touched."""

import json
import os

import pytest

from src.utils.keypool import (
    DAY,
    MINUTE,
    UNKNOWN,
    KeyPool,
    NoKeysAvailable,
    classify_quota,
    short_reason,
    discover_keys,
)
from src.utils.timeutils import next_midnight_ts, utc_now_ts


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("GEMINI_API_KEY"):
            monkeypatch.delenv(name, raising=False)


def _cfg(tmp_path, min_interval_s=0):
    return {
        "paths": {"llm_state": str(tmp_path / "llm_state.json")},
        "llm": {
            "min_interval_s": min_interval_s, "key_cooldown_s": 900,
            "gemini": {"key_env": "GEMINI_API_KEY",
                       "min_interval_s": min_interval_s,
                       "quota_reset_tz": "America/Los_Angeles"},
        },
    }


# ------------------------------------------------------------------ discovery

def test_discovery_accepts_all_three_spellings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "solo")
    monkeypatch.setenv("GEMINI_API_KEYS", "list_a, list_b")
    monkeypatch.setenv("GEMINI_API_KEY_1", "num_1")
    labels = dict((v, l) for l, v in discover_keys("GEMINI_API_KEY"))
    assert set(labels) == {"solo", "list_a", "list_b", "num_1"}
    assert labels["list_a"] == "GEMINI_API_KEYS#1"


def test_numbered_keys_are_ordered_numerically_not_lexically(monkeypatch):
    for i in (1, 2, 10, 12):
        monkeypatch.setenv(f"GEMINI_API_KEY_{i}", f"k{i}")
    assert [v for _, v in discover_keys("GEMINI_API_KEY")] == \
        ["k1", "k2", "k10", "k12"]


def test_duplicate_keys_collapse_to_one_quota(monkeypatch):
    """Two teammates pasting the same key is one budget, not two."""
    monkeypatch.setenv("GEMINI_API_KEY_1", "same")
    monkeypatch.setenv("GEMINI_API_KEY_2", "same")
    monkeypatch.setenv("GEMINI_API_KEY_3", "other")
    assert [v for _, v in discover_keys("GEMINI_API_KEY")] == ["same", "other"]


def test_blank_and_missing_keys_are_ignored(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    monkeypatch.setenv("GEMINI_API_KEY_1", "")
    monkeypatch.setenv("GEMINI_API_KEY_2", "real")
    assert [v for _, v in discover_keys("GEMINI_API_KEY")] == ["real"]


def test_pool_refuses_to_start_with_no_keys(tmp_path):
    with pytest.raises(RuntimeError, match="No API keys"):
        KeyPool("gemini", _cfg(tmp_path))


# ------------------------------------------------------------ classification

@pytest.mark.parametrize("body,expected", [
    # Gemini's daily-quota body names the metric in error.details
    ('{"error":{"details":[{"quotaId":'
     '"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}', DAY),
    ('{"error":{"details":[{"quotaId":"GenerateRequestsPerMinute"}]}}', MINUTE),
    ("Rate limit reached ... on requests per day (RPD)", DAY),
    ("Rate limit reached ... on tokens per minute (TPM)", MINUTE),
    ("You exceeded your current quota", UNKNOWN),
])
def test_classify_quota(body, expected):
    assert classify_quota(body) == expected


def test_a_body_naming_both_limits_counts_as_daily(monkeypatch):
    """Daily wins: rotating a spent key is right, resting a live one is not."""
    assert classify_quota("requests per minute ... requests per day") == DAY


# --------------------------------------------------------------- rotation

def test_acquire_spreads_load_across_keys(tmp_path, monkeypatch):
    """Least-recently-used, so one teammate's key is not drained first."""
    for i in (1, 2, 3):
        monkeypatch.setenv(f"GEMINI_API_KEY_{i}", f"k{i}")
    pool = KeyPool("gemini", _cfg(tmp_path, min_interval_s=0.01))
    picked = [pool.acquire().label for _ in range(6)]
    assert picked == ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"] * 2


def test_cooled_key_is_skipped_then_returns(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_1", "k1")
    monkeypatch.setenv("GEMINI_API_KEY_2", "k2")
    pool = KeyPool("gemini", _cfg(tmp_path))
    first = pool.acquire()
    pool.cool(first, DAY, "out of requests")
    assert [k.label for k in pool.available()] == ["GEMINI_API_KEY_2"]
    assert pool.acquire().label == "GEMINI_API_KEY_2"

    first.cooling_until = 0          # quota reset
    assert len(pool.available()) == 2


def test_daily_cooldown_runs_to_the_providers_midnight(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    pool = KeyPool("gemini", _cfg(tmp_path))
    key = pool.keys[0]
    pool.cool(key, DAY, "spent")
    assert key.cooling_until == next_midnight_ts("America/Los_Angeles")


def test_soft_cooldown_is_short(tmp_path, monkeypatch):
    """An unexplained failure rests a key; it does not retire it for the day."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    pool = KeyPool("gemini", _cfg(tmp_path))
    pool.cool(pool.keys[0], "soft", "timeout")
    assert pool.keys[0].cooling_until - utc_now_ts() == pytest.approx(900, abs=5)


def test_disabled_key_never_comes_back(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_1", "revoked")
    monkeypatch.setenv("GEMINI_API_KEY_2", "good")
    pool = KeyPool("gemini", _cfg(tmp_path))
    pool.cool(pool.keys[0], "disabled", "HTTP 400 API key not valid")
    assert "disabled" in pool.keys[0].status(utc_now_ts())
    assert [k.label for k in pool.available()] == ["GEMINI_API_KEY_2"]


def test_empty_pool_raises_rather_than_blocking(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    pool = KeyPool("gemini", _cfg(tmp_path))
    pool.cool(pool.keys[0], DAY, "spent")
    with pytest.raises(NoKeysAvailable, match="1 gemini keys unavailable"):
        pool.acquire()


# ------------------------------------------------------------------- state

def test_cooldowns_survive_a_restart(tmp_path, monkeypatch):
    """The 30-min resume loop must not re-discover yesterday's dead keys."""
    monkeypatch.setenv("GEMINI_API_KEY_1", "k1")
    monkeypatch.setenv("GEMINI_API_KEY_2", "k2")
    cfg = _cfg(tmp_path)
    pool = KeyPool("gemini", cfg)
    pool.cool(pool.keys[0], DAY, "spent")

    fresh = KeyPool("gemini", cfg)
    assert [k.label for k in fresh.available()] == ["GEMINI_API_KEY_2"]


def test_expired_cooldowns_lapse_on_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    cfg = _cfg(tmp_path)
    pool = KeyPool("gemini", cfg)
    pool.cool(pool.keys[0], "soft", "throttled")
    state = json.loads((tmp_path / "llm_state.json").read_text())
    state["gemini"][pool.keys[0].fingerprint]["cooling_until"] = utc_now_ts() - 1
    (tmp_path / "llm_state.json").write_text(json.dumps(state))

    assert len(KeyPool("gemini", cfg).available()) == 1


def test_state_file_never_contains_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    cfg = _cfg(tmp_path)
    pool = KeyPool("gemini", cfg)
    pool.cool(pool.keys[0], DAY, "spent")
    assert "super-secret-value" not in (tmp_path / "llm_state.json").read_text()
    assert "super-secret-value" not in pool.describe()


def test_reset_puts_every_key_back(tmp_path, monkeypatch):
    """A re-issued key must be usable without hand-editing the state file."""
    monkeypatch.setenv("GEMINI_API_KEY_1", "k1")
    monkeypatch.setenv("GEMINI_API_KEY_2", "k2")
    cfg = _cfg(tmp_path)
    pool = KeyPool("gemini", cfg)
    pool.cool(pool.keys[0], "disabled", "HTTP 403 denied")
    pool.cool(pool.keys[1], DAY, "spent")
    pool.reset()
    assert len(pool.available()) == 2
    assert len(KeyPool("gemini", cfg).available()) == 2


def test_reason_keeps_the_sentence_and_drops_the_json(tmp_path, monkeypatch):
    """Provider errors are multi-line JSON; `keypool` output must stay readable."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    pool = KeyPool("gemini", _cfg(tmp_path))
    pool.cool(pool.keys[0], DAY, 'HTTP 429 {\n  "error": {\n    "code": 429,\n'
                                 '    "message": "You exceeded your quota"\n  }\n}')
    assert pool.keys[0].reason == "HTTP 429: You exceeded your quota"


def test_short_reason_falls_back_to_flattening():
    assert short_reason("connection\n  reset") == "connection reset"


def test_short_reason_handles_a_message_cut_off_mid_string():
    """State written by the earlier truncating version ends without its quote."""
    assert short_reason('HTTP 429 { "error": { "code": 429, "message": '
                        '"You exceeded your current quota, please check your pl'
                        ) == "HTTP 429: You exceeded your current quota, please check your pl"


def test_short_reason_is_idempotent():
    """It runs on write *and* on read; twice must not mean 'HTTP 429: HTTP 429'."""
    once = short_reason('HTTP 403 {"error": {"message": "Denied access."}}')
    assert once == "HTTP 403: Denied access."
    assert short_reason(once) == once


def test_describe_summarises_instead_of_listing_every_key(tmp_path, monkeypatch):
    """This string lands in every error; twelve key states made logs unreadable."""
    for i in range(1, 13):
        monkeypatch.setenv(f"GEMINI_API_KEY_{i}", f"k{i}")
    pool = KeyPool("gemini", _cfg(tmp_path))
    for key in pool.keys[:9]:
        pool.cool(key, DAY, "HTTP 429 spent")
    for key in pool.keys[9:11]:
        pool.cool(key, "disabled", "HTTP 403 denied")

    summary = pool.describe()
    assert summary.startswith("1/12 ready")
    assert "9 cooling until" in summary
    assert "2 disabled (GEMINI_API_KEY_10, GEMINI_API_KEY_11)" in summary
    assert len(summary) < 200 and summary.count("GEMINI_API_KEY") == 2


def test_state_is_per_provider(tmp_path, monkeypatch):
    """Cooling a gemini key must not disturb another provider's entries."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    cfg = _cfg(tmp_path)
    (tmp_path / "llm_state.json").write_text(json.dumps({"groq": {"abc": {}}}))
    pool = KeyPool("gemini", cfg)
    pool.cool(pool.keys[0], DAY, "spent")
    blob = json.loads((tmp_path / "llm_state.json").read_text())
    assert set(blob) == {"groq", "gemini"}
