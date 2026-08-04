"""LLM client tests: caching, failover, JSON parsing. No network is touched."""

import json

import pytest

from src.utils import llm as llm_mod
from src.utils.llm import LLMClient, LLMError, parse_json


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Backoff sleeps for real seconds; tests must not."""
    class _Instant:
        def __init__(self, *a, **k):
            self.failures = 0

        def sleep(self, reason=""):
            self.failures += 1

        def reset(self):
            self.failures = 0

    monkeypatch.setattr(llm_mod, "Backoff", _Instant)


def _cfg(tmp_path):
    return {
        "paths": {"llm_cache": str(tmp_path / "cache")},
        "llm": {
            "provider": "gemini", "fallback": "groq", "min_interval_s": 0,
            "temperature": 0, "max_output_tokens": 100, "timeout_s": 5,
            "quota_failures_before_fallback": 2, "prompt_version": "v1",
            "gemini": {"api_base": "x", "model": "gem-1",
                       "key_env": "GEMINI_API_KEY", "min_interval_s": 0},
            "groq": {"api_base": "y", "model": "groq-1",
                     "key_env": "GROQ_API_KEY", "min_interval_s": 0},
        },
    }


@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Sure! Here it is: {"a": 1} hope that helps', {"a": 1}),
])
def test_parse_json_tolerates_wrapping(raw, expected):
    assert parse_json(raw) == expected


@pytest.mark.parametrize("raw", ["", "no json here", "{broken"])
def test_parse_json_raises_rather_than_guessing(raw):
    with pytest.raises(LLMError):
        parse_json(raw)


def test_cache_prevents_a_second_call(tmp_path, monkeypatch):
    client = LLMClient(_cfg(tmp_path))
    calls = []
    monkeypatch.setattr(client, "_request",
                        lambda provider, prompt: calls.append(prompt) or '{"n": 1}')

    first = client.complete_json("prompt", cache_key="k1")
    second = client.complete_json("prompt", cache_key="k1")
    assert first.data == second.data == {"n": 1}
    assert len(calls) == 1
    assert first.cached is False and second.cached is True
    assert client.calls == 1 and client.cache_hits == 1


def test_cache_is_scoped_by_prompt_version(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    client = LLMClient(cfg)
    monkeypatch.setattr(client, "_request", lambda p, prompt: '{"n": 1}')
    client.complete_json("prompt", cache_key="k1")

    cfg["llm"]["prompt_version"] = "v2"
    fresh = LLMClient(cfg)
    monkeypatch.setattr(fresh, "_request", lambda p, prompt: '{"n": 2}')
    assert fresh.complete_json("prompt", cache_key="k1").data == {"n": 2}


def test_falls_back_to_second_provider_on_quota(tmp_path, monkeypatch):
    client = LLMClient(_cfg(tmp_path))
    seen = []

    def flaky(provider, prompt):
        seen.append(provider)
        if provider == "gemini":
            raise llm_mod._Retryable("HTTP 429 quota")
        return '{"ok": true}'

    monkeypatch.setattr(client, "_request", flaky)
    reply = client.complete_json("prompt", cache_key="k")
    assert reply.provider == "groq" and reply.model == "groq-1"
    assert seen == ["gemini", "gemini", "groq"]   # 2 attempts, then switch


def test_raises_when_every_provider_is_exhausted(tmp_path, monkeypatch):
    client = LLMClient(_cfg(tmp_path))
    monkeypatch.setattr(client, "_request", lambda p, prompt: (_ for _ in ()).throw(
        llm_mod._Retryable("HTTP 429")))
    with pytest.raises(LLMError, match="all providers failed"):
        client.complete_json("prompt", cache_key="k")


def test_cached_reply_survives_a_new_client(tmp_path, monkeypatch):
    """An interrupted multi-day job must resume, not restart."""
    cfg = _cfg(tmp_path)
    first = LLMClient(cfg)
    monkeypatch.setattr(first, "_request", lambda p, prompt: '{"n": 7}')
    first.complete_json("prompt", cache_key="k")

    second = LLMClient(cfg)
    monkeypatch.setattr(second, "_request", lambda p, prompt: pytest.fail("recalled"))
    assert second.complete_json("prompt", cache_key="k").data == {"n": 7}


def test_cache_file_records_provenance(tmp_path, monkeypatch):
    client = LLMClient(_cfg(tmp_path))
    monkeypatch.setattr(client, "_request", lambda p, prompt: '{"n": 1}')
    client.complete_json("prompt", cache_key="k")
    blob = json.loads(next((tmp_path / "cache").glob("*.json")).read_text())
    assert blob["provider"] == "gemini" and blob["prompt_version"] == "v1"
