"""LLM client tests: caching, failover, JSON parsing. No network is touched."""

import json
import os

import pytest

from src.utils import llm as llm_mod
from src.utils.llm import LLMClient, LLMError, LLMRefused, parse_json


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


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    """One key per provider unless a test asks for a pool."""
    for name in list(os.environ):
        if name.startswith(("GEMINI_API_KEY", "GROQ_API_KEY")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")


def _cfg(tmp_path):
    return {
        "paths": {"llm_cache": str(tmp_path / "cache"),
                  "llm_state": str(tmp_path / "llm_state.json")},
        "llm": {
            "provider": "gemini", "fallback": "groq", "min_interval_s": 0,
            "temperature": 0, "max_output_tokens": 100, "timeout_s": 5,
            "quota_failures_before_fallback": 2, "prompt_version": "v1",
            "key_cooldown_s": 900,
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


def test_parse_json_unwraps_a_one_element_list():
    """Observed live: the model answered `[{...}]` and crashed the run."""
    assert parse_json('[{"is_rumor": false}]') == {"is_rumor": False}


@pytest.mark.parametrize("raw", ['[1, 2]', '"just a string"', '[{"a": 1}, {"b": 2}]'])
def test_unusable_shapes_are_refused_not_fatal(raw):
    """A reply we cannot read is this prompt's problem: skip it, don't abort."""
    with pytest.raises(LLMRefused):
        parse_json(raw)


def test_a_malformed_cache_entry_is_re_asked_not_replayed(tmp_path, monkeypatch):
    """A bad shape on disk would otherwise fail identically on every re-run."""
    cfg = _cfg(tmp_path)
    client = LLMClient(cfg)
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    path = client._cache_path("k", "gem-1")
    path.write_text(json.dumps({"cache_key": "k", "provider": "gemini",
                                "model": "gem-1", "data": [{"is_rumor": False}]}))

    monkeypatch.setattr(client, "_request", lambda p, prompt, key: '{"n": 1}')
    assert client.complete_json("prompt", cache_key="k").data == {"n": 1}
    assert json.loads(path.read_text())["data"] == {"n": 1}   # repaired on disk


def test_cache_prevents_a_second_call(tmp_path, monkeypatch):
    client = LLMClient(_cfg(tmp_path))
    calls = []
    monkeypatch.setattr(client, "_request",
                        lambda provider, prompt, key: calls.append(prompt) or '{"n": 1}')

    first = client.complete_json("prompt", cache_key="k1")
    second = client.complete_json("prompt", cache_key="k1")
    assert first.data == second.data == {"n": 1}
    assert len(calls) == 1
    assert first.cached is False and second.cached is True
    assert client.calls == 1 and client.cache_hits == 1


def test_cache_is_scoped_by_prompt_version(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    client = LLMClient(cfg)
    monkeypatch.setattr(client, "_request", lambda p, prompt, key: '{"n": 1}')
    client.complete_json("prompt", cache_key="k1")

    cfg["llm"]["prompt_version"] = "v2"
    fresh = LLMClient(cfg)
    monkeypatch.setattr(fresh, "_request", lambda p, prompt, key: '{"n": 2}')
    assert fresh.complete_json("prompt", cache_key="k1").data == {"n": 2}


def test_falls_back_to_second_provider_on_quota(tmp_path, monkeypatch):
    client = LLMClient(_cfg(tmp_path))
    seen = []

    def flaky(provider, prompt, key):
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
    monkeypatch.setattr(client, "_request", lambda p, prompt, key: (_ for _ in ()).throw(
        llm_mod._Retryable("HTTP 429")))
    with pytest.raises(LLMError, match="all providers failed"):
        client.complete_json("prompt", cache_key="k")


def test_cached_reply_survives_a_new_client(tmp_path, monkeypatch):
    """An interrupted multi-day job must resume, not restart."""
    cfg = _cfg(tmp_path)
    first = LLMClient(cfg)
    monkeypatch.setattr(first, "_request", lambda p, prompt, key: '{"n": 7}')
    first.complete_json("prompt", cache_key="k")

    second = LLMClient(cfg)
    monkeypatch.setattr(second, "_request", lambda p, prompt, key: pytest.fail("recalled"))
    assert second.complete_json("prompt", cache_key="k").data == {"n": 7}


def test_no_fallback_configured_means_one_provider(tmp_path, monkeypatch):
    """Triage runs single-model on purpose: no silent second opinion."""
    cfg = _cfg(tmp_path)
    cfg["llm"]["fallback"] = None
    client = LLMClient(cfg)
    assert client.providers == ["gemini"]
    monkeypatch.setattr(client, "_request", lambda p, prompt, key: (_ for _ in ()).throw(
        llm_mod._Retryable("HTTP 429")))
    with pytest.raises(LLMError):
        client.complete_json("prompt", cache_key="k")


def test_cache_is_scoped_by_model(tmp_path, monkeypatch):
    """A different model must re-ask, not inherit another model's answer."""
    cfg = _cfg(tmp_path)
    client = LLMClient(cfg)
    monkeypatch.setattr(client, "_request", lambda p, prompt, key: '{"n": 1}')
    client.complete_json("prompt", cache_key="k")

    cfg["llm"]["gemini"]["model"] = "gem-2"
    other = LLMClient(cfg)
    monkeypatch.setattr(other, "_request", lambda p, prompt, key: '{"n": 2}')
    assert other.complete_json("prompt", cache_key="k").data == {"n": 2}


def test_daily_quota_rotates_to_the_next_key_not_the_next_model(
        tmp_path, monkeypatch):
    """12 free keys are 12 daily budgets; the model must not change."""
    monkeypatch.setenv("GEMINI_API_KEY_1", "spent")
    monkeypatch.setenv("GEMINI_API_KEY_2", "fresh")
    monkeypatch.delenv("GEMINI_API_KEY")
    client = LLMClient(_cfg(tmp_path))
    tried = []

    def one_key_is_spent(provider, prompt, key):
        tried.append(key)
        if key == "spent":
            raise llm_mod._Retryable("HTTP 429 ... PerDay ...", scope="day")
        return '{"ok": true}'

    monkeypatch.setattr(client, "_request", one_key_is_spent)
    reply = client.complete_json("prompt", cache_key="k")
    assert reply.provider == "gemini" and reply.model == "gem-1"
    assert tried == ["spent", "fresh"]     # one attempt each, no retry storm
    assert client.key_rotations == 1


def test_per_minute_limit_retries_the_same_key(tmp_path, monkeypatch):
    """Rotating on throttling would burn every key in under a minute."""
    monkeypatch.setenv("GEMINI_API_KEY_1", "a")
    monkeypatch.setenv("GEMINI_API_KEY_2", "b")
    monkeypatch.delenv("GEMINI_API_KEY")
    client = LLMClient(_cfg(tmp_path))
    tried = []

    def throttled_once(provider, prompt, key):
        tried.append(key)
        if len(tried) == 1:
            raise llm_mod._Retryable("HTTP 429 requests per minute",
                                     scope="minute")
        return '{"ok": true}'

    monkeypatch.setattr(client, "_request", throttled_once)
    client.complete_json("prompt", cache_key="k")
    assert tried == ["a", "a"] and client.key_rotations == 0


def test_a_dead_key_is_retired_without_ending_the_run(tmp_path, monkeypatch):
    """Borrowed keys include revoked ones; one must not stop a batch."""
    monkeypatch.setenv("GEMINI_API_KEY_1", "revoked")
    monkeypatch.setenv("GEMINI_API_KEY_2", "good")
    monkeypatch.delenv("GEMINI_API_KEY")
    client = LLMClient(_cfg(tmp_path))
    tried = []

    def one_is_revoked(provider, prompt, key):
        tried.append(key)
        if key == "revoked":
            raise llm_mod._BadKey("HTTP 400 API key not valid")
        return '{"ok": true}'

    monkeypatch.setattr(client, "_request", one_is_revoked)
    assert client.complete_json("prompt", cache_key="k").data == {"ok": True}
    assert tried == ["revoked", "good"]
    # ... and it is not tried again on the next prompt
    client.complete_json("other", cache_key="k2")
    assert tried[2:] == ["good"]


def test_a_refused_prompt_does_not_burn_the_pool(tmp_path, monkeypatch):
    """A safety block is the post's fault, not the key's — never rotate on it."""
    monkeypatch.setenv("GEMINI_API_KEY_1", "a")
    monkeypatch.setenv("GEMINI_API_KEY_2", "b")
    monkeypatch.delenv("GEMINI_API_KEY")
    client = LLMClient(_cfg(tmp_path))
    monkeypatch.setattr(client, "_request", lambda p, prompt, key: (
        _ for _ in ()).throw(LLMRefused("empty candidates")))
    with pytest.raises(LLMRefused):
        client.complete_json("prompt", cache_key="k")
    assert client.key_rotations == 0
    assert len(client._pools["gemini"].available()) == 2


def test_cache_file_records_provenance(tmp_path, monkeypatch):
    client = LLMClient(_cfg(tmp_path))
    monkeypatch.setattr(client, "_request", lambda p, prompt, key: '{"n": 1}')
    client.complete_json("prompt", cache_key="k")
    blob = json.loads(next((tmp_path / "cache").glob("*.json")).read_text())
    assert blob["provider"] == "gemini" and blob["prompt_version"] == "v1"


def test_a_server_outage_does_not_cool_the_key(tmp_path, monkeypatch):
    """503 is the provider's backend; every key reaches it, so rotating cannot
    help and cooling one spends a healthy credential on someone else's fault."""
    import requests

    from src.utils import llm as llm_mod

    cfg = _cfg(tmp_path)
    cfg["llm"]["service_outage_rounds"] = 3
    monkeypatch.setenv("GEMINI_API_KEY_1", "k1")
    monkeypatch.setenv("GEMINI_API_KEY_2", "k2")
    client = llm_mod.LLMClient(cfg)
    monkeypatch.setattr(llm_mod.Backoff, "sleep", lambda self, why=None: None)

    def always_503(self, provider, prompt, key):
        resp = requests.Response()
        resp.status_code = 503
        resp._content = b'{"error": {"message": "The model is overloaded."}}'
        llm_mod.LLMClient._raise_for_status(resp)

    monkeypatch.setattr(llm_mod.LLMClient, "_request", always_503)
    with pytest.raises(llm_mod.LLMError, match="unavailable"):
        client.complete_json("hi", cache_key="c1")

    pool = client._pools["gemini"]
    assert len(pool.available()) == len(pool)   # nothing was cooled
    assert client.key_rotations == 0
