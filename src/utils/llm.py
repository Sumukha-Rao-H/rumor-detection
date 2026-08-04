"""Provider-agnostic JSON-mode LLM client for the free-tier work (plan §6.2,
§6.4, §10).

Talks to Gemini and Groq over plain REST — neither vendor SDK is a dependency,
so nothing here breaks when an SDK version moves, and both providers go through
the project's existing RateLimiter/Backoff instead of their own retry logic.

Three properties everything downstream relies on:

  cached      every answer is written to data/llm_cache/ keyed by the caller's
              logical id plus the prompt version, so a re-run costs nothing and
              an interrupted multi-day job resumes exactly where it stopped.
  failover    when the primary provider's daily quota runs out (repeated 429s)
              the client switches to the fallback for the rest of the session
              rather than dying halfway through a batch.
  strict      responses are parsed as JSON or raise; a model that returns prose
              is a bug to see, not a row to silently drop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from src.utils.config import require_env
from src.utils.ratelimit import Backoff, RateLimiter

log = logging.getLogger(__name__)

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
QUOTA_STATUSES = {429, 503}


class LLMError(RuntimeError):
    """Raised when no provider could answer, or the answer was not JSON."""


@dataclass
class LLMReply:
    data: dict
    provider: str
    model: str
    cached: bool


class LLMClient:
    def __init__(self, cfg: dict, cache_dir: str | Path | None = None):
        self.cfg = cfg["llm"]
        self.prompt_version = self.cfg["prompt_version"]
        self.cache_dir = Path(cache_dir or cfg["paths"]["llm_cache"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        order = [self.cfg["provider"]]
        if self.cfg.get("fallback") and self.cfg["fallback"] != self.cfg["provider"]:
            order.append(self.cfg["fallback"])
        self.providers = order
        self._active = 0
        self._limiters = {
            name: RateLimiter(self.cfg[name].get("min_interval_s",
                                                 self.cfg["min_interval_s"]))
            for name in order
        }
        self._keys: dict[str, str] = {}
        self.calls = 0
        self.cache_hits = 0

    # ---------------------------------------------------------------- caching

    def _cache_path(self, cache_key: str, model: str) -> Path:
        # The model is part of the key: two models answer the same question
        # differently, and a dataset must not silently mix them.
        digest = hashlib.sha1(
            f"{self.prompt_version}|{model}|{cache_key}".encode()
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, cache_key: str, model: str) -> LLMReply | None:
        path = self._cache_path(cache_key, model)
        if not path.exists():
            return None
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return None
        return LLMReply(blob["data"], blob.get("provider", "?"),
                        blob.get("model", "?"), cached=True)

    def _write_cache(self, cache_key: str, reply: LLMReply) -> None:
        self._cache_path(cache_key, reply.model).write_text(
            json.dumps({"cache_key": cache_key,
                        "prompt_version": self.prompt_version,
                        "provider": reply.provider, "model": reply.model,
                        "data": reply.data}),
            encoding="utf-8",
        )

    # ----------------------------------------------------------------- public

    def complete_json(self, prompt: str, cache_key: str) -> LLMReply:
        """Ask the active provider for JSON. Cached by (prompt_version, key)."""
        last_error = "no provider attempted"
        while self._active < len(self.providers):
            name = self.providers[self._active]
            model = self.cfg[name]["model"]
            cached = self._read_cache(cache_key, model)
            if cached is not None:
                self.cache_hits += 1
                return cached
            backoff = Backoff(base_s=20)
            for _ in range(self.cfg["quota_failures_before_fallback"]):
                self._limiters[name].wait()
                try:
                    text = self._request(name, prompt)
                except _Retryable as exc:
                    last_error = str(exc)
                    backoff.sleep(f"{name}: {exc}")
                    continue
                except requests.RequestException as exc:
                    last_error = str(exc)
                    backoff.sleep(f"{name}: {exc}")
                    continue
                self.calls += 1
                reply = LLMReply(parse_json(text), name, model, cached=False)
                self._write_cache(cache_key, reply)
                return reply
            log.warning("%s exhausted (%s) — switching provider", name, last_error)
            self._active += 1
        raise LLMError(f"all providers failed: {last_error}")

    # --------------------------------------------------------------- internal

    def _key(self, provider: str) -> str:
        if provider not in self._keys:
            self._keys[provider] = require_env(self.cfg[provider]["key_env"])
        return self._keys[provider]

    def _request(self, provider: str, prompt: str) -> str:
        pcfg = self.cfg[provider]
        timeout = self.cfg["timeout_s"]
        if provider == "gemini":
            resp = requests.post(
                f"{pcfg['api_base']}/models/{pcfg['model']}:generateContent",
                params={"key": self._key(provider)},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": self.cfg["temperature"],
                        "maxOutputTokens": self.cfg["max_output_tokens"],
                        "responseMimeType": "application/json",
                    },
                },
                timeout=timeout,
            )
            self._raise_for_status(resp)
            candidates = resp.json().get("candidates") or []
            if not candidates:
                raise _Retryable("empty candidates (likely a safety block)")
            parts = candidates[0].get("content", {}).get("parts") or []
            return "".join(p.get("text", "") for p in parts)

        if provider == "groq":
            resp = requests.post(
                f"{pcfg['api_base']}/chat/completions",
                headers={"Authorization": f"Bearer {self._key(provider)}"},
                json={
                    "model": pcfg["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.cfg["temperature"],
                    "max_tokens": self.cfg["max_output_tokens"],
                    "response_format": {"type": "json_object"},
                },
                timeout=timeout,
            )
            self._raise_for_status(resp)
            return resp.json()["choices"][0]["message"]["content"]

        raise LLMError(f"unknown provider {provider!r}")

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code in QUOTA_STATUSES or resp.status_code >= 500:
            raise _Retryable(f"HTTP {resp.status_code} {resp.text[:120]}")
        if not resp.ok:
            raise LLMError(f"HTTP {resp.status_code} {resp.text[:300]}")


class _Retryable(Exception):
    """Transient/quota failure: back off, then possibly change provider."""


def parse_json(text: str) -> dict:
    """Parse a model reply as JSON, tolerating markdown fences and preamble."""
    text = (text or "").strip()
    if not text:
        raise LLMError("empty response")
    try:
        return json.loads(text)
    except ValueError:
        pass
    match = JSON_BLOCK_RE.search(text)
    if not match:
        raise LLMError(f"no JSON object in response: {text[:200]!r}")
    try:
        return json.loads(match.group(0))
    except ValueError as exc:
        raise LLMError(f"malformed JSON: {text[:200]!r}") from exc
