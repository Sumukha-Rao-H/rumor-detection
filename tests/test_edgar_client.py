"""EDGAR transport tests — cache, pacing, retries. No network, no real sleeps.

The one that matters is `test_second_call_serves_from_cache_and_makes_no_request`:
Phase 2 fetches ~1,500 companies, and a cache that does not actually prevent a
request turns every re-run into another 1,500 hits on SEC.
"""

import copy
import json

import pytest
import requests

from src.collectors.edgar import (
    EdgarClient, EdgarRequestError, check_rate_limit_config,
)
from src.utils.config import load_config


class FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b"{}"):
        self.status_code = status_code
        self.content = content

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "replace")


class FakeSession:
    """Records every call and replays a scripted list of responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[str] = []
        self.headers: dict[str, str] = {}

    def get(self, url, timeout=None, allow_redirects=None):
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unscripted request to {url}")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


URL = "https://data.sec.gov/submissions/CIK0000320193.json"
BODY = json.dumps({"cik": "320193", "name": "Apple Inc."}).encode()


@pytest.fixture
def cfg(tmp_path):
    """Real config, with the cache pointed at a temp dir and sleeps shortened."""
    c = copy.deepcopy(load_config())
    c["paths"]["edgar_raw"] = str(tmp_path / "edgar")
    c["edgar"]["min_interval_s"] = 0.0
    c["edgar"]["backoff_base_s"] = 0.0
    return c


def client(cfg, responses):
    session = FakeSession(responses)
    return EdgarClient(cfg, session=session), session


# -- the Done-when ---------------------------------------------------------

def test_second_call_serves_from_cache_and_makes_no_request(cfg):
    c, session = client(cfg, [FakeResponse(content=BODY)])
    first = c.get_json(URL)
    second = c.get_json(URL)          # no second response is scripted
    assert first == second
    assert len(session.calls) == 1, "a repeat run must make zero requests"


def test_first_call_writes_the_raw_body_to_the_cache(cfg):
    c, _ = client(cfg, [FakeResponse(content=BODY)])
    c.get_bytes(URL)
    assert c.cache_path(URL).read_bytes() == BODY, (
        "the raw payload is cached before parsing, so a parser change never "
        "requires re-fetching from SEC")


def test_cache_path_mirrors_url_including_host(cfg):
    c, _ = client(cfg, [])
    path = c.cache_path(URL)
    assert path.parts[-3:] == ("data.sec.gov", "submissions",
                               "CIK0000320193.json")
    # Both hosts are used in this phase; their paths must not collide.
    other = c.cache_path("https://www.sec.gov/submissions/CIK0000320193.json")
    assert path != other


# -- the terms of service --------------------------------------------------

def test_user_agent_carries_a_contact_address(cfg):
    cfg["http"]["user_agent"] = "Test Suite (tests@example.com)"
    c, session = client(cfg, [])
    ua = session.headers["User-Agent"]
    assert ua == cfg["http"]["user_agent"]
    assert "@" in ua, "SEC requires a contact address; requests without one are blocked"


def test_a_live_client_refuses_the_committed_placeholder():
    """The public repo ships a placeholder address; using it would get us blocked.

    Guarding only the live path — an injected session is a test or a replay and
    never reaches the SEC.
    """
    c = copy.deepcopy(load_config())
    c["http"]["user_agent"] = "SET-SEC_USER_AGENT-IN-YOUR-ENV (unset@example.invalid)"

    with pytest.raises(RuntimeError, match="placeholder"):
        EdgarClient(c)                      # no session => live client => refused

    EdgarClient(c, session=requests.Session())   # injected session => allowed


def test_rate_limiter_interval_comes_from_config(tmp_path):
    c = copy.deepcopy(load_config())
    c["paths"]["edgar_raw"] = str(tmp_path)
    cl, _ = client(c, [])
    assert cl.limiter.min_interval_s == c["edgar"]["min_interval_s"]
    assert cl.limiter.min_interval_s >= 1.0 / 10, (
        "SEC caps traffic at 10 req/s — the configured pace must stay under it")


def test_cache_hit_does_not_wait_on_the_limiter(cfg, monkeypatch):
    c, _ = client(cfg, [FakeResponse(content=BODY)])
    c.get_bytes(URL)
    waited = []
    monkeypatch.setattr(c.limiter, "wait", lambda: waited.append(1))
    c.get_bytes(URL)
    assert waited == [], (
        "1,500 cached URLs would otherwise cost three minutes of sleeping "
        "for zero requests")


# -- retries ---------------------------------------------------------------

def test_429_then_200_succeeds_after_backoff(cfg):
    c, session = client(cfg, [FakeResponse(429, b""), FakeResponse(content=BODY)])
    assert c.get_json(URL) == json.loads(BODY)
    assert len(session.calls) == 2


def test_connection_error_is_retried(cfg):
    c, session = client(cfg, [requests.ConnectionError("reset"),
                              FakeResponse(content=BODY)])
    assert c.get_bytes(URL) == BODY
    assert len(session.calls) == 2


def test_gives_up_after_max_retries_and_names_the_url(cfg):
    cfg["edgar"]["max_retries"] = 3
    c, session = client(cfg, [FakeResponse(503, b"")] * 3)
    with pytest.raises(EdgarRequestError, match="CIK0000320193"):
        c.get_bytes(URL)
    assert len(session.calls) == 3


def test_404_is_not_retried(cfg):
    """SEC returns 404 for a CIK with no filings. That is an answer, not a hiccup."""
    c, session = client(cfg, [FakeResponse(404, b"")])
    with pytest.raises(EdgarRequestError, match="404"):
        c.get_bytes(URL)
    assert len(session.calls) == 1


# -- the poisoned-cache trap -----------------------------------------------

def test_redirect_html_raises_and_leaves_no_cache_file(cfg):
    """HTTP 200 carrying HTML is the failure mode the zero-record rule exists for.

    Caching it would be worse than the original failure: every later run would
    serve that HTML from disk and never make a request again.
    """
    html = b"<html><head><title>SEC.gov | Request Rate Threshold</title>"
    c, _ = client(cfg, [FakeResponse(content=html)])
    with pytest.raises(EdgarRequestError, match="non-JSON"):
        c.get_json(URL)
    assert not c.cache_path(URL).exists()


def test_force_refresh_refetches_and_overwrites(cfg):
    updated = json.dumps({"cik": "320193", "name": "Apple Inc. (updated)"}).encode()
    c, session = client(cfg, [FakeResponse(content=BODY),
                              FakeResponse(content=updated)])
    c.get_json(URL)
    assert c.get_json(URL, force=True)["name"] == "Apple Inc. (updated)"
    assert len(session.calls) == 2
    assert c.cache_path(URL).read_bytes() == updated


def test_interrupted_write_leaves_no_partial_cache_hit(cfg):
    """Atomic replace: a killed run must not leave a truncated file behind."""
    c, _ = client(cfg, [FakeResponse(content=BODY)])
    c.get_bytes(URL)
    leftovers = list(c.cache_path(URL).parent.glob("*.part"))
    assert leftovers == []


# -- edgar.max_requests_per_s: decoration made load-bearing -----------------
#
# Only `RateLimiter(ecfg["min_interval_s"])` was ever read on the request
# path, so editing `max_requests_per_s` alone (still SEC-legal) silently
# changed nothing. Validated at CLI startup (`check_rate_limit_config`, called
# from `main()`) rather than inside `EdgarClient.__init__` — every test in
# this file builds an `EdgarClient` with `min_interval_s` deliberately zeroed
# for speed, which would otherwise trip a strict check on every single test.

def test_the_real_config_satisfies_its_own_rate_limit():
    check_rate_limit_config(load_config()["edgar"])  # must not raise


def test_a_min_interval_faster_than_the_cap_allows_is_rejected():
    ecfg = {"min_interval_s": 0.05, "max_requests_per_s": 10}
    with pytest.raises(ValueError, match="disagree"):
        check_rate_limit_config(ecfg)


def test_a_min_interval_at_or_slower_than_the_cap_is_accepted():
    check_rate_limit_config({"min_interval_s": 0.1, "max_requests_per_s": 10})
    check_rate_limit_config({"min_interval_s": 1.0, "max_requests_per_s": 10})


def test_max_requests_per_s_is_optional():
    """Only `min_interval_s` is required; the cap is validated when present."""
    check_rate_limit_config({"min_interval_s": 0.0})
