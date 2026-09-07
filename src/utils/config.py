"""Config loader — the single entry point for config/config.yaml and .env.

All code takes its knobs from here (plan §0 rule 5): no hardcoded tickers,
dates, paths, or API URLs anywhere else.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=4)
def _read_config(cfg_path: Path) -> dict:
    """Parse config.yaml once per path. See `load_config` for why.

    `load_dotenv` runs here as a side effect of the FIRST `load_config()`
    call anywhere in the process — including from code that only wants
    `market.calendar` or some other non-secret key — because this is cached
    to run once. `override=False` is passed explicitly (it is already
    python-dotenv's default) so this can never clobber a value a test has
    already set with `monkeypatch.setenv`; a test that instead needs a key to
    be ABSENT must `monkeypatch.delenv` at test time, since nothing here
    stops `.env` from having populated it earlier in the process.
    """
    load_dotenv(REPO_ROOT / ".env", override=False)
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else REPO_ROOT / p)

    # This repository is public, so the SEC contact address cannot live in the
    # committed config — see the comment on `http.user_agent` in config.yaml.
    # The environment wins when it is set; otherwise the placeholder survives
    # and `require_sec_user_agent` refuses it at the point of use.
    env_ua = os.getenv("SEC_USER_AGENT", "").strip()
    if env_ua:
        cfg.setdefault("http", {})["user_agent"] = env_ua
    return cfg


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml and .env. Relative paths in `paths:` are resolved
    against the repo root so collectors work from any CWD.

    Cached, and for two reasons. The obvious one is cost: this was re-parsing
    the YAML and re-reading .env on every call — 61 disk reads in a single
    `report_table`, about half its runtime. The important one is consistency:
    a run must use ONE configuration throughout. Re-reading mid-run would let
    an edit to config.yaml change a threshold partway through an evaluation,
    which is exactly the kind of silent, unreproducible drift this project
    cannot afford.

    A deep copy is returned so a caller mutating the result cannot corrupt the
    config every other caller sees.

    Call `load_config.cache_clear()` if a test genuinely needs a re-read.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    return copy.deepcopy(_read_config(cfg_path))


load_config.cache_clear = _read_config.cache_clear  # type: ignore[attr-defined]


#: Marker for the placeholder User-Agent shipped in the public config.yaml.
#: Matching on this rather than the whole string means the placeholder text
#: can be reworded without silently disarming the check.
PLACEHOLDER_SEC_UA = "SET-SEC_USER_AGENT"


def require_sec_user_agent(cfg: dict) -> str:
    """The SEC User-Agent, refusing the placeholder the public repo ships with.

    The SEC's entire terms of service is "say who you are and how to reach
    you", and it blocks requests that do not. Failing here — loudly, before a
    single request goes out — is kinder than letting a clone run for a while
    and then get blocked for identifying itself as nobody.
    """
    ua = str(cfg.get("http", {}).get("user_agent", "")).strip()
    if not ua or PLACEHOLDER_SEC_UA in ua:
        raise RuntimeError(
            "config.http.user_agent is still the committed placeholder. The SEC "
            "requires a real contact address and blocks requests without one, and "
            "this repository is public so the address cannot be committed. Copy "
            ".env.example to .env and set, for example:\n"
            "    SEC_USER_AGENT=Your Project Name (you@example.com)\n"
            "In GitHub Actions, set it as the SEC_USER_AGENT repository secret."
        )
    return ua


def require_env(name: str) -> str:
    """Fetch a secret from the environment, failing loudly if missing."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name!r} — copy .env.example "
            f"to .env and fill it in."
        )
    return value
