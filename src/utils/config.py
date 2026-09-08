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
def _read_config(cfg_path: Path, mtime_ns: int) -> dict:
    """Parse config.yaml once per (path, modification time). See `load_config`.

    `mtime_ns` is not used in the body — it is here purely as part of the cache
    key, so that editing config.yaml invalidates the cached parse. Keying on
    the path alone meant an edit was invisible for the life of the process,
    which made `app/data.py`'s advertised 5-minute config refresh a no-op: the
    Streamlit cache expired on schedule and then got handed the same stale dict
    underneath. Within a single run nothing edits its own config, so this costs
    one `stat` per call and changes no behaviour there.
    """
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"{cfg_path} did not parse to a mapping (got {type(cfg).__name__}). "
            f"An empty or malformed config.yaml is not a config with defaults — "
            f"every knob in this project is meant to come from that file."
        )
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else REPO_ROOT / p)

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

    The cache is keyed on the file's modification time as well as its path, so
    a genuine EDIT to config.yaml is picked up while a run still sees one
    configuration throughout — nothing edits its own config mid-run.

    A deep copy is returned so a caller mutating the result cannot corrupt the
    config every other caller sees.

    Call `load_config.cache_clear()` if a test genuinely needs a re-read.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    # Outside the cached body on purpose. It used to sit inside, so the
    # environment was frozen at whatever it happened to be on the first
    # `load_config()` call anywhere in the process — setting SEC_USER_AGENT
    # after that had no effect at all. `override=False` is python-dotenv's
    # default, passed explicitly so this can never clobber a value a test has
    # already set with `monkeypatch.setenv`; a test that instead needs a key to
    # be ABSENT must `monkeypatch.delenv`, since nothing here stops `.env` from
    # having populated it earlier in the process.
    load_dotenv(REPO_ROOT / ".env", override=False)
    cfg = copy.deepcopy(_read_config(cfg_path, os.stat(cfg_path).st_mtime_ns))

    # This repository is public, so the SEC contact address cannot live in the
    # committed config — see the comment on `http.user_agent` in config.yaml.
    # The environment wins when it is set; otherwise the placeholder survives
    # and `require_sec_user_agent` refuses it at the point of use. Applied to
    # the caller's copy rather than the cached parse so that it tracks the
    # environment rather than whichever call happened to populate the cache.
    env_ua = os.getenv("SEC_USER_AGENT", "").strip()
    if env_ua:
        cfg.setdefault("http", {})["user_agent"] = env_ua
    return cfg


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
