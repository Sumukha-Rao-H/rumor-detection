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


def require_env(name: str) -> str:
    """Fetch a secret from the environment, failing loudly if missing."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name!r} — copy .env.example "
            f"to .env and fill it in."
        )
    return value
