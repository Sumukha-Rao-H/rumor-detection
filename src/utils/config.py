"""Config loader — the single entry point for config/config.yaml and .env.

All code takes its knobs from here (plan §0 rule 5): no hardcoded tickers,
dates, paths, or API URLs anywhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def _abs(value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else REPO_ROOT / p)


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml and .env. Relative paths in `paths:`, and any `*_csv`
    key in any section, are resolved against the repo root so collectors work
    from any CWD."""
    load_dotenv(REPO_ROOT / ".env")
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key, value in cfg.get("paths", {}).items():
        cfg["paths"][key] = _abs(value)
    for section in cfg.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if key.endswith("_csv") and isinstance(value, str):
                    section[key] = _abs(value)
    return cfg


def require_env(name: str) -> str:
    """Fetch a secret from the environment, failing loudly if missing."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name!r} — copy .env.example "
            f"to .env and fill it in."
        )
    return value
