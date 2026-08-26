"""Ticker extraction (plan §6.1).

A post maps to a ticker only if:
  - cashtag match ($TSLA), OR
  - company-name dictionary match (whole word, case-insensitive), OR
  - bare uppercase token that is in the ticker universe.
Blacklisted symbols (common English words that are also tickers) never match.
Posts with more than `max_tickers_per_post` distinct tickers are discarded
as portfolio-spam (extract() returns []).

The universe lives in config/tickers.csv (columns: ticker,name) — extend it
empirically after the first extraction run by reviewing top-100 extracted
tickers for junk.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
BARE_TOKEN_RE = re.compile(r"\b[A-Z]{2,5}\b")


def load_universe(csv_path: str | Path) -> dict[str, str]:
    """tickers.csv -> {ticker: company_name}."""
    universe: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            universe[row["ticker"].strip().upper()] = row["name"].strip()
    return universe


class TickerExtractor:
    def __init__(
        self,
        universe: dict[str, str],
        blacklist: list[str],
        max_tickers_per_post: int = 3,
    ):
        self.universe = universe
        self.blacklist = {b.upper() for b in blacklist}
        self.max_tickers = max_tickers_per_post
        # Longest-name-first alternation so "Berkshire Hathaway" wins over "Berkshire"
        names = sorted(
            ((name, tkr) for tkr, name in universe.items() if name),
            key=lambda x: -len(x[0]),
        )
        self._name_to_ticker = {name.lower(): tkr for name, tkr in names}
        self._name_re = re.compile(
            r"\b(" + "|".join(re.escape(name) for name, _ in names) + r")\b",
            re.IGNORECASE,
        ) if names else None

    @classmethod
    def from_config(cls, cfg: dict) -> "TickerExtractor":
        tcfg = cfg["tickers"]
        return cls(
            universe=load_universe(tcfg["universe_csv"]),
            blacklist=tcfg["blacklist"],
            max_tickers_per_post=tcfg["max_tickers_per_post"],
        )

    def extract(self, text: str) -> list[str]:
        """Distinct tickers mentioned in text; [] if none or portfolio-spam."""
        if not text:
            return []
        found: set[str] = set()
        for sym in CASHTAG_RE.findall(text):
            if sym not in self.blacklist:
                found.add(sym)
        for sym in BARE_TOKEN_RE.findall(text):
            if sym in self.universe and sym not in self.blacklist:
                found.add(sym)
        if self._name_re:
            for match in self._name_re.findall(text):
                found.add(self._name_to_ticker[match.lower()])
        found -= self.blacklist
        if len(found) > self.max_tickers:
            return []
        return sorted(found)
