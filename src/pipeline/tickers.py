"""Ticker extraction (plan §6.1).

Three tiers, each with its own admission rule — a symbol is only accepted from
the evidence that is actually trustworthy for that kind of match:

  $CASHTAG      accepted if the symbol is a real US-listed security
                (config/listed_symbols.csv). Without this check, pennystock
                pump posts invent tickers: 18.8% of the first extraction run's
                links were cashtags for symbols that do not exist.
  BARE TOKEN    accepted only for the core universe (config/tickers.csv) — the
                symbols Reddit actually talks about. Matching bare uppercase
                against all ~10k listed symbols would tag every acronym.
  company name  accepted only for names flagged name_match=1, i.e. names whose
                ticker demonstrably co-occurs with them in this corpus. That
                flag is computed by src/pipeline/build_universe.py; it is what
                stops "price target" -> TGT and "on reddit" -> RDDT.

Blacklisted symbols (common English words that are also tickers) never match,
and posts with more than `max_tickers_per_post` distinct tickers are discarded
as portfolio-spam (extract() returns []).

Both CSVs are generated — run `python -m src.pipeline.build_universe` to
refresh them rather than hand-editing.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
BARE_TOKEN_RE = re.compile(r"\b[A-Z]{2,5}\b")
WORD_RE = re.compile(r"[a-z][a-z&']*")
MAX_NAME_WORDS = 3


def load_universe(csv_path: str | Path) -> dict[str, str]:
    """tickers.csv -> {ticker: company_name}."""
    universe: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            universe[row["ticker"].strip().upper()] = row["name"].strip()
    return universe


def load_matchable_names(csv_path: str | Path) -> dict[str, str]:
    """tickers.csv -> {lowercase company name: ticker} for name_match=1 rows.

    A file without the name_match column (the pre-build_universe format) is
    read conservatively: only multi-word names, which are distinctive enough
    that they cannot collide with ordinary English.
    """
    names: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["name"].strip()
            if not name:
                continue
            flag = row.get("name_match")
            safe = (flag or "").strip() == "1" if flag is not None else " " in name
            if safe:
                names.setdefault(name.lower(), row["ticker"].strip().upper())
    return names


def load_listed(csv_path: str | Path) -> set[str]:
    """listed_symbols.csv -> {ticker}."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return {row["ticker"].strip().upper() for row in csv.DictReader(fh)}


def ngrams(text: str, max_n: int = MAX_NAME_WORDS) -> set[str]:
    """Lowercase word n-grams, for set-lookup name matching."""
    words = WORD_RE.findall(text.lower())
    grams = set(words)
    for n in range(2, max_n + 1):
        grams.update(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    return grams


class TickerExtractor:
    def __init__(
        self,
        universe: dict[str, str],
        blacklist: list[str],
        max_tickers_per_post: int = 3,
        listed: set[str] | None = None,
        matchable_names: dict[str, str] | None = None,
    ):
        self.universe = universe
        self.blacklist = {b.upper() for b in blacklist}
        self.max_tickers = max_tickers_per_post
        # No listed directory available -> cashtags fall back to the core
        # universe, which is stricter, never looser.
        self.listed = (listed or set(universe)) - self.blacklist
        self.names = {
            name: tkr
            for name, tkr in (matchable_names or {}).items()
            if tkr not in self.blacklist
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "TickerExtractor":
        tcfg = cfg["tickers"]
        universe_csv = tcfg["universe_csv"]
        listed_csv = tcfg.get("listed_csv")
        return cls(
            universe=load_universe(universe_csv),
            blacklist=tcfg["blacklist"],
            max_tickers_per_post=tcfg["max_tickers_per_post"],
            listed=(load_listed(listed_csv)
                    if listed_csv and Path(listed_csv).exists() else None),
            matchable_names=load_matchable_names(universe_csv),
        )

    def extract(self, text: str) -> list[str]:
        """Distinct tickers mentioned in text; [] if none or portfolio-spam."""
        if not text:
            return []
        found: set[str] = set()
        for sym in CASHTAG_RE.findall(text):
            if sym in self.listed and sym not in self.blacklist:
                found.add(sym)
        for sym in BARE_TOKEN_RE.findall(text):
            if sym in self.universe and sym not in self.blacklist:
                found.add(sym)
        if self.names:
            for gram in ngrams(text) & self.names.keys():
                found.add(self.names[gram])
        found -= self.blacklist
        if len(found) > self.max_tickers:
            return []
        return sorted(found)
