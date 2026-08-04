"""Build the ticker universe from the US-listed symbol directory + the corpus.

Implements plan §6.1 item 3 ("extend empirically — after the first extraction
run, review the extracted tickers and prune junk") as a reproducible script
rather than a hand-edited CSV. Three outputs, all written to config/:

  listed_symbols.csv  every US-listed symbol (NASDAQ Trader SymbolDirectory).
                      A `$CASHTAG` only counts if the symbol is in here — that
                      is what stops pump spam inventing tickers.
  tickers.csv         the *core* universe: listed symbols ranked by how often
                      Reddit actually mentions them, capped at tickers.core_size.
                      Bare uppercase tokens only match against this list.
  (column) name_match whether a company name is safe to match as free text.

The name_match decision is data-driven. For a name that is a single word this
corpus uses often — the only case that can collide with ordinary English — the
name is usable only if the symbol itself shows up alongside it often enough:

    precision(name) = P(symbol appears in post | name appears in post)

Real company references co-occur with their ticker ("Nvidia ... NVDA"); English
words that happen to be company names do not ("price target" almost never
appears alongside TGT). This is what kills target→TGT, reddit→RDDT, block→SQ,
shell→SHEL, which were 12% of all ticker links before this script existed.

Usage:
  python -m src.pipeline.build_universe            # download + scan dumps
  python -m src.pipeline.build_universe --no-download   # reuse cached directory
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import Counter
from pathlib import Path

import requests

from src.collectors.arctic_shift import CREATED_UTC_RE, find_dumps, iter_raw_lines
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

log = logging.getLogger(__name__)

CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
BARE_TOKEN_RE = re.compile(r"\b[A-Z]{2,5}\b")
WORD_RE = re.compile(r"[a-z][a-z&']*")
SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")
MAX_NAME_WORDS = 3  # longest company name we look for as an n-gram

# Security-name noise from the symbol directory, stripped before matching.
SHARE_CLASS_RE = re.compile(
    r"\s*[-–]?\s*(?:(?:class|series)\s+[a-z0-9]+\s+)?"
    r"(?:common\s+stock|ordinary\s+shares?|common\s+shares?|capital\s+stock|"
    r"american\s+depositary\s+(?:shares?|receipts?)|depositary\s+shares?|"
    r"units?|warrants?|rights?|preferred\s+stock|notes?|"
    r"limited\s+partnership|beneficial\s+interest).*$",
    re.IGNORECASE,
)
LEGAL_SUFFIX_RE = re.compile(
    r"[,\s]+(?:inc|incorporated|corp|corporation|company|co|ltd|limited|plc|"
    r"llc|lp|nv|sa|ag|se|holdings?|group|the)\.?$",
    re.IGNORECASE,
)


def fetch_symbol_directory(cfg: dict, download: bool = True) -> list[dict]:
    """NASDAQ Trader SymbolDirectory files -> [{ticker, name, etf}].

    Raw payloads are cached under data/raw/ (plan §3) so a rebuild is offline.
    """
    cache_dir = Path(cfg["paths"]["symbol_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for url in cfg["tickers"]["symbol_dir_urls"]:
        cached = cache_dir / url.rsplit("/", 1)[-1]
        if download or not cached.exists():
            log.info("Downloading %s", url)
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            cached.write_text(resp.text, encoding="utf-8")
        text = cached.read_text(encoding="utf-8")

        reader = csv.DictReader(text.splitlines(), delimiter="|")
        for row in reader:
            # Both files end with a "File Creation Time: ..." trailer line.
            symbol = (row.get("Symbol") or row.get("ACT Symbol") or "").strip()
            if not SYMBOL_RE.match(symbol):
                continue
            if (row.get("Test Issue") or "").strip() == "Y":
                continue
            rows.append({
                "ticker": symbol,
                "name": clean_company_name(row.get("Security Name") or ""),
                "etf": 1 if (row.get("ETF") or "").strip() == "Y" else 0,
            })

    deduped = {r["ticker"]: r for r in rows}
    log.info("Symbol directory: %d listed symbols", len(deduped))
    return sorted(deduped.values(), key=lambda r: r["ticker"])


def clean_company_name(raw: str) -> str:
    """'Apple Inc. - Common Stock' -> 'Apple'. Empty string if nothing usable."""
    name = SHARE_CLASS_RE.sub("", raw.strip())
    for _ in range(3):  # 'Alphabet Inc. Class A' -> strip suffixes repeatedly
        stripped = LEGAL_SUFFIX_RE.sub("", name).strip(" ,.-")
        if stripped == name.strip(" ,.-"):
            break
        name = stripped
    name = re.sub(r"\s+", " ", name).strip(" ,.-")
    return name if len(name) >= 3 and re.search(r"[A-Za-z]{3}", name) else ""


def _ngrams(text: str, max_n: int) -> set[str]:
    words = WORD_RE.findall(text.lower())
    grams = set(words)
    for n in range(2, max_n + 1):
        grams.update(
            " ".join(words[i:i + n]) for i in range(len(words) - n + 1)
        )
    return grams


class CorpusStats:
    """One pass over the dumps: symbol mention counts + name/symbol co-occurrence."""

    def __init__(self, candidate_names: dict[str, str], start_ts: int, end_ts: int):
        self.candidates = candidate_names  # lowercase name -> ticker
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.mentions: Counter[str] = Counter()      # symbol -> posts mentioning it
        self.cashtags: Counter[str] = Counter()      # ... written as $SYMBOL
        self.name_seen: Counter[str] = Counter()     # name -> posts containing it
        self.name_hit: Counter[str] = Counter()      # ... and also its symbol
        self.posts = 0

    def offer_text(self, text: str) -> None:
        self.posts += 1
        cashtags = set(CASHTAG_RE.findall(text))
        symbols = cashtags | set(BARE_TOKEN_RE.findall(text))
        for sym in cashtags:
            self.cashtags[sym] += 1
        for sym in symbols:
            self.mentions[sym] += 1
        for gram in _ngrams(text, MAX_NAME_WORDS) & self.candidates.keys():
            self.name_seen[gram] += 1
            if self.candidates[gram] in symbols:
                self.name_hit[gram] += 1

    def precision(self, name: str) -> float | None:
        """P(symbol in post | name in post); None when support is too thin."""
        seen = self.name_seen[name]
        return self.name_hit[name] / seen if seen else None


def scan_corpus(cfg: dict, candidate_names: dict[str, str]) -> CorpusStats:
    """Stream every dump once, gathering mention and co-occurrence counts."""
    import json

    stats = CorpusStats(
        candidate_names,
        date_str_to_ts(cfg["backtest_window"]["start"]),
        date_str_to_ts(cfg["backtest_window"]["end"]),
    )
    for path in find_dumps(cfg["paths"]["arctic_dumps"]):
        log.info("Scanning %s ...", path.name)
        for line in iter_raw_lines(path):
            match = CREATED_UTC_RE.search(line)
            if not match or not (stats.start_ts <= int(match.group(1)) < stats.end_ts):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            title = rec.get("title")
            if title is None:  # comment dumps have no title — posts only
                continue
            stats.offer_text(f"{title} {rec.get('selftext') or ''}")
        log.info("  %d in-window posts scanned so far", stats.posts)
    return stats


def build_core_universe(
    listed: list[dict], stats: CorpusStats, cfg: dict, curated: dict[str, str]
) -> list[dict]:
    """Rank listed symbols by how often Reddit *cashtags* them, keep the top N.

    Ranking on cashtags rather than all mentions is deliberate: bare uppercase
    counts are dominated by ordinary acronyms that happen to be listed symbols
    (USD, EU, RSI, TLDR, HYSA, PR). Writing `$XYZ` is an intentional reference
    to a security, so it is the honest popularity signal — and admission to the
    core universe is what licenses bare-token matching later.
    """
    tcfg = cfg["tickers"]
    blacklist = {b.upper() for b in tcfg["blacklist"]}
    min_support = tcfg["name_match_min_support"]
    min_precision = tcfg["name_match_min_precision"]

    ranked = sorted(
        (r for r in listed if r["ticker"] not in blacklist),
        key=lambda r: (-stats.cashtags[r["ticker"]], -stats.mentions[r["ticker"]]),
    )
    core = [r for r in ranked
            if stats.cashtags[r["ticker"]] >= tcfg["core_min_cashtags"]
            ][: tcfg["core_size"]]
    kept = {r["ticker"] for r in core}
    # Curated symbols stay in even if the corpus barely mentions them.
    core += [r for r in ranked if r["ticker"] in curated and r["ticker"] not in kept]

    out = []
    for row in core:
        ticker = row["ticker"]
        name = curated.get(ticker) or row["name"]
        precision = stats.precision(name.lower()) if name else None
        support = stats.name_seen[name.lower()] if name else 0
        # Only ambiguous names need the co-occurrence test. A multi-word name
        # ("Bank of America") cannot collide with ordinary English, and neither
        # can a single word this corpus almost never uses ("Erayak") — both are
        # trusted outright. A single *frequent* word has to earn it, which is
        # exactly where target/reddit/robinhood/shell/ford/visa fail.
        if not name:
            safe = False
        elif " " in name or support < min_support:
            safe = True
        else:
            safe = precision >= min_precision
        out.append({
            "ticker": ticker,
            "name": name,
            "name_match": int(safe),
            "etf": row["etf"],
            "cashtags": stats.cashtags[ticker],
            "mentions": stats.mentions[ticker],
            "name_posts": support,
            "name_precision": round(precision, 4) if precision is not None else "",
        })
    return sorted(out, key=lambda r: (-r["cashtags"], -r["mentions"], r["ticker"]))


def write_csv(path: str | Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: r[c] for c in columns} for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-download", action="store_true",
                        help="reuse the cached symbol directory under data/raw/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config()
    tcfg = cfg["tickers"]
    listed = fetch_symbol_directory(cfg, download=not args.no_download)

    # Hand-written names are cleaner than directory names ("Alphabet" beats
    # "Alphabet Inc. Class A Common Stock"), and their symbols stay in the core
    # universe regardless of mention count. This file is an input, never
    # rewritten — the generated universe_csv is the output.
    curated: dict[str, str] = {}
    curated_csv = Path(tcfg["curated_csv"])
    if curated_csv.exists():
        with open(curated_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                curated[row["ticker"].strip().upper()] = row["name"].strip()
        log.info("Seeded %d curated names from %s", len(curated), curated_csv.name)

    candidates = {}
    for row in listed:
        name = curated.get(row["ticker"]) or row["name"]
        if name and len(name.split()) <= MAX_NAME_WORDS:
            candidates.setdefault(name.lower(), row["ticker"])

    stats = scan_corpus(cfg, candidates)
    core = build_core_universe(listed, stats, cfg, curated)

    write_csv(tcfg["listed_csv"], listed, ("ticker", "name", "etf"))
    write_csv(tcfg["universe_csv"], core,
              ("ticker", "name", "name_match", "etf", "cashtags", "mentions",
               "name_posts", "name_precision"))

    safe = sum(r["name_match"] for r in core)
    log.info("Scanned %d in-window posts.", stats.posts)
    log.info("Wrote %s (%d symbols) and %s (%d core, %d name-matchable).",
             tcfg["listed_csv"], len(listed), tcfg["universe_csv"], len(core), safe)


if __name__ == "__main__":
    main()
