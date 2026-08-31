"""The t0 correction — when did the market actually learn?

The SEC's acceptance time is not the moment news became public. Companies wire
a press release first and file the 8-K with that release attached minutes to
hours later; acceptance times cluster at 20:00-21:00 UTC while earnings
releases hit the wire at ~16:05 ET. Using acceptance alone silently counts time
the market already knew as "advance warning", which is the single easiest way
to report a result that is too good.

    t0 = min(8-K acceptanceDateTime, earliest credible article)

This module finds the second term. Storing all three columns is P4-02's job;
here we match, and measure how well the matching works.

Two limits are inherent and belong in the report rather than in a fix:

  * We match on ticker and time only. Nothing here confirms the article is
    ABOUT this filing — checking that would mean reading article text, which
    the project's framing excludes.
  * Finnhub's free tier carries no wire services, so a tier-1 t0 falls back to
    filing time almost everywhere. `--report` states that as a number.

Usage:
  python -m src.pipeline.t0 --report          # match rate, gap distribution
  python -m src.pipeline.t0 --report --tier 1 # wires only, the sensitivity check
  python -m src.pipeline.t0 --sample 20       # matched headlines, to eyeball
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import NamedTuple

from src import db
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, ts_to_iso

log = logging.getLogger(__name__)

HOUR_S = 3600


class Match(NamedTuple):
    """One filing's t0 candidates."""
    ticker: str
    acceptance_utc: int
    news_utc: int | None      # earliest credible article in the lookback
    t0_utc: int               # min(acceptance, news)
    source: str               # 'news' | 'filing'

    @property
    def gap_hours(self) -> float:
        """How much earlier the news moment is than acceptance."""
        return (self.acceptance_utc - self.t0_utc) / HOUR_S


def lookback_window(cfg: dict, acceptance_utc: int) -> tuple[int, int]:
    """The span searched for the release: `t0_lookback_hours` before acceptance.

    Ends AT acceptance and never reaches past it. An article published later
    cannot lower t0, because t0 is a minimum — and reaching forward would be
    exactly the look-ahead the leakage tests exist to prevent.
    """
    return (acceptance_utc - cfg["news"]["t0_lookback_hours"] * HOUR_S,
            acceptance_utc)


def match_filing(cfg: dict, conn, ticker: str, acceptance_utc: int,
                 max_tier: int | None = 2) -> Match:
    """Find the earliest credible article before one filing.

    `max_tier=1` restricts to wires and top-tier outlets — the release itself;
    `max_tier=2` also admits fast republishers of wire copy. Running both and
    reporting the difference is the sensitivity analysis P1-13b promised, and
    on this data it is stark: no tier-1 article exists anywhere.

    Publication time only, never the aggregator's crawl time (P1-14): crawl
    time lags publication by an unknown amount and would push t0 later.
    """
    lo, hi = lookback_window(cfg, acceptance_utc)
    news_utc = db.earliest_news_ts(conn, ticker, lo, hi, max_tier=max_tier)
    if news_utc is None:
        return Match(ticker, acceptance_utc, None, acceptance_utc, "filing")
    return Match(ticker, acceptance_utc, news_utc,
                 min(acceptance_utc, news_utc), "news")


def match_all(cfg: dict, conn, max_tier: int | None = 2,
              universe_only: bool = True) -> list[Match]:
    """Match every in-window filing. Read-only; P4-02 stores the result."""
    lo = date_str_to_ts(cfg["study_window"]["start"])
    hi = date_str_to_ts(cfg["study_window"]["end"])
    universe = set(db.universe_tickers(conn)) if universe_only else None

    filings = db.filing_acceptance_times(conn, lo, hi, cfg["edgar"]["forms"])
    if universe is not None:
        filings = [(t, a) for t, a in filings if t in universe]
    if not filings:
        raise SystemExit(
            "no in-window filings to match — run the EDGAR collector and the "
            "liquidity filter first."
        )
    return [match_filing(cfg, conn, t, a, max_tier=max_tier) for t, a in filings]


def gap_percentiles(matches: list[Match]) -> dict[str, float]:
    """Where the matched articles actually sit, relative to acceptance.

    This is the diagnostic behind issue 27. A press release precedes its 8-K by
    minutes to a few hours; a median far beyond that means the rule is matching
    unrelated daily coverage rather than the release.
    """
    gaps = sorted(m.gap_hours for m in matches if m.source == "news")
    if not gaps:
        return {}
    def q(p: float) -> float:
        return gaps[min(int(len(gaps) * p), len(gaps) - 1)]
    return {"p10": q(.10), "p25": q(.25), "p50": q(.50), "p75": q(.75),
            "p90": q(.90), "mean": sum(gaps) / len(gaps),
            "within_2h": sum(g <= 2 for g in gaps) / len(gaps),
            "beyond_6h": sum(g > 6 for g in gaps) / len(gaps),
            "beyond_12h": sum(g > 12 for g in gaps) / len(gaps)}


def print_report(cfg: dict, conn, max_tier: int | None = 2) -> None:
    """Match rate and gap distribution — the evidence, not an assertion."""
    matches = match_all(cfg, conn, max_tier=max_tier)
    matched = [m for m in matches if m.source == "news"]
    tier = "any" if max_tier is None else f"<= {max_tier}"

    print(f"\n=== t0 matching (tier {tier}, lookback "
          f"{cfg['news']['t0_lookback_hours']}h) ===")
    print(f"filings          : {len(matches):,}")
    print(f"matched to news  : {len(matched):,}  ({len(matched)/len(matches):.1%})")
    print(f"fall back to filing time: {len(matches)-len(matched):,}")

    pct = gap_percentiles(matches)
    if not pct:
        print("\nNo article matched at this tier — t0 equals filing time for "
              "every event. That is the headline limitation, not a bug.")
        return
    print(f"\ngap, acceptance minus matched article:")
    for k in ("p10", "p25", "p50", "p75", "p90", "mean"):
        print(f"  {k:<5}: {pct[k]:6.1f} h")
    print(f"\nwithin 2h of acceptance (a plausible release window): "
          f"{pct['within_2h']:.0%}")
    print(f"more than  6h before acceptance                     : "
          f"{pct['beyond_6h']:.0%}")
    print(f"more than 12h before acceptance                     : "
          f"{pct['beyond_12h']:.0%}")
    if pct["p50"] > 6:
        print("\n⚠ The median match sits far earlier than a press release "
              "plausibly would.\n  See issue 27: a 24h window on a liquid stock "
              "nearly always contains\n  SOME article, and taking the earliest "
              "picks up unrelated coverage.")


def print_sample(cfg: dict, conn, n: int, max_tier: int | None = 2,
                 seed: int = 0) -> None:
    """Matched headlines next to their filing — how issue 27 was spotted.

    A distribution says something is wrong; the headlines say what.
    """
    matches = [m for m in match_all(cfg, conn, max_tier=max_tier)
               if m.source == "news"]
    random.Random(seed).shuffle(matches)
    print(f"\n=== {min(n, len(matches))} matched filings ===")
    for m in matches[:n]:
        row = conn.execute(
            "SELECT title, source_name FROM news WHERE ticker = ? AND "
            "published_utc = ? LIMIT 1", (m.ticker, m.news_utc)).fetchone()
        title = (row["title"] or "")[:78] if row else "(headline not found)"
        who = row["source_name"] if row else "?"
        print(f"\n  {m.ticker:<6} filed {ts_to_iso(m.acceptance_utc)}")
        print(f"  {'':<6} news  {ts_to_iso(m.news_utc)}  "
              f"({m.gap_hours:.1f}h earlier, {who})")
        print(f"  {'':<6} {title}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true",
                        help="match rate and gap distribution")
    parser.add_argument("--sample", type=int, metavar="N",
                        help="print N matched filings with their headlines")
    parser.add_argument("--tier", type=int, choices=(1, 2),
                        help="credibility tier ceiling (default 2)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    tier = args.tier if args.tier else 2
    if args.sample:
        print_sample(cfg, conn, args.sample, max_tier=tier)
    else:
        print_report(cfg, conn, max_tier=tier)


if __name__ == "__main__":
    main()
