"""Item filtering and the scheduled split — turning filings into labels.

An 8-K carries one or more item codes saying what happened: 2.02 is results,
5.02 a director or officer change, 1.01 a material agreement. Those codes are
the project's free labels, published by the regulator, which is why no
hand-labelling or LLM labelling appears anywhere in this repo.

Two things happen here. Codes that are not events get dropped — 9.01 is
"Financial Statements and Exhibits", an attachment marker that rides along with
real items and would dominate the label distribution if kept. And every
survivor is tagged scheduled or unscheduled, because an earnings date published
weeks ahead is a fundamentally different prediction problem from a surprise
resignation, and pooling them would flatter the result.

Item codes are STRINGS throughout. "1.10" and "1.1" are different codes and a
float would silently merge them.

`usable` is deliberately not set here. Materiality (P4-04) is the last gate and
owns the final verdict; this pass records why an event is out, not whether it
is in.

Usage:
  python -m src.pipeline.events            # apply the filters and report
  python -m src.pipeline.events --report   # report only, write nothing
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from typing import NamedTuple

from src import db
from src.pipeline import coverage
from src.utils.config import load_config

log = logging.getLogger(__name__)


class Label(NamedTuple):
    """What one filing's item codes say about it."""
    kept: list[str]              # codes surviving exclusion
    is_scheduled: int            # 1 if any survivor was known in advance
    exclude_reason: str | None   # set when the event is not an event


def parse_items(items: str | None) -> list[str]:
    """Comma-separated codes -> list of strings, blanks dropped."""
    return [c.strip() for c in (items or "").split(",") if c.strip()]


def classify_items(cfg: dict, items: str | None) -> Label:
    """Drop non-event codes, then decide scheduled vs unscheduled.

    **Scheduled if ANY surviving code is scheduled.** An 8-K carrying
    `2.02,8.01` is earnings plus something else; the earnings date was public
    weeks ahead, so the filing's timing was known. Calling it unscheduled
    because it also contains a surprise would overstate the surprise. 4,945 of
    16,842 events carry more than one code, so this is not a corner case.

    **A code in none of the lists is kept, and is unscheduled.**
    `unscheduled_focus` is a reporting-emphasis list, not a filter — treating
    absence from it as exclusion would silently shrink the study.
    """
    icfg = cfg["items"]
    codes = parse_items(items)
    if not codes:
        return Label([], 0, "no_item_codes")

    kept = [c for c in codes if c not in icfg["exclude"]]
    if not kept:
        return Label([], 0, "only_excluded_items")

    scheduled = int(any(c in icfg["scheduled"] for c in kept))
    return Label(kept, scheduled, None)


def coverage_exclusions(cfg: dict, conn) -> dict[str, str]:
    """accession number -> why its price history is unusable.

    Issue 25: 14 tickers changed symbol, and the provider merges daily history
    across the change but not hourly, so those events have no pre-event bars.
    No other task owned excluding them, which is how they nearly reached the
    feature matrix as NaNs to be imputed or silently zeroed.

    Joined on the accession number rather than (ticker, timestamp), which is
    not guaranteed unique.
    """
    return {v.accession_no: f"no_price_coverage:{v.outcome}"
            for v in coverage.audit(cfg, conn) if v.outcome != "ok"}


def apply_filters(cfg: dict, conn) -> list[dict]:
    """Label every event and record why the excluded ones are out.

    Order matters: the item rule runs first, because "this is not an event" is
    prior to "we cannot measure it". An event that is only a 9.01 attachment
    should not be reported as a price-coverage problem.

    The reason is rewritten on every pass rather than merged, so a reason that
    no longer applies cannot go stale after a config change.
    """
    # Events first: it is the cheap query, and it gives the error that actually
    # names what to do. Running the coverage audit ahead of it would report a
    # missing universe when the real problem is that no events exist yet.
    rows = conn.execute(
        "SELECT event_id, accession_no, items FROM events").fetchall()
    if not rows:
        raise SystemExit(
            "no events to filter — run `python -m src.pipeline.t0 --build` first."
        )
    no_bars = coverage_exclusions(cfg, conn)

    out = []
    for r in rows:
        label = classify_items(cfg, r["items"])
        reason = label.exclude_reason or no_bars.get(r["accession_no"])
        out.append({"event_id": r["event_id"], "accession_no": r["accession_no"],
                    "is_scheduled": label.is_scheduled,
                    "exclude_reason": reason})

    if all(r["exclude_reason"] for r in out):
        raise SystemExit(
            f"EVERY one of {len(out)} events was excluded — check items.exclude "
            f"in config. Do not treat this run as successful."
        )
    return out


def write_filters(cfg: dict, conn) -> int:
    """Apply and store. Returns the number of events left standing."""
    rows = apply_filters(cfg, conn)
    conn.executemany(
        "UPDATE events SET is_scheduled = :is_scheduled, "
        "exclude_reason = :exclude_reason WHERE event_id = :event_id",
        [{k: r[k] for k in ("is_scheduled", "exclude_reason", "event_id")}
         for r in rows],
    )
    conn.commit()
    kept = sum(1 for r in rows if not r["exclude_reason"])
    log.info("events: %d labelled, %d excluded, %d remain",
             len(rows), len(rows) - kept, kept)
    return kept


def print_report(cfg: dict, conn) -> None:
    """The acceptance check: what was dropped, why, and the resulting split."""
    rows = apply_filters(cfg, conn)
    reasons = Counter(r["exclude_reason"] for r in rows if r["exclude_reason"])
    kept = [r for r in rows if not r["exclude_reason"]]
    scheduled = sum(r["is_scheduled"] for r in kept)

    print(f"\n=== Item filtering and the scheduled split ===")
    print(f"excluded codes : {', '.join(cfg['items']['exclude'])}")
    print(f"scheduled codes: {', '.join(cfg['items']['scheduled'])}")
    print(f"\nevents in      : {len(rows):,}")
    for reason, n in reasons.most_common():
        print(f"  excluded {reason:<28} {n:>6,}")
    print(f"  remaining {'':<28} {len(kept):>6,}")
    print(f"\nof the {len(kept):,} remaining:")
    print(f"  scheduled   : {scheduled:>6,}  ({scheduled/len(kept):.1%})")
    print(f"  unscheduled : {len(kept)-scheduled:>6,}  "
          f"({1-scheduled/len(kept):.1%})")
    print(f"\nEvery headline number is reported split this way, never pooled: a "
          f"known\nearnings date and a surprise resignation are different "
          f"prediction problems.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true",
                        help="report only, writing nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    if not args.report:
        write_filters(cfg, conn)
    print_report(cfg, conn)


if __name__ == "__main__":
    main()
