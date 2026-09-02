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

`usable` is deliberately not set to 1 here. Materiality (P4-04) is the last gate
and owns the positive verdict; this pass records why an event is out, and only
ever forces `usable` DOWN to 0 for an event it excludes, so that nothing this
pass rejects can still reach the feature matrix through `db.usable_events`.

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
from src.utils.timeutils import date_str_to_ts

log = logging.getLogger(__name__)

# The exclusion reasons this pass computes. `exclude_reason` is one shared
# column written by three passes (items and coverage here, materiality in
# P4-04), so a re-run must be able to tell its own verdicts — which it may
# overwrite — from a later pass's, which it must preserve. Rewriting the whole
# column from scratch is what silently erased 8,937 materiality verdicts and
# left the census funnel refusing to add up.
ITEM_REASONS = ("no_item_codes", "only_excluded_items",
                "outside_study_window", "not_in_universe")
COVERAGE_REASON_PREFIX = "no_price_coverage:"


class Label(NamedTuple):
    """What one filing's item codes say about it."""
    kept: list[str]              # codes surviving exclusion
    is_scheduled: int            # 1 if any survivor was known in advance
    exclude_reason: str | None   # set when the event is not an event


def owns_reason(reason: str | None) -> bool:
    """Is `reason` one this pass computes, and may therefore overwrite?"""
    return bool(reason) and (reason in ITEM_REASONS
                             or reason.startswith(COVERAGE_REASON_PREFIX))


def items_config(cfg: dict) -> dict:
    """`cfg['items']`, checked for the one contradiction it can express.

    A code in both `exclude` and `scheduled` states two incompatible intents:
    "this is not an event" and "this is an event whose timing was known".
    Exclusion runs first, so the scheduled entry is unreachable — 5.07 sat in
    both lists, `print_report` announced it as a scheduled code, and every
    5.07 filing was either dropped or called a surprise. Nothing noticed.
    Refuse to run rather than pick one silently.
    """
    icfg = cfg["items"]
    both = sorted(set(icfg["exclude"]) & set(icfg["scheduled"]))
    if both:
        raise SystemExit(
            f"items.exclude and items.scheduled both list {', '.join(both)} — "
            f"contradictory intents. Exclusion runs first, so a code in both "
            f"is dropped and can never be labelled scheduled. Pick one list."
        )
    return icfg


def parse_items(items: str | None) -> list[str]:
    """Comma-separated codes -> list of strings, blanks dropped."""
    return [c.strip() for c in (items or "").split(",") if c.strip()]


def unknown_codes(cfg: dict, codes: list[str]) -> list[str]:
    """Codes in none of the three configured lists.

    Kept, never excluded — see `classify_items`. Counted because the failure
    they announce is silent: if EDGAR's `items` field ever changes shape
    ("2.02 Results of Operations", "5.02(b)"), every code becomes unknown,
    `is_scheduled` drops to 0 across the board, and the scheduled/unscheduled
    split that every headline number is reported against quietly collapses to
    "all unscheduled" with no error anywhere.
    """
    icfg = cfg["items"]
    known = set(icfg["exclude"]) | set(icfg["scheduled"]) | set(
        icfg["unscheduled_focus"])
    return [c for c in codes if c not in known]


def classify_items(cfg: dict, items: str | None) -> Label:
    """Drop non-event codes, then decide scheduled vs unscheduled.

    **Scheduled if ANY surviving code is scheduled.** An 8-K carrying
    `2.02,8.01` is earnings plus something else; the earnings date was public
    weeks ahead, so the filing's timing was known. Calling it unscheduled
    because it also contains a surprise would overstate the surprise. 4,945 of
    16,842 events carry more than one code, so this is not a corner case.

    Decided over the SURVIVING codes, which is only well defined because
    `items_config` guarantees the two lists are disjoint.

    **A code in none of the lists is kept, and is unscheduled.**
    `unscheduled_focus` is a reporting-emphasis list, not a filter — treating
    absence from it as exclusion would silently shrink the study. Item codes
    are free labels published by the regulator, so an unfamiliar one is far
    more likely to be a code this config has not listed than a broken row;
    dropping it would shrink the study on a guess. Unknown codes are counted
    and reported instead — see `unknown_codes`.
    """
    icfg = items_config(cfg)
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
    return {v.accession_no: f"{COVERAGE_REASON_PREFIX}{v.outcome}"
            for v in coverage.audit(cfg, conn) if v.outcome != "ok"}


def study_window(cfg: dict) -> tuple[int, int]:
    """The configured window as epoch seconds, inclusive at both ends."""
    return (date_str_to_ts(cfg["study_window"]["start"]),
            date_str_to_ts(cfg["study_window"]["end"]))


def out_of_scope(row, lo: int, hi: int, universe: set[str]) -> str | None:
    """Why this event row is not part of the study at all, if it is not.

    The coverage audit only inspects in-window filings of in-universe tickers,
    so an event outside that population gets no coverage verdict — and, before
    this check existed, no reason at all, which meant it passed the gate. The
    event builder upserts and never deletes, so shrinking the window (as
    happened on 2026-08-29, from 2024-09-01 to 2025-09-01) or the universe
    leaves exactly such orphans behind, and they would be counted as positives.

    Tested on the acceptance clock, not `t0_utc`: the news correction can pull
    t0 to just before the window start, and that event is still in the study.
    """
    filed = row["t0_filing_utc"]
    if filed is None or not lo <= filed <= hi:
        return "outside_study_window"
    if universe and row["ticker"] not in universe:
        return "not_in_universe"
    return None


def _dominant_hint(reason: str) -> str:
    """What to actually go and check, given the reason that dominates."""
    if reason.startswith(COVERAGE_REASON_PREFIX):
        return ("no event has usable price history — check the market "
                "collector and the `bars` table")
    return {
        "only_excluded_items": "check items.exclude in config",
        "no_item_codes": ("no filing carries any item code — check the EDGAR "
                          "collection"),
        "outside_study_window": ("every event sits outside study_window — "
                                 "rebuild with `python -m src.pipeline.t0 "
                                 "--build`"),
        "not_in_universe": ("no event belongs to an in-universe ticker — "
                            "check `python -m src.pipeline.universe`"),
    }.get(reason, f"check what produces {reason!r}")


def apply_filters(cfg: dict, conn) -> list[dict]:
    """Label every event and record why the excluded ones are out.

    Order matters. Scope comes first — an event outside the study window or
    outside the universe is not a filtering question at all. Then the item
    rule, because "this is not an event" is prior to "we cannot measure it": an
    event that is only a 9.01 attachment should not be reported as a
    price-coverage problem.

    The reason is rewritten on every pass rather than merged, so a reason that
    no longer applies cannot go stale after a config change. Only the reasons
    this pass owns are rewritten — see `write_filters`.
    """
    # Events first: it is the cheap query, and it gives the error that actually
    # names what to do. Running the coverage audit ahead of it would report a
    # missing universe when the real problem is that no events exist yet.
    rows = conn.execute(
        "SELECT event_id, accession_no, ticker, items, t0_filing_utc "
        "FROM events").fetchall()
    if not rows:
        raise SystemExit(
            "no events to filter — run `python -m src.pipeline.t0 --build` first."
        )
    lo, hi = study_window(cfg)
    universe = set(db.universe_tickers(conn))
    no_bars = coverage_exclusions(cfg, conn)

    out = []
    for r in rows:
        label = classify_items(cfg, r["items"])
        reason = (out_of_scope(r, lo, hi, universe)
                  or label.exclude_reason
                  or no_bars.get(r["accession_no"]))
        out.append({"event_id": r["event_id"], "accession_no": r["accession_no"],
                    "kept": label.kept, "is_scheduled": label.is_scheduled,
                    "exclude_reason": reason})

    if all(r["exclude_reason"] for r in out):
        dominant, n = Counter(
            r["exclude_reason"] for r in out).most_common(1)[0]
        raise SystemExit(
            f"EVERY one of {len(out)} events was excluded — {n:,} of them as "
            f"{dominant!r}: {_dominant_hint(dominant)}. Do not treat this run "
            f"as successful."
        )
    return out


def write_filters(cfg: dict, conn) -> int:
    """Apply and store. Returns the number of events left standing.

    Two invariants this used to break. A reason written by a LATER pass is
    preserved: materiality owns `immaterial` / `no_pre_t0_bar`, and rewriting
    the column from scratch NULLed 8,937 of them on every re-run while `usable`
    stayed 0, so the census funnel silently stopped subtracting. And an event
    this pass newly excludes has `usable` forced to 0, because `usable` is what
    `db.usable_events` — the feature matrix and the sampler — actually read; a
    config change that dropped an event used to leave it flagged usable.

    `usable` is never raised to 1 here. That is still materiality's verdict.
    """
    rows = apply_filters(cfg, conn)
    stored = {r["event_id"]: r["exclude_reason"]
              for r in conn.execute("SELECT event_id, exclude_reason FROM events")}

    updates = []
    for r in rows:
        reason = r["exclude_reason"]
        if reason is None and not owns_reason(stored.get(r["event_id"])):
            reason = stored.get(r["event_id"])   # a later pass's verdict
        updates.append({"event_id": r["event_id"],
                        "is_scheduled": r["is_scheduled"],
                        "exclude_reason": reason})

    conn.executemany(
        "UPDATE events SET is_scheduled = :is_scheduled, "
        "exclude_reason = :exclude_reason, "
        "usable = CASE WHEN :exclude_reason IS NULL THEN usable ELSE 0 END "
        "WHERE event_id = :event_id", updates)
    conn.commit()

    leaked = conn.execute(
        "SELECT COUNT(*) FROM events "
        "WHERE usable = 1 AND exclude_reason IS NOT NULL").fetchone()[0]
    if leaked:
        raise SystemExit(
            f"{leaked:,} events are flagged usable while carrying an exclude "
            f"reason — db.usable_events would hand them to the feature matrix."
        )

    kept = sum(1 for r in updates if not r["exclude_reason"])
    log.info("events: %d labelled, %d excluded, %d remain",
             len(updates), len(updates) - kept, kept)
    return kept


def split_reasons(rows) -> dict[str, tuple[int, int, int]]:
    """reason -> (total, scheduled, unscheduled), commonest first.

    Every number is reported split (plan §0 rule 5). A pooled exclusion tally
    hides whether coverage failures or immaterial drops land disproportionately
    on one half of the study.
    """
    total: Counter = Counter()
    sched: Counter = Counter()
    for r in rows:
        reason = r["exclude_reason"]
        if not reason:
            continue
        total[reason] += 1
        sched[reason] += int(r["is_scheduled"] or 0)
    return {reason: (n, sched[reason], n - sched[reason])
            for reason, n in total.most_common()}


def print_report(cfg: dict, conn) -> None:
    """The acceptance check: what was dropped, why, and the resulting split."""
    rows = apply_filters(cfg, conn)
    kept = [r for r in rows if not r["exclude_reason"]]
    scheduled = sum(r["is_scheduled"] for r in kept)
    unknown = Counter(c for r in kept for c in unknown_codes(cfg, r["kept"]))

    print(f"\n=== Item filtering and the scheduled split ===")
    print(f"excluded codes : {', '.join(cfg['items']['exclude'])}")
    print(f"scheduled codes: {', '.join(cfg['items']['scheduled'])}")
    print(f"\nevents in      : {len(rows):,}")
    for reason, (n, s, u) in split_reasons(rows).items():
        print(f"  excluded {reason:<28} {n:>6,}   "
              f"(scheduled {s:,} / unscheduled {u:,})")
    print(f"  remaining {'':<28} {len(kept):>6,}")
    print(f"\nof the {len(kept):,} remaining:")
    print(f"  scheduled   : {scheduled:>6,}  ({scheduled/len(kept):.1%})")
    print(f"  unscheduled : {len(kept)-scheduled:>6,}  "
          f"({1-scheduled/len(kept):.1%})")
    if unknown:
        print(f"\ncodes in none of the config lists: "
              f"{sum(unknown.values()):,} occurrences over "
              f"{len(unknown):,} distinct codes")
        print(f"  {', '.join(f'{c} ({n:,})' for c, n in unknown.most_common(8))}")
        print(f"  Kept, and counted as unscheduled. A sudden jump here means "
              f"the item\n  field changed shape and the split is no longer "
              f"meaningful.")
    print(f"\nMateriality (P4-04) still sits downstream and removes more than "
          f"half of\nthe remaining events — run --census for the real positive "
          f"count.")
    print(f"\nEvery headline number is reported split this way, never pooled: a "
          f"known\nearnings date and a surprise resignation are different "
          f"prediction problems.")


def distinct_announcements(rows: list[tuple[str, int]], gap_s: int) -> int:
    """Collapse events for the same ticker that sit within `gap_s` of each other.

    Issue 28: Capricor filed two 8-Ks on the same day about the same news, and
    both are usable with an identical price move. Counting one announcement as
    several positives inflates the total and would let a model be credited
    twice for detecting the same thing.

    Clusters are transitive along consecutive events, not pairwise: three
    filings each an hour apart are one announcement, not two. The boundary is
    exclusive: two filings EXACTLY `gap_s` apart are one announcement.
    """
    by_ticker: dict[str, list[int]] = {}
    for ticker, ts in rows:
        by_ticker.setdefault(ticker, []).append(ts)
    total = 0
    for stamps in by_ticker.values():
        stamps.sort()
        total += 1
        for prev, cur in zip(stamps, stamps[1:]):
            if cur - prev > gap_s:
                total += 1
    return total


def census(cfg: dict, conn) -> dict:
    """The filing -> event -> usable funnel, plus the clustering sensitivity.

    Read-only. Raises rather than soft-landing on zero usable events: the
    positive count is the deliverable's denominator, and a zero is the
    stop-and-tell the plan asks for, not a report to print with 0.0% in it.
    """
    rows = conn.execute(
        "SELECT e.ticker, e.t0_utc, e.t0_filing_utc, e.is_scheduled, e.usable, "
        "e.exclude_reason, e.items, f.form "
        "FROM events e LEFT JOIN filings f ON f.accession_no = e.accession_no"
    ).fetchall()
    if not rows:
        raise SystemExit(
            "no events — run `python -m src.pipeline.t0 --build` first."
        )

    lo, hi = study_window(cfg)
    in_window = [r for r in rows
                 if r["t0_filing_utc"] is not None
                 and lo <= r["t0_filing_utc"] <= hi]
    usable = [r for r in in_window if r["usable"]]
    if not usable:
        raise SystemExit(
            f"0 of {len(rows):,} events are usable — nothing to report. Run "
            f"`python -m src.pipeline.events` and "
            f"`python -m src.pipeline.materiality`, and if the count is still "
            f"zero the study has no positives: a stop-and-tell, see the plan."
        )

    months = (hi - lo) / (365.25 / 12 * 86400)
    if months <= 0:
        raise SystemExit(
            f"study_window is empty ({cfg['study_window']['start']} .. "
            f"{cfg['study_window']['end']}) — no rate can be computed."
        )
    tickers = len({r["ticker"] for r in usable})

    pairs = [(r["ticker"], r["t0_utc"]) for r in usable]
    sched = sum(r["is_scheduled"] or 0 for r in usable)
    return {
        "events": len(rows),
        "out_of_window": len(rows) - len(in_window),
        # usable=0 with no reason: built, but not yet run through the filters.
        "pending": sum(1 for r in in_window
                       if not r["usable"] and not r["exclude_reason"]),
        "usable": len(usable),
        "scheduled": sched,
        "unscheduled": len(usable) - sched,
        "reasons": split_reasons(in_window),
        "tickers": tickers,
        "months": months,
        "per_stock_month": len(usable) / tickers / months,
        "announcements": {h: distinct_announcements(pairs, h * 3600)
                          for h in cfg["census"]["cluster_gap_hours"]},
        "forms": Counter(r["form"] for r in usable),
        "top_items": Counter(
            c for r in usable for c in parse_items(r["items"])
            if c not in cfg["items"]["exclude"]).most_common(8),
        "unknown_items": Counter(
            c for r in usable
            for c in unknown_codes(cfg, parse_items(r["items"]))),
    }


def print_census(cfg: dict, conn) -> None:
    """The acceptance check: the real positive count, against the plan."""
    c = census(cfg, conn)
    ccfg = cfg["census"]
    budget = cfg["eval"]["alert_budget_per_stock_per_month"]
    usable, sched = c["usable"], c["scheduled"]

    print(f"\n=== The real positive count ===")
    print(f"events built            : {c['events']:,}")
    if c["out_of_window"]:
        print(f"  outside study_window {'':<22} {c['out_of_window']:>6,}  "
              f"(stale rows: rebuild the events table)")
    for reason, (n, s, u) in c["reasons"].items():
        print(f"  excluded {reason:<26} {n:>6,}   "
              f"(scheduled {s:,} / unscheduled {u:,})")
    if c["pending"]:
        print(f"  not yet filtered {'':<21} {c['pending']:>6,}  "
              f"(run events, then materiality)")
    print(f"  USABLE POSITIVES {'':<21} {usable:>6,}")
    print(f"\n  scheduled   : {sched:,}  ({sched/usable:.1%})")
    print(f"  unscheduled : {c['unscheduled']:,}  ({c['unscheduled']/usable:.1%})")

    print(f"\n--- against implementation_plan.md §5 ---")
    print(f"plan estimated ~{ccfg['expected_usable_min']:,}-"
          f"{ccfg['expected_usable_max']:,} usable positives "
          f"('10-20% of filings')")
    print(f"actual usable           : {usable:,}")
    print(f"actual unscheduled      : {c['unscheduled']:,}  "
          f"({c['unscheduled']/c['events']:.1%} of events)")
    verdict = ("ABOVE the estimate" if usable > ccfg["expected_usable_max"] else
               "within the estimate" if usable >= ccfg["expected_usable_min"]
               else "BELOW the estimate — a stop-and-tell, see the plan")
    print(f"verdict                 : {verdict}")

    print(f"\n--- issue 28: events are not announcements ---")
    for h, n in c["announcements"].items():
        print(f"  collapsing events within {h:>2}h : {n:,} distinct "
              f"announcements ({usable-n:,} absorbed)")
    print(f"  No de-duplication is applied to the data here — that is a "
          f"labelling\n  decision for negative sampling (P4-12), not for a "
          f"census.")
    if len(c["forms"]) > 1:
        print(f"  by form: " + ", ".join(
            f"{form} {n:,}" for form, n in
            sorted(c["forms"].items(), key=lambda kv: -kv[1])))
        print(f"  An amendment is counted separately from the filing it "
              f"amends, and lands\n  days later, so no clustering gap above "
              f"absorbs it. Whether that is a\n  double count is a study "
              f"design question, not a census one.")

    print(f"\n--- rate, against the alert budget ---")
    print(f"usable events           : {usable:,} across {c['tickers']:,} "
          f"tickers over {c['months']:.1f} months")
    print(f"events per stock/month  : {c['per_stock_month']:.2f}")
    print(f"alert budget            : {budget} per stock/month")
    print(f"  the budget is {budget/c['per_stock_month']:.1f}x the event rate, "
          f"so precision at budget is\n  bounded above by roughly "
          f"{c['per_stock_month']/budget:.0%} even for a perfect detector.")

    print(f"\n--- most common surviving item codes ---")
    for code, n in c["top_items"]:
        tag = "scheduled" if code in cfg["items"]["scheduled"] else ""
        print(f"  {code:<6}{n:>6,}  {tag}")
    if c["unknown_items"]:
        print(f"  {sum(c['unknown_items'].values()):,} occurrences of "
              f"{len(c['unknown_items']):,} codes in no config list, kept and "
              f"counted as unscheduled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true",
                        help="report only, writing nothing")
    parser.add_argument("--census", action="store_true",
                        help="the real positive count, against the plan's "
                             "estimate. Read-only.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    try:
        if args.census:
            print_census(cfg, conn)
            return
        if not args.report:
            write_filters(cfg, conn)
        print_report(cfg, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
