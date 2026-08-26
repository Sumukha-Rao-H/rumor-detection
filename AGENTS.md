# AGENTS.md — Rules for AI agents and contributors

This repo implements **Pre-Announcement Footprints Around SEC Filings**
(Project Code 31, NMAMIT ISE). The authoritative design document is
[`implementation_plan.md`](implementation_plan.md). **Read it fully before
writing any code.** This file is the short operational contract.

Background reading, in order: `implementation_plan.md`, then
`docs/project-ideas-review.pdf` (Idea 1) and `docs/context.md`. Where the PDF
and `context.md` disagree, **`context.md` wins**.

## Non-negotiable rules

1. **Do not change the architecture** without explicit user approval. Frozen
   decisions: SQLite storage; SEC 8-K item codes as free labels; hourly
   decision steps; a two-action stopping problem (WAIT / FLAG); classical
   baselines built *before* the learned policy.
2. **Work phase by phase** (plan §9). Do not start a phase until the previous
   phase's acceptance criteria pass.
3. **All timestamps in UTC everywhere.** Store epoch seconds in SQLite; convert
   at display time only. Use `src/utils/timeutils.py` — never naive datetimes.
4. **Lead time is measured in trading hours, never wall-clock hours.** Most
   8-Ks are filed outside market hours, so "the hours before the filing" is
   usually the previous trading session.
5. **t₀ is `min(8-K acceptanceDateTime, earliest news article)`**, never the
   filing time alone. Both variants get reported. See plan §4.
6. **Never train on future data.** Every feature at time `t` must be computable
   using only data timestamped `≤ t`, with backward-looking windows only.
   Leakage tests in `tests/` are mandatory and must keep passing.
7. **Never report plain accuracy.** The base rate is ~0.3%; always-quiet scores
   99.7%. The headline metric is precision at a fixed alert budget. Every
   number is split scheduled vs unscheduled.
8. **Any collector cycle that parses zero records must fail loudly.** HTTP 200
   responses carrying redirect HTML or empty JSON are the failure mode that
   burns people — a pipeline that looks healthy while writing nothing.
9. **Config-driven code only.** No hardcoded tickers, dates, paths, API URLs,
   thresholds, or rate limits — everything comes from `config/config.yaml`
   (loaded via `src.utils.config.load_config`). Secrets come from `.env`.
10. **Respect external services.** SEC EDGAR: descriptive User-Agent with a
    contact address, ≤10 req/s (config uses 8). Finnhub: ~60 calls/min. GDELT:
    ≥5s between requests and never a blocking dependency. Cache every raw
    response to disk and never re-fetch. Never scrape Reuters/Bloomberg.
11. **Say "footprint", never "insider trading."** Volume spikes have innocent
    causes. We detect a footprint, not a culprit.

## Repo conventions

- Layout is fixed (`src/collectors`, `src/pipeline`, `src/baselines`, `src/rl`,
  `src/eval`, `src/utils`, `app/`, `tests/`). New code goes in the module the
  plan assigns it to.
- Collectors are runnable as modules: `python -m src.collectors.<name> [args]`.
- SQLite schema lives in `src/db.py`. Writes are idempotent upserts — re-running
  any collector must never duplicate rows.
- Raw API payloads are archived under `data/raw/` (gitignored) before parsing.
- `archive/` holds the abandoned Reddit approach. It is kept for the report.
  **Nothing in `src/` may import from it**, and `pytest.ini` excludes it.
- Python ≥3.10, standard library `sqlite3` (no ORM), type hints on public
  functions.

## Git workflow

- **Commit after every completed feature** — small, reviewable commits, one
  feature each, conventional-commit prefixes (`feat:`, `chore:`, `fix:`,
  `test:`, `docs:`).
- **Never add an AI co-author line** or any other AI attribution to commits or PRs.
- Never commit anything under `data/` or `.env`.

## Project status (update this section as phases complete)

| Phase | Status |
|---|---|
| 0 — Repointing from Reddit to EDGAR | ✅ Reddit code archived; schema, config, collectors, docs, tests rebuilt. 20 tests pass. |
| 1 — Week 1 foundations (market-hours helpers, headline metric, eval skeleton, news collection running) | ⏳ not started |
| 2 — Week 2 EDGAR collector + universe + leakage test | ⏳ not started |
| 3 — Week 3 price snapshot, frozen and stamped | ⏳ not started |
| 4 — Week 4 t₀ correction, event filtering, features, sampling | ⏳ not started |
| 5 — Week 5 baselines (always-quiet, z-score, CUSUM, GBM) | ⏳ not started |
| 6 — Weeks 6–8 learned stopping policy | ⏳ not started |
| 7 — Week 8+ live monitor running continuously | ⏳ not started |
| 8 — Weeks 9–11 news channel ablation | ⏳ not started |
| 9 — Weeks 12–13 dashboard with per-alert reasons | ⏳ not started |
| 10 — Week 14 final evaluation, run once | ⏳ not started |
| 11 — Weeks 15–16 report and viva | ⏳ not started |

### Current state of the code

| File | Status |
|---|---|
| `src/db.py` | Rewritten: `companies`, `filings`, `events`, `bars`, `news`, `meta` |
| `src/utils/*` | Unchanged and reusable. **Missing: market-hours helpers (phase 1).** |
| `src/collectors/market.py` | Reusable; now universe-driven, with a snapshot stamp and a zero-record guard |
| `src/collectors/news.py` | Reusable; Finnhub is now primary, company names come from `companies` |
| `src/collectors/edgar.py` | **Does not exist yet — phase 2, the next file to write** |
| `src/pipeline/*` | Empty — `universe.py`, `t0.py`, `events.py`, `features.py`, `sampling.py` to come |
| `src/baselines/*`, `src/rl/*`, `src/eval/*` | Empty |

## Handoff checklist for a new agent session

1. Read `implementation_plan.md` in full, then this file.
2. Check the status table above and `git log --oneline` for current progress.
3. Verify the environment: `pytest` must pass before you build on top.
4. Confirm the current phase's acceptance criteria before declaring it done.
