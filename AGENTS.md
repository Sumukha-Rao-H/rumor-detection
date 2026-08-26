# AGENTS.md — Rules for AI agents and contributors

This repo implements **Pre-Announcement Footprints Around SEC Filings**
(Project Code 31, NMAMIT ISE). The authoritative design document is
[`implementation_plan.md`](implementation_plan.md). **Read it fully before
writing any code.** This file is the short operational contract.

Background reading, in order: `implementation_plan.md`, then
`docs/project-ideas-review.pdf` (Idea 1) and `docs/context.md`. Where the PDF
and `context.md` disagree, **`context.md` wins**.

## Local context folder (`context/`, gitignored)

A working memory for agent sessions lives in `context/`. It is **local only** —
it is gitignored and will not be in a fresh clone. It summarises and points
into this file and `implementation_plan.md`; **it never overrides them.**

| File | Read it when |
|---|---|
| `workflow-rules.md` | first — it says when to read the others |
| `project-overview.md` | every session start: goal, scope, out-of-scope |
| `progress-tracker.md` | every session start and end: live state |
| `task-breakdown.md` | when picking up work — 75 ordered tasks, and the file the user's **"next"** command reads |
| `architecture-context.md` | before writing, moving, or naming a file |
| `code-standards.md` | before writing code |
| `UI-context.md` | before touching `app/` or any output a human reads |

If `context/` is missing, this file plus `implementation_plan.md` are enough to
work from. To recreate it, ask.

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
- Never commit anything under `data/`, `.env`, or `context/`.

## Project status

**Live status is not kept in this file.** It lives in
`context/progress-tracker.md` — phases, per-file code status, decision log,
open questions, risks — so there is exactly one place to update and nothing can
drift out of sync.

> **Summary for anyone without `context/`:** Phase 0 (repointing to EDGAR) is
> done; Phase 1 (week-1 foundations) is the active phase. Phases and their
> acceptance criteria are in `implementation_plan.md` §9. `git log --oneline`
> is the other reliable record.

## Handoff checklist for a new agent session

1. Read `implementation_plan.md` in full, then this file.
2. If `context/` exists, read `context/workflow-rules.md` and
   `context/progress-tracker.md`. Otherwise use `git log --oneline` and
   `implementation_plan.md` §9 for current progress.
3. Verify the environment: `pytest` must pass before you build on top.
4. Confirm the current phase's acceptance criteria before declaring it done.
