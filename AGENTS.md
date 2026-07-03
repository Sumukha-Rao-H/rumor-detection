# AGENTS.md — Rules for AI agents and contributors

This repo implements **Cross-Domain Rumor Verification: Detecting Informed Trading
Footprints via Sequential Decision-Making** (Project Code 31, NMAMIT ISE).
The authoritative design document is [`implementation_plan.md`](implementation_plan.md).
**Read it fully before writing any code.** This file is the short operational contract.

## Non-negotiable rules

1. **Do not change the architecture** defined in the plan (§2) without explicit user
   approval. Frozen decisions: SQLite storage, precomputed frozen embeddings
   (MiniLM/FinBERT run once offline), tiny MLP RL policy via stable-baselines3,
   hourly decision steps, 3-action MDP (WAIT / COMMIT_TRUE / COMMIT_FALSE).
2. **Work phase by phase** (plan §5–§12). Do not start a phase until the previous
   phase's acceptance criteria pass.
3. **All timestamps in UTC everywhere.** Store epoch seconds (UTC) in SQLite; convert
   at display time only. Use `src/utils/timeutils.py` helpers — never naive datetimes.
4. **Never train on future data.** Every feature at time `t` must be computable using
   only data timestamped `≤ t`. If unsure, exclude it. Leakage tests in `tests/` are
   mandatory and must keep passing.
5. **Config-driven code only.** No hardcoded tickers, dates, paths, API URLs, or rate
   limits — everything comes from `config/config.yaml` (loaded via
   `src.utils.config.load_config`). Secrets come from `.env` (see `.env.example`).
6. **Respect external services.** No PRAW / Reddit OAuth (access was denied). Live
   Reddit uses public `.json` endpoints with a custom User-Agent and ≥7s between
   requests; Arctic Shift API ≤1 req/s; exponential backoff on 429/403; cache
   everything. Never scrape Reuters/Bloomberg directly.

## Repo conventions

- Layout is fixed by plan §3 (`src/collectors`, `src/pipeline`, `src/rl`,
  `src/baselines`, `src/eval`, `src/utils`, `app/`, `tests/`). New code goes in the
  module the plan assigns it to.
- Collectors are runnable as modules: `python -m src.collectors.<name> [args]`.
- SQLite schema lives in `src/db.py` (plan Appendix C). Writes are idempotent
  upserts — re-running any collector must never duplicate rows.
- Raw API payloads are archived under `data/raw/` (gitignored) before parsing.
- Python ≥3.10, standard library `sqlite3` (no ORM), type hints on public functions.

## Git workflow

- **Commit after every completed feature** — small, reviewable commits, one feature
  each, using conventional-commit prefixes (`feat:`, `chore:`, `fix:`, `test:`, `docs:`).
- **Never add an AI co-author line** (no `Co-Authored-By: Claude ...`) or any other
  AI attribution to commits or PRs.
- Never commit anything under `data/` (dumps, DBs, caches) or `.env`.

## Project status (update this section as phases complete)

| Phase | Status |
|---|---|
| 0 — Environment setup | ✅ repo layout, config, requirements |
| 1 — Data acquisition (§5) | ✅ collectors implemented: arctic_shift, reddit_live, market, news — **data collection runs pending** |
| 2 — Event construction & labeling (§6) | ⏳ not started (minimal `pipeline/tickers.py` pulled forward — collectors need it to filter posts) |
| 3 — Features & state (§7) | ⏳ not started |
| 4 — RL environment (§8) | ⏳ not started |
| 5 — Training (§9) | ⏳ not started |
| 6 — LLM policy baseline (§10) | ⏳ not started |
| 7 — Baselines & evaluation (§11) | ⏳ not started |
| 8 — Streamlit demo (§12) | ⏳ not started |

## Handoff checklist for a new agent session

1. Read `implementation_plan.md` in full, then this file.
2. Check the status table above and `git log --oneline` for current progress.
3. Verify the environment: `pytest tests/` must pass before you build on top.
4. Confirm the current phase's acceptance criteria (stated in each plan section)
   before declaring it done.
