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
| 1 — Data acquisition (§5) | ✅ Reddit collection complete (2026-08-04): 106.9k posts / 134.1k ticker links across all 577 days of the window, 5 subreddits. 3.7M hourly bars (1,566 tickers). News is a smoke test only — real collection is per-event in §6.4. |
| 2 — Event construction & labeling (§6) | 🟡 §6.1 extraction + §6.2 keyword pre-filter + §6.3 clustering done (`pipeline/events.py`): **4,770 unlabeled candidate events**, 866 tickers, 2025-01-01..2026-07-28. §6.2 LLM triage and §6.4 labeling not started |
| 3 — Features & state (§7) | ⏳ not started |
| 4 — RL environment (§8) | ⏳ not started |
| 5 — Training (§9) | ⏳ not started |
| 6 — LLM policy baseline (§10) | ⏳ not started |
| 7 — Baselines & evaluation (§11) | ⏳ not started |
| 8 — Streamlit demo (§12) | ⏳ not started |

### Ticker universe (regenerate, don't hand-edit) — 2026-08-04

`config/tickers.csv` (1,235 core symbols) and `config/listed_symbols.csv`
(12,497 US-listed symbols) are **generated**:

```
python -m src.pipeline.build_universe     # rebuild both CSVs from the corpus
python -m src.pipeline.relink             # re-apply them to posts already in SQLite
```

Hand-written names live in `config/tickers_curated.csv` (an input, never
rewritten). After any rebuild, run `relink` or the DB keeps stale links.
Matching is tiered — cashtags need a listed symbol, bare tokens need the core
universe, company names need `name_match=1` (see `pipeline/tickers.py`).

### Event set (regenerate, don't hand-edit) — 2026-08-04

```
python -m src.pipeline.events [--dry-run]   # keyword pre-filter + clustering
```

Idempotent: `event_id` is `{ticker}-{t0_utc}`, re-runs upsert, and events that
clustering no longer produces are pruned **unless** they carry a label or
`human_reviewed=1`. Labeling fields are never overwritten by a re-run — that
guarantee is what makes it safe to re-cluster after §6.2 triage.

Two deviations from a literal reading of §6.3, both measured (see git log):
`event.title_ticker_priority` seeds events only from tickers named in the post
title when there are any (a $BBAI post that name-drops PLTR was creating a PLTR
event: −19% events, all noise), and `event.exclude_etfs` drops index funds,
which have no company claim to confirm. Both are config-switchable.

### LLM triage (§6.2 stage 2) — 2026-08-04

```
python -m src.pipeline.triage --seeds-only    # judge each event's top post
python -m src.pipeline.triage --events-only   # rollup only, no API calls
```

Resumable and cached (`data/llm_cache/`, keyed by prompt version + model +
prompt content), so an interrupted run continues rather than restarts. Free
tier measured 2026-08-04: **gemini ~500 calls/day**, groq 70b 1,000/day,
groq 8b 14,400/day. A background loop re-runs the command every 30 min so it
picks up automatically when the daily quota resets:

```
setsid nohup bash -c 'while true; do python -m src.pipeline.triage --seeds-only \
  >> triage.log 2>&1; sleep 1800; done' &
```

**Triage is single-model on purpose** (`llm.fallback: null`). On identical
posts gemini-3.5-flash-lite called 36% of seeds rumors vs 10% for
llama-3.3-70b-versatile, which rejects real checkable claims; llama-3.1-8b's
disagreements were all false positives (68% agreement). Mixing providers would
make dataset *selection* depend on which model answered. Change the fallback
only for work where cross-event consistency does not matter (§10 baseline).

### Phase 1 notes (2026-08-04)

- **106.9k posts vs the §5.1 target of ≥200k.** The gap is not a collection
  failure: the target assumed unfiltered posts, while `posts` only stores rows
  that map to a ticker under the §6.1 rules. All 577 days of the window are
  covered for all five subreddits, so more Reddit collection would not help —
  only loosening the ticker filter would, at the cost of precision.
- `r_wallstreetbets_posts.jsonl` is a **truncated download** ending 2021-01-29,
  so no WSB post in the window comes from a dump. WSB was filled instead via
  the Arctic Shift API (`--mode api --subreddits wallstreetbets`, ~22 min for
  14 months). Re-downloading the dump would make re-runs faster but adds nothing.
- `backtest_window.end` was extended 2026-05-31 → 2026-07-29 to match what the
  dumps actually cover.
- **Live poller still blocked** (2026-07-03): unauthenticated reddit.com `.json`
  returns 403 from the dev network (www + old, any UA) — the plan's anticipated
  risk. Poller auto-rotates hosts; if it persists, use the Arctic Shift API with
  recent dates as the near-live fallback (§5.2 fallback c).
- `data/raw/arctic_dumps/*.crswap` (2.9 GB) are partial browser-download temp
  files, safe to delete. Comment dumps are scanned but never stored — the schema
  has no comments table.

## Handoff checklist for a new agent session

1. Read `implementation_plan.md` in full, then this file.
2. Check the status table above and `git log --oneline` for current progress.
3. Verify the environment: `pytest tests/` must pass before you build on top.
4. Confirm the current phase's acceptance criteria (stated in each plan section)
   before declaring it done.
