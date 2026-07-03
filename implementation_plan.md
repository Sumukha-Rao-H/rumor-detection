# Implementation Plan — Cross-Domain Rumor Verification: Detecting Informed Trading Footprints via Sequential Decision-Making

**Project Code 31 · NMAM Institute of Technology · Dept. of ISE**
**Team:** Sumukha Rao H, Swati Prabhu, Ujwal Hegde, Vachan J Poojary · **Guide:** Dr. Vaikunth Pai

---

## 0. How to Use This Document

This plan is written to be executable by a human team OR followed step-by-step by an LLM assistant (Claude Sonnet/Opus, GPT, Gemini). Rules for any LLM continuing this work:

1. **Do not change the architecture** defined in §2 without explicit user approval. All design decisions here were made deliberately under known constraints (§1.2).
2. **Work phase by phase.** Each phase (§5–§12) has: inputs, outputs, exact file paths, acceptance criteria. Do not start a phase until the previous phase's acceptance criteria pass.
3. **All timestamps in UTC everywhere.** Convert at display time only. This is the #1 source of bugs in this project class.
4. **Never train on future data.** Every feature at time `t` must be computable using only data with timestamp `≤ t`. If unsure, exclude it.
5. When writing code, follow the repository layout in §3 and the config-driven pattern in Appendix B. No hardcoded tickers, dates, or paths.

---

## 1. Project Summary & Constraints

### 1.1 One-paragraph summary

The system ingests rumors about publicly traded companies from Reddit (r/wallstreetbets, r/stocks, etc.), fuses each rumor with the stock's recent price/volume behavior, and trains a Reinforcement Learning agent that at each hourly step decides to **WAIT** (gather more market evidence) or **COMMIT** (declare the rumor TRUE or FALSE). Ground truth comes from timestamps of official news (GDELT/Finnhub). The headline metric is **Time Delta Advantage (Δ)** — how many hours before official news the agent correctly resolved the rumor — alongside calibration metrics (Brier score, ECE).

### 1.2 Hard constraints (why the plan looks the way it does)

| Constraint | Consequence in this plan |
|---|---|
| **No Reddit API access** (appeal rejected) | Historical data → **Arctic Shift** dumps/API. Live data → **public `.json` endpoints** with conservative rate limiting. No PRAW anywhere. |
| **RTX 3050 (4–8 GB VRAM) + free Colab** | Embeddings are **precomputed offline** with small models (MiniLM 22M params, FinBERT 110M). The RL policy is a tiny MLP (<1M params) that trains on CPU/3050 in minutes. No LLM fine-tuning. |
| **Free-tier APIs only** | LLM policy baseline uses **Gemini free tier** (fallback: Groq free tier). Market data via **yfinance** (free), not Alpha Vantage (25 req/day is unusable as primary). News via **GDELT** (free, no key) + **Finnhub free tier**. |
| **Academic timeline (~16 weeks)** | Scope: **binary veracity** (TRUE/FALSE) on **US large/mid-cap tickers**, hourly decision granularity, ~300–800 labeled events. |

### 1.3 Glossary

- **Rumor event:** a cluster of Reddit posts about one ticker in one time window containing an unverified factual claim (merger, earnings leak, bankruptcy, FDA approval, partnership, etc.).
- **t₀:** timestamp of the first Reddit post in the event.
- **t_official:** timestamp of the first credible news article confirming or denying the claim.
- **t_commit:** timestamp at which the agent issues its verdict.
- **Time Delta Advantage:** Δ = t_official − t_commit (positive = agent beat the news).
- **Episode:** one rumor event, unrolled as a sequence of hourly steps from t₀ until COMMIT or timeout.

---

## 2. System Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              DATA ACQUISITION               │
                        │                                             │
  Historical Reddit ──► │ Arctic Shift dumps/API  ─┐                  │
  Live Reddit ────────► │ reddit.com/*.json poller ─┼─► SQLite: posts │
  Market data ────────► │ yfinance (OHLCV)         ─┼─► SQLite: bars  │
  Official news ──────► │ GDELT + Finnhub          ─┴─► SQLite: news  │
                        └───────────────┬─────────────────────────────┘
                                        │
                        ┌───────────────▼─────────────────────────────┐
                        │        EVENT CONSTRUCTION & LABELING        │
                        │  ticker extraction → event clustering →     │
                        │  ground-truth labeling (news within 72h)    │
                        │  output: events.parquet (one row per event) │
                        └───────────────┬─────────────────────────────┘
                                        │
                        ┌───────────────▼─────────────────────────────┐
                        │        FEATURE / STATE PRECOMPUTATION       │
                        │  MiniLM text embedding (384-d, frozen)      │
                        │  FinBERT sentiment · social stats ·         │
                        │  market stats per hourly step               │
                        │  output: states/{event_id}.npz             │
                        └───────────────┬─────────────────────────────┘
                                        │
              ┌─────────────────────────┼──────────────────────────┐
              ▼                         ▼                          ▼
   ┌──────────────────┐   ┌─────────────────────────┐  ┌────────────────────┐
   │  STATIC BASELINES│   │   RL AGENT (core)       │  │ LLM POLICY BASELINE│
   │  LogReg, XGBoost,│   │ Gymnasium env + PPO/DQN │  │ Gemini free tier,  │
   │  fixed-time BERT │   │ (stable-baselines3)     │  │ zero-shot WAIT/    │
   │  classifier      │   │ actions: WAIT/COMMIT_T/ │  │ COMMIT prompting   │
   └────────┬─────────┘   │ COMMIT_F                │  └─────────┬──────────┘
            │             └───────────┬─────────────┘            │
            └─────────────────────────┼──────────────────────────┘
                                      ▼
                        ┌─────────────────────────────┐
                        │         EVALUATION          │
                        │ Accuracy/F1 · Brier · ECE · │
                        │ Time Delta Advantage ·      │
                        │ reliability diagrams        │
                        └──────────────┬──────────────┘
                                       ▼
                        ┌─────────────────────────────┐
                        │   STREAMLIT DEMO DASHBOARD  │
                        │  live poller + agent replay │
                        └─────────────────────────────┘
```

Key design decision: **the heavy neural nets (MiniLM, FinBERT) are frozen feature extractors run once offline.** The RL agent only ever sees a fixed-size numeric vector. This makes training feasible on a 3050 (or pure CPU), makes experiments fast and reproducible, and cleanly separates NLP from RL.

---

## 3. Repository Structure

```
rumor-rl/
├── config/
│   └── config.yaml               # all knobs (Appendix B)
├── data/
│   ├── raw/
│   │   ├── arctic_dumps/         # .zst files (gitignored)
│   │   └── live_json/            # raw JSON snapshots (gitignored)
│   ├── db/
│   │   └── rumor.db              # SQLite (schema: Appendix C)
│   └── processed/
│       ├── events.parquet
│       └── states/               # {event_id}.npz precomputed state tensors
├── src/
│   ├── collectors/
│   │   ├── arctic_shift.py       # historical Reddit
│   │   ├── reddit_live.py        # .json endpoint poller
│   │   ├── market.py             # yfinance OHLCV
│   │   └── news.py               # GDELT + Finnhub ground truth
│   ├── pipeline/
│   │   ├── tickers.py            # cashtag/name extraction + blacklist
│   │   ├── events.py             # clustering posts into events
│   │   ├── labeling.py           # ground-truth resolution rules
│   │   └── features.py           # embeddings + numeric features → .npz
│   ├── rl/
│   │   ├── env.py                # RumorVerificationEnv (Gymnasium)
│   │   ├── train.py              # SB3 PPO/DQN training
│   │   └── callbacks.py          # eval callbacks, logging
│   ├── baselines/
│   │   ├── static_clf.py         # LogReg / XGBoost at fixed horizons
│   │   └── llm_policy.py         # Gemini/Groq zero-shot policy
│   ├── eval/
│   │   ├── metrics.py            # Brier, ECE, Time Delta
│   │   └── report.py             # tables + reliability diagrams
│   └── utils/
│       ├── ratelimit.py
│       └── timeutils.py          # UTC helpers, market-hours calendar
├── app/
│   └── dashboard.py              # Streamlit demo
├── notebooks/                    # Colab notebooks (training runs)
├── tests/                        # pytest — leakage tests are mandatory
├── requirements.txt              # Appendix A
└── README.md
```

---

## 4. Phase 0 — Environment Setup

**Local (RTX 3050 machine):**

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# PyTorch with CUDA for the 3050 (check https://pytorch.org for current command):
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

**Colab (free):** used only for (a) FinBERT embedding of large post batches if the 3050 has 4 GB VRAM, and (b) longer PPO runs (>1M steps). Mount Google Drive, sync `data/processed/` via Drive. Keep every Colab notebook idempotent — free Colab disconnects; checkpoint every 50k steps.

**Division of labor rule:** everything except long training runs happens locally. Colab is a compute overflow valve, not the primary environment.

**Acceptance criteria:** `pytest tests/test_setup.py` passes (imports torch, gymnasium, stable_baselines3, sentence_transformers, yfinance, checks CUDA optional).

---

## 5. Phase 1 — Data Acquisition

### 5.1 Historical Reddit (Arctic Shift) — replaces PRAW entirely

Arctic Shift (github.com/ArthurHeitmann/arctic_shift) is the maintained successor to Pushshift: monthly full-Reddit dumps via Academic Torrents, per-subreddit downloads, a free REST API, and a Hugging Face Parquet mirror queryable with DuckDB. As of mid-2026 it is current through recent months.

**Recommended path (simplest first):**

1. **Per-subreddit download tool:** `https://arctic-shift.photon-reddit.com/download-tool` — download `r/wallstreetbets`, `r/stocks`, `r/StockMarket`, `r/pennystocks`, `r/investing` posts (+ optionally comments) for your chosen backtest window (recommend **12–18 recent months**). Large subs take hours; narrow the date range if needed.
2. **Fallback for programmatic access:** Arctic Shift REST API, e.g.
   `GET https://arctic-shift.photon-reddit.com/api/posts/search?subreddit=wallstreetbets&after=2025-01-01&before=2025-02-01&limit=100`
   Paginate by advancing `after` to the last item's `created_utc`. It's a free community service — throttle to ≤1 req/sec and cache everything.
3. **Fallback #2 (big-data option):** DuckDB over the Arctic Shift Hugging Face Parquet mirror — SQL queries over the archive without downloading dumps. Use only if 1 and 2 fail; document the exact dataset path in config.

**Processing `.zst` dumps** (files are newline-delimited JSON inside zstandard compression — stream, never fully decompress):

```python
# src/collectors/arctic_shift.py — core loop specification
import zstandard as zstd, json, io

def stream_zst(path):
    with open(path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        stream = io.TextIOWrapper(dctx.stream_reader(fh), encoding='utf-8', errors='ignore')
        for line in stream:
            yield json.loads(line)

# For each record keep: id, subreddit, title, selftext, author,
# created_utc, score, upvote_ratio, num_comments, link_flair_text, url
# Filter: pass record through tickers.extract() (§6.1); insert matches into SQLite `posts`.
```

**Acceptance criteria:** ≥200k posts in `posts` table spanning the backtest window; spot-check 20 random rows against reddit.com permalinks.

### 5.2 Live Reddit — public `.json` endpoints (no auth)

Any Reddit listing URL returns JSON when suffixed with `.json`:

| Purpose | Endpoint |
|---|---|
| Newest posts | `https://www.reddit.com/r/wallstreetbets/new.json?limit=100` |
| Rising | `https://www.reddit.com/r/wallstreetbets/rising.json?limit=50` |
| Multi-sub in one call | `https://www.reddit.com/r/wallstreetbets+stocks+StockMarket/new.json?limit=100` |
| Single post + comments | `https://www.reddit.com/comments/{post_id}.json` |
| Pagination | append `&after=t3_{last_id}` |

**Non-negotiable rules for the poller (`src/collectors/reddit_live.py`):**

1. **Custom User-Agent header**, e.g. `nmamit-major-project-31:v1.0 (academic research)`. Default python-requests UA gets blocked instantly.
2. **Conservative rate limit: 1 request per 7–10 seconds** (unauthenticated access is throttled far below the old 100/10-min; assume ~10 req/min ceiling and stay well under it). Implement a token-bucket in `utils/ratelimit.py`.
3. **Exponential backoff on HTTP 429/403:** sleep 60s, 120s, 240s…; log and continue. Never hammer.
4. Poll cycle: every 5 minutes, hit the multi-sub `new.json`, dedupe by post id against SQLite, insert new rows, and store the raw JSON snapshot in `data/raw/live_json/` (you will thank yourself later).
5. Run as a long-lived script (`while True` + sleep) on the local machine, or via Windows Task Scheduler/cron. Target **4+ weeks of continuous live collection** running in the background while you build everything else — start this in Week 2.
6. Re-fetch each interesting post once at t₀+6h and t₀+24h to capture score/comment growth (these are features).

**Risk note & fallback:** unauthenticated JSON endpoints occasionally get tightened. If blocked: (a) rotate to `old.reddit.com/...json`, (b) increase interval to 60s, (c) fall back to the Arctic Shift API for near-real-time data (it ingests continuously). The project's evaluation is backtest-first, so live collection failing is degraded-demo, not project-fatal.

**Acceptance criteria:** poller runs 72h unattended without crashing; no 429s in steady state; new posts appear in SQLite within 10 minutes of appearing on Reddit.

### 5.3 Market data — yfinance (primary)

```python
import yfinance as yf
df = yf.download("TSLA", start=..., end=..., interval="60m", auto_adjust=True)
```

**Critical granularity limits (design around these):**

- `1m` bars: only ~last 30 days, 8 days per request → useless for backtests. **Do not build on 1m data.**
- `5m/15m/30m` bars: only ~last 60 days.
- `60m` bars: ~last 730 days ✅ — **this is your backtest granularity** and it matches the hourly decision step.
- `1d` bars: unlimited history.

**Plan:** hourly (`60m`) bars for all tickers in the event set, cached into SQLite `bars` (one fetch per ticker, incremental updates). Daily bars as supplementary context. Alpha Vantage (25 req/day free) kept only as an outage fallback, and `stooq`/`pandas-datareader` as a second fallback for daily data.

**Off-hours handling:** Reddit rumors often land nights/weekends when no bars exist. Rule: a decision step at time t uses the most recent completed bar at or before t; additionally include a binary `market_open` feature and `hours_since_last_bar`. Use the `pandas_market_calendars` NYSE calendar.

**Acceptance criteria:** for every event ticker, hourly bars cover [t₀ − 7d, t₀ + 5d] with <2% missing regular-session hours.

### 5.4 Ground-truth news — GDELT + Finnhub (do NOT scrape Reuters/Bloomberg)

Direct scraping of Reuters/Bloomberg violates their ToS and gets IP-blocked; the synopsis's intent (official timestamps) is satisfied better by news aggregation APIs:

1. **GDELT 2.0 DOC API** — free, no key, 15-min update latency, full-text search over global news:
   `https://api.gdeltproject.org/api/v2/doc/doc?query="Tesla" (merger OR acquisition)&mode=artlist&maxrecords=75&format=json&startdatetime=20250101000000&enddatetime=20250108000000`
   Returns article URL, title, source domain, and `seendate` (UTC) → this is `t_official`.
2. **Finnhub free tier** (`/company-news?symbol=TSLA&from=...&to=...`) — clean per-ticker headlines with UNIX timestamps; free key, generous limits (~60 calls/min).
3. **yfinance `Ticker.news`** — recent headlines only; use for the live demo path.

**Source credibility whitelist** (config-driven): reuters.com, bloomberg.com, apnews.com, cnbc.com, wsj.com, ft.com, prnewswire.com, businesswire.com, sec.gov, company IR domains. An article counts as "official" only if from the whitelist. (You never fetch full article bodies from paywalled sites — headline + timestamp from the aggregator is sufficient for labeling.)

**Acceptance criteria:** for a hand-picked set of 10 known events (e.g., a real merger announcement), the pipeline retrieves the correct announcement article within ±1h of its true publication time.

---

## 6. Phase 2 — Event Construction & Labeling

This phase turns raw posts into the dataset the RL agent trains on. **It is the highest-effort, highest-value phase. Budget 3 weeks.**

### 6.1 Ticker extraction (`src/pipeline/tickers.py`)

1. Regex cashtags: `\$([A-Z]{1,5})\b`.
2. Bare uppercase tokens `\b[A-Z]{2,5}\b` matched against a ticker universe (Russell 1000 + top-500-by-Reddit-mention list, stored as CSV).
3. **Mandatory blacklist** — common words that are also tickers destroy precision: `A, ALL, AI, ARE, BE, BIG, BUY, CAN, CEO, DD, EDIT, EV, FOR, GO, IPO, IT, LOL, LOVE, NOW, ON, ONE, OR, OUT, PUMP, RH, SO, TV, U, UK, USA, WSB, YOLO, IMO, ATH, FOMO, FD, PT, EPS, ER` (extend empirically — after first extraction run, review the top-100 extracted tickers by frequency and prune junk).
4. Company-name matching (e.g. "Tesla" → TSLA) via a name→ticker dictionary; require exact case-insensitive whole-word match.
5. A post maps to a ticker only if: cashtag match, OR (dictionary match AND ticker in universe). Posts with >3 distinct tickers are discarded (portfolio-spam).

### 6.2 Rumor filtering

Not every post is a rumor. A post qualifies as a **rumor candidate** if it contains a *forward-looking or unverified factual claim*. Two-stage filter:

1. **Keyword pre-filter** (cheap, high recall): title/selftext matches any of: `merger, acquisition, acquire, buyout, takeover, bankrupt, chapter 11, delist, halt, FDA, approval, recall, partnership, contract, earnings leak, insider, SEC, investigation, lawsuit, guidance, short squeeze, offering, dilution, split`.
2. **LLM triage** (Gemini free tier, batched): for each pre-filtered post, one call → JSON `{is_rumor: bool, claim_summary: str, claim_type: enum}`. Cache every response keyed by post id (Appendix D has the prompt). Budget: a few thousand calls spread over days fits free-tier daily quotas.

### 6.3 Event clustering

Group rumor candidates: same ticker, posts within a rolling 12-hour gap window are the same event (i.e., an event closes after 12h of silence on that ticker's claim type). Minimum event size: 2 posts OR 1 post with score ≥ 50. Each event row stores: `event_id, ticker, t0, claim_summary, claim_type, post_ids[], n_posts, subreddits[]`.

### 6.4 Ground-truth labeling (the reward oracle)

For each event, query GDELT + Finnhub in window [t₀ − 24h, t₀ + 72h] for the ticker/company + claim keywords. Then:

- **TRUE:** whitelist article within 72h whose headline confirms the claim → store `t_official`, label 1.
- **FALSE:** whitelist article denies it, OR no confirming article within 72h AND no abnormal sustained price move (|3-day return| < 4% and volume z-score < 2) → label 0, `t_official` = denial time or t₀+72h.
- **UNVERIFIED/AMBIGUOUS:** everything else → **excluded from train/eval** (keep in a side table; report count in the paper).

Labeling is semi-automated: the LLM triage call (Gemini) compares `claim_summary` vs retrieved headlines and proposes a label; **a human must review every event label** in a simple CSV/Streamlit review UI. With ~500 events this is 2 person-days. Do not skip human review — label noise here poisons everything downstream.

**Dataset targets & split:**
- ≥ 300 labeled events (aim 500–800), class balance forced to ≥30% minority via targeted mining of FALSE rumors (they're rarer to confirm).
- **Time-based split only:** train = oldest 70%, val = next 15%, test = newest 15%. Never random split (leakage via market regimes and repeated tickers). Also enforce: no ticker's events straddle the val/test boundary within ±7 days.

**Acceptance criteria:** `events.parquet` with ≥300 rows, human-reviewed labels, class balance documented, leakage test passing (`tests/test_leakage.py`: asserts max(train.t0) < min(val.t0), etc.).

---

## 7. Phase 3 — Features & State Representation

For each event, precompute a per-step state matrix and save as `data/processed/states/{event_id}.npz` containing `X ∈ ℝ^(T×D)` where T = number of hourly steps (max 48) and D as below. The RL env just indexes this array — training then requires zero NLP compute.

### 7.1 Text block (static per event, repeated each step) — 388 dims

| Feature | Dim | Source |
|---|---|---|
| MiniLM embedding of `title + claim_summary` of the seed post | 384 | `sentence-transformers/all-MiniLM-L6-v2` (runs fine on 3050/CPU) |
| FinBERT sentiment (pos, neg, neutral probs) | 3 | `ProsusAI/finbert` |
| claim_type one-hot collapsed to learned index / 8 buckets → use 1 scaled scalar or 8-dim one-hot (choose 8-dim; total becomes 395) | 8 | LLM triage |

### 7.2 Social dynamics block (updates each step) — 8 dims

Post velocity (posts/hr in trailing 6h, log1p), cumulative unique authors (log1p), mean upvote_ratio, cumulative score (log1p), comments/hr, share of posts from "young" accounts if available (else 0), max post score so far (log1p), hours since t₀ / 48.

### 7.3 Market block (updates each step) — 14 dims

Using hourly bars up to current step t: last 6 hourly log-returns (6), volume z-score vs trailing 20-day same-hour mean (1), realized volatility 24h vs 20-day baseline ratio (1), cumulative return since t₀ (1), gap vs previous close (1), `market_open` flag (1), hours_since_last_bar / 24 (1), daily return of SPY same window — market control (1), bid-ask proxy = (high−low)/close last bar (1).

### 7.4 Final state vector

`D = 395 (text) + 8 (social) + 14 (market) + 1 (step fraction t/T) = 418`. Standardize the non-embedding numeric features with a scaler **fit on train events only** (persist with joblib).

**Compute note:** embedding ~800 events × ~1 post each = trivial. FinBERT over all posts: batch size 16, fp16 — minutes on the 3050; if 4 GB VRAM OOMs, run on Colab once and sync `.npz` via Drive.

**Acceptance criteria:** every event has an `.npz`; `X` contains no NaNs; a script confirms feature at step t never uses bars/posts timestamped > t₀ + t hours.

---

## 8. Phase 4 — The RL Environment (`src/rl/env.py`)

Formalized as an MDP over precomputed states (a "batch/offline-replay" environment — standard for early-detection RL, cf. reference [5] in the synopsis).

```python
import gymnasium as gym
import numpy as np

class RumorVerificationEnv(gym.Env):
    """One episode = one rumor event. Hourly steps."""
    WAIT, COMMIT_TRUE, COMMIT_FALSE = 0, 1, 2

    def __init__(self, event_store, split="train", cfg=None):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(418,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.cfg = cfg  # rewards + T_max from config.yaml
        self.store = event_store; self.split = split

    def reset(self, *, seed=None, options=None):
        self.event = self.store.sample(self.split)     # random event (train); sequential (eval)
        self.t = 0
        return self.event.X[0], {"event_id": self.event.id}

    def step(self, action):
        c = self.cfg
        if action == self.WAIT:
            self.t += 1
            truncated = self.t >= min(self.event.T, c.T_max)   # T_max = 48
            if truncated:
                # forced abstention at timeout counts as wrong-but-mild
                return self.event.X[self.t-1], c.r_timeout, False, True, self._info()
            return self.event.X[self.t], c.r_wait, False, False, self._info()
        pred = 1 if action == self.COMMIT_TRUE else 0
        correct = (pred == self.event.label)
        if correct:
            earliness = 1.0 - self.t / c.T_max
            reward = c.r_correct + c.r_early_bonus * earliness
        else:
            reward = c.r_wrong
        return self.event.X[self.t], reward, True, False, self._info(pred=pred)
```

**Reward config (starting values — tune in §9):**

```yaml
reward:
  r_correct:      +1.0
  r_early_bonus:  +0.5     # scaled by (1 - t/T_max): correct at t=0 → 1.5, at t=T → 1.0
  r_wrong:        -2.0     # asymmetric: wrong is worse than slow (synopsis: "penalize speculative guessing")
  r_wait:         -0.02    # per-step time cost
  r_timeout:      -0.5
  T_max:          48       # hours
```

**Design rationale to preserve:** (a) 3 actions, not 2 — "Commit" must carry a verdict; (b) the wrong/correct asymmetry plus wait cost is exactly the trade-off the synopsis promises to study; (c) the early bonus makes Time Delta an optimized quantity, not a side effect. Ablate `r_wait ∈ {0.005, 0.02, 0.05}` and report the earliness/accuracy frontier — that's a strong results figure.

**Acceptance criteria:** env passes `gymnasium.utils.env_checker.check_env`; random policy runs 1000 episodes without error; reward accounting unit-tested.

---

## 9. Phase 5 — Training the Agent

**Library:** `stable-baselines3`. **Algorithms:** train both **PPO** (primary) and **DQN** (comparison). Policy network: MLP `[418 → 256 → 128 → 3]` (~140k params — trains on CPU; the 3050 is overkill).

**Starting hyperparameters:**

| | PPO | DQN |
|---|---|---|
| total_timesteps | 1,000,000 | 500,000 |
| learning_rate | 3e-4 | 1e-4 |
| gamma | 0.99 | 0.99 |
| n_steps / buffer | 2048 | 100,000 |
| batch_size | 256 | 128 |
| ent_coef / ε-schedule | 0.01 | 1.0→0.05 over 20% |
| n_envs (VecEnv) | 8 | 1 |

**Protocol:**
1. Sanity check: overfit 20 events → agent should reach ~100% on them. If not, there's a bug, stop and fix.
2. Full training, EvalCallback on the val split every 20k steps, keep best-by-val-reward checkpoint.
3. 5 seeds per configuration; report mean ± std (examiners like this; it's also honest).
4. **Probability outputs for calibration metrics:** at COMMIT time, record PPO's softmax over {COMMIT_TRUE, COMMIT_FALSE} renormalized as p(TRUE). Additionally train a small supervised "confidence head" (logistic regression on the state at t_commit) as a calibrated fallback; report both, apply temperature scaling fit on val.
5. Track experiments with `tensorboard` (free, local).

**Colab usage:** only if local wall-clock exceeds ~2h per run; the notebook loads `states/` from Drive, checkpoints to Drive every 50k steps, and can resume after disconnects.

**Acceptance criteria:** PPO beats random policy and beats "always commit at t=0" on val reward; training curves saved.

---

## 10. Phase 6 — LLM-Based Policy Baseline (free tier)

The synopsis mentions an "LLM-based policy." Under free-tier constraints this is a **zero-shot baseline**, not the core engine:

- **Provider:** Gemini free tier (`google-generativeai`, model `gemini-*-flash` — pick the current flash model; free tier gives on the order of 10–15 requests/min and a daily cap, verify current limits at ai.google.dev before running). **Fallback:** Groq free tier (Llama-3.x-70B) — same prompt.
- At each step of a *test* episode, render the state as text (claim summary + social stats + recent returns table) and ask for JSON `{"action": "WAIT|COMMIT_TRUE|COMMIT_FALSE", "p_true": 0.0-1.0, "reason": "..."}` (prompt in Appendix D).
- **Budget control:** run it only on the test split (~50–100 events × avg ~10 steps ≈ 500–1000 calls), cache aggressively, add 6s sleeps. Feasible on free tier over 1–2 days.
- This yields a genuinely interesting comparison for the report: tiny trained RL policy vs large zero-shot LLM on the same MDP.

---

## 11. Phase 7 — Baselines & Evaluation

### 11.1 Static baselines (`src/baselines/static_clf.py`)

1. **Majority class** and **always-commit-at-t₀ logistic regression** (text features only) — the "static classification" strawman from the synopsis.
2. **XGBoost at fixed horizons** t = 6h, 12h, 24h using the full state vector — the serious baseline. Its 24h accuracy vs the agent's accuracy-at-average-commit-time is the key comparison.

### 11.2 Metrics (`src/eval/metrics.py`)

- **Accuracy / F1** on committed verdicts + **abstention rate** (timeouts).
- **Brier score:** `np.mean((p_true - y)**2)` over committed events.
- **ECE (15 equal-width bins):** `sum(n_b/N * |acc_b - conf_b|)`; also plot the reliability diagram (`sklearn.calibration.calibration_curve`). Report pre/post temperature scaling.
- **Time Delta Advantage:** per correctly-resolved event, `Δ = t_official − t_commit` in hours. Report median and distribution (box plot), plus **% of events with Δ ≥ 24h** (the synopsis's stated target). Compare agent Δ vs fixed-horizon baselines with a **Wilcoxon signed-rank test**.
- **Earliness–accuracy frontier:** sweep `r_wait`, plot accuracy vs mean commit time for each trained agent; overlay the fixed-horizon XGBoost points.

### 11.3 Honest-reporting rules (put these in the report's threats-to-validity section)

- Δ is measured against *aggregator-visible* news timestamps (GDELT seendate), which lag true wire time by minutes — state this.
- Excluded UNVERIFIED events are reported as a count and discussed.
- No trading-strategy claims; this is a verification/monitoring tool (matches the synopsis's "market integrity" framing and avoids examiner pushback).

---

## 12. Phase 8 — Live Demo (Streamlit)

`app/dashboard.py`: (1) live poller feed of new candidate posts, (2) event view: price chart with post markers + agent action timeline (WAIT…WAIT…COMMIT) replayed step-by-step, (3) metrics page with reliability diagram and Δ histogram, (4) "run agent on this live event" button using the frozen checkpoint. Streamlit + plotly, all local. This is your project-demo centerpiece; keep it simple and robust.

---

## 13. Risk Register & Fallbacks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Reddit tightens `.json` endpoints | Medium | Arctic Shift API for near-live data; backtest-first evaluation means the thesis survives |
| Too few clean labeled events | Medium | Widen backtest window to 24 months; add r/pennystocks (rumor-dense); lower min event size with human review |
| Label noise | High | Mandatory human review; report inter-annotator agreement on a 50-event double-labeled subset |
| Gemini free-tier quota cuts | Medium | Groq fallback; triage prompts are provider-agnostic; caching means no repeated spend |
| yfinance breaks (it occasionally does) | Low-Med | Pin version; SQLite cache means historical bars are fetched once; Stooq/Alpha Vantage daily fallback |
| RL doesn't beat baselines | Medium | The earliness–accuracy frontier is a result either way; also report DQN, reward ablations — negative results with analysis still make a strong report |
| 4 GB VRAM OOM on FinBERT | Low | fp16, batch 8, or one-time Colab run |

---

## 14. Timeline (16 weeks)

| Weeks | Deliverable |
|---|---|
| 1 | Repo, env setup, config skeleton, this plan reviewed with guide |
| 2–3 | Arctic Shift download + parse; **live poller deployed and running from Week 2**; market + news collectors |
| 4–6 | Ticker extraction, rumor triage, event clustering, labeling pipeline + human review UI → `events.parquet` v1 |
| 7 | Feature pipeline → `states/*.npz`; leakage tests |
| 8–9 | Gym env + PPO/DQN training; overfit sanity check; first full runs |
| 10 | Static baselines; metrics module |
| 11 | LLM policy baseline on test split; calibration (temperature scaling) |
| 12 | Ablations (reward sweep, 5 seeds), final evaluation tables/figures |
| 13 | Streamlit dashboard |
| 14–15 | Report writing, threats-to-validity, demo rehearsal |
| 16 | Buffer + submission |

---

## Appendix A — requirements.txt

```
requests
zstandard
duckdb
pandas
pyarrow
numpy
yfinance
pandas-market-calendars
finnhub-python
sentence-transformers
transformers
torch                # install with CUDA index-url separately
gymnasium
stable-baselines3
scikit-learn
xgboost
google-generativeai  # Gemini free tier
groq                 # fallback LLM
streamlit
plotly
tensorboard
pytest
joblib
tqdm
python-dotenv
```

## Appendix B — config.yaml (skeleton)

```yaml
subreddits: [wallstreetbets, stocks, StockMarket, investing, pennystocks]
backtest_window: {start: "2025-01-01", end: "2026-05-31"}
live_poll_seconds: 300
reddit_user_agent: "nmamit-major-project-31:v1.0 (academic research)"
ticker_universe_csv: "config/tickers.csv"
ticker_blacklist: [A, ALL, AI, ARE, BE, BIG, BUY, CAN, CEO, DD, EDIT, EV, FOR, GO, IPO, IT, LOL, NOW, ON, ONE, OR, OUT, PUMP, SO, TV, U, UK, USA, WSB, YOLO, IMO, ATH, FOMO, FD, PT, EPS, ER]
news_whitelist: [reuters.com, bloomberg.com, apnews.com, cnbc.com, wsj.com, ft.com, prnewswire.com, businesswire.com, sec.gov]
event: {gap_hours: 12, min_posts: 2, min_solo_score: 50, label_horizon_hours: 72}
state: {embed_model: "sentence-transformers/all-MiniLM-L6-v2", sent_model: "ProsusAI/finbert", dim: 418}
reward: {r_correct: 1.0, r_early_bonus: 0.5, r_wrong: -2.0, r_wait: -0.02, r_timeout: -0.5, T_max: 48}
split: {train: 0.70, val: 0.15, test: 0.15, method: temporal}
llm: {provider: gemini, fallback: groq, cache_dir: "data/llm_cache", min_interval_s: 6}
```

## Appendix C — SQLite schema (data/db/rumor.db)

```sql
CREATE TABLE posts (
  id TEXT PRIMARY KEY, subreddit TEXT, title TEXT, selftext TEXT,
  author TEXT, created_utc INTEGER, score INTEGER, upvote_ratio REAL,
  num_comments INTEGER, flair TEXT, url TEXT, source TEXT,  -- 'arctic'|'live'
  fetched_utc INTEGER, score_6h INTEGER, score_24h INTEGER
);
CREATE TABLE bars (
  ticker TEXT, ts_utc INTEGER, open REAL, high REAL, low REAL,
  close REAL, volume REAL, interval TEXT, PRIMARY KEY (ticker, ts_utc, interval)
);
CREATE TABLE news (
  url TEXT PRIMARY KEY, ticker TEXT, title TEXT, source_domain TEXT,
  seen_utc INTEGER, api TEXT  -- 'gdelt'|'finnhub'|'yf'
);
CREATE TABLE events (
  event_id TEXT PRIMARY KEY, ticker TEXT, t0_utc INTEGER, claim_summary TEXT,
  claim_type TEXT, post_ids TEXT, label INTEGER,  -- 1 TRUE, 0 FALSE, NULL unverified
  t_official_utc INTEGER, label_source TEXT, human_reviewed INTEGER DEFAULT 0
);
```

## Appendix D — LLM prompts

**D.1 Rumor triage (per post, temperature 0, JSON mode):**

```
You are labeling Reddit posts about stocks for a research dataset.
A "rumor" is an unverified factual claim about a specific company that could later
be confirmed or denied by official news (merger, acquisition, bankruptcy, FDA decision,
earnings leak, major contract, investigation, delisting, offering).
NOT rumors: opinions, price predictions, memes, questions, technical analysis, general DD
without a specific checkable claim.

POST TITLE: {title}
POST BODY: {selftext[:1500]}
TICKER: {ticker}

Respond ONLY with JSON:
{"is_rumor": true/false,
 "claim_summary": "one sentence stating the checkable claim, or empty",
 "claim_type": "merger|bankruptcy|regulatory|earnings|contract|legal|offering|other"}
```

**D.2 Label proposal (per event):**

```
CLAIM (from Reddit, posted {t0}): {claim_summary}
CANDIDATE NEWS HEADLINES (source, UTC time):
{headline_list}

Did credible news CONFIRM or DENY the claim within 72 hours?
Respond ONLY with JSON:
{"verdict": "TRUE|FALSE|UNVERIFIED", "t_official": "<UTC of earliest deciding headline or null>",
 "deciding_headline": "<text or null>", "confidence": 0.0-1.0}
```

**D.3 Zero-shot policy (per step, test split only):**

```
You are a sequential rumor-verification agent. At each hourly step choose one action.
Committing early with a wrong verdict is heavily penalized; waiting has a small cost.

RUMOR: {claim_summary} (ticker {ticker}, first posted {hours_elapsed}h ago)
SOCIAL: {n_posts} posts, {posts_per_hr}/hr velocity, {unique_authors} authors, avg upvote ratio {ur}
MARKET (hourly): returns last 6h: {returns}; volume z-score: {vz}; cum. return since rumor: {cr}; market open: {mo}
STEP: {t}/48

Respond ONLY with JSON:
{"action": "WAIT|COMMIT_TRUE|COMMIT_FALSE", "p_true": 0.0-1.0, "reason": "<15 words"}
```

## Appendix E — Handoff block (paste this when switching to another LLM)

```
You are continuing a college major project. Read implementation_plan.md fully first.
Current phase: <FILL IN>. Completed: <FILL IN>. Blockers: <FILL IN>.
Rules: follow the plan's architecture and repo layout exactly; UTC everywhere;
no data with timestamp > current step in any feature; config-driven code only;
each phase must pass its stated acceptance criteria before moving on.
Task for you: <FILL IN>.
```
