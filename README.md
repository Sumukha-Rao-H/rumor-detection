# Cross-Domain Rumor Verification — Detecting Informed Trading Footprints via Sequential Decision-Making

**Project Code 31 · NMAM Institute of Technology · Dept. of ISE**

The system ingests rumors about publicly traded companies from Reddit (r/wallstreetbets, r/stocks, …), fuses each rumor with the stock's recent price/volume behavior, and trains an RL agent that at each hourly step decides to **WAIT** (gather more market evidence) or **COMMIT** (declare the rumor TRUE or FALSE). Ground truth comes from timestamps of official news (GDELT/Finnhub). The headline metric is **Time Delta Advantage (Δ)** — how many hours before official news the agent correctly resolved the rumor.

Full design: [`implementation_plan.md`](implementation_plan.md). Agent/contributor rules: [`AGENTS.md`](AGENTS.md).

## Repository layout

```
├── config/config.yaml            # all knobs — no hardcoded tickers/dates/paths in code
├── data/
│   ├── raw/arctic_dumps/         # Arctic Shift .zst dumps (gitignored)
│   ├── raw/live_json/            # raw live-poller JSON snapshots (gitignored)
│   ├── db/rumor.db               # SQLite (posts, bars, news, events)
│   └── processed/                # events.parquet + per-event state tensors
├── src/
│   ├── collectors/               # arctic_shift, reddit_live, market, news
│   ├── pipeline/                 # tickers, events, labeling, features
│   ├── rl/                       # Gymnasium env + SB3 training
│   ├── baselines/                # static classifiers + LLM zero-shot policy
│   ├── eval/                     # Brier, ECE, Time Delta Advantage
│   └── utils/                    # rate limiting, UTC time helpers
├── app/dashboard.py              # Streamlit demo
├── tests/                        # pytest (leakage tests are mandatory)
└── implementation_plan.md        # the authoritative plan — read first
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your free-tier API keys (Finnhub, Gemini, Groq)
```

## Phase 1 — Data collection

All collectors are config-driven (`config/config.yaml`), write to SQLite at `data/db/rumor.db`, and store all timestamps in UTC.

**Historical Reddit (Arctic Shift):**

```bash
# Option A: parse downloaded .zst dumps from data/raw/arctic_dumps/
python -m src.collectors.arctic_shift --mode dumps

# Option B: pull from the Arctic Shift REST API (throttled ≤1 req/s)
python -m src.collectors.arctic_shift --mode api --start 2025-01-01 --end 2025-02-01
```

**Live Reddit poller** (public `.json` endpoints, no auth — run long-lived from Week 2):

```bash
python -m src.collectors.reddit_live           # loops forever, polls every 5 min
python -m src.collectors.reddit_live --once    # single cycle (testing)
```

**Market data** (yfinance hourly bars, cached incrementally):

```bash
python -m src.collectors.market --tickers TSLA,AAPL --start 2025-01-01 --end 2025-06-01
python -m src.collectors.market --from-db      # all tickers seen in collected posts
```

**Ground-truth news** (GDELT + Finnhub):

```bash
python -m src.collectors.news --ticker TSLA --query "merger OR acquisition" \
    --start 2025-01-01 --end 2025-01-08
```

## Tests

```bash
pytest tests/
```
