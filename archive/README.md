# archive/ — the Reddit rumor-verification approach (abandoned)

This directory is **not part of the build**. It is kept deliberately, and it
belongs in the report: an explored-and-abandoned approach with working code is
evidence of work, not clutter. Nothing in `src/` imports from here, and
`pytest.ini` excludes it from collection.

## What this was

The original project ingested rumor posts about listed companies from Reddit
(r/wallstreetbets, r/stocks, …), clustered them into rumor "events", and
trained a sequential agent that at each hourly step decided WAIT or COMMIT
TRUE/FALSE. Ground truth came from the timestamp of the first confirming news
article. The headline metric was Time Delta Advantage — hours of advance
warning over official news.

The full original design is in
[`implementation_plan_reddit.md`](implementation_plan_reddit.md).

## Why it was abandoned

Two independent reasons, both external:

1. **The mentor rejected the Reddit data source** — five subreddits is a
   narrow, self-selecting slice of the market.
2. **Reddit shut down unauthenticated `.json` access on 30 May 2026.** This is
   permanent and enforced at infrastructure level (TLS fingerprinting, IP
   reputation), not a rate limit. No User-Agent, backoff, or host-rotation
   strategy recovers it, and the team's Reddit API appeal had already been
   rejected.

Point 2 matters for how the pivot is written up: it was forced by an external
platform policy change, not by an engineering failure.

## What is in here

| File | What it did |
|---|---|
| `src/collectors/arctic_shift.py` | Historical Reddit via the Arctic Shift API and `.zst` dumps |
| `src/collectors/reddit_live.py` | Live poller against public `.json` endpoints, with host rotation |
| `src/pipeline/tickers.py` | Ticker extraction from free text — cashtags, company names, a blacklist of English words that are also tickers |
| `src/pipeline/events.py` | Clustering posts into rumor events by ticker + time gap |
| `src/pipeline/labeling.py` | "Find the confirming article after t₀" labeling rule |
| `src/baselines/rule_based.py` | Non-AI WAIT/COMMIT threshold policy |
| `tests/` | The matching test suite, all passing at time of archival |
| `tickers_reddit_universe.csv` | Hand-curated 241-name ticker→company map |

## What carried forward into the current project

- `src/db.py` — the SQLite storage discipline (idempotent upserts, UTC epoch
  seconds everywhere) survived; the tables changed.
- `src/utils/` — rate limiting, backoff, UTC helpers, config loading: unchanged.
- `src/collectors/market.py` — yfinance OHLCV with incremental caching: the
  single most reusable file, essentially unchanged.
- `src/collectors/news.py` — promoted from an optional second channel to core
  label infrastructure, because t₀ correction depends on it.
- The **shape** of `rule_based.py` — a backward-looking, hour-by-hour threshold
  policy built before the learned model — survives as the volume z-score
  baseline, though the rumor-direction logic does not.

What died with Reddit: free-text ticker extraction, post clustering, social
dynamics features, text embeddings (MiniLM/FinBERT), and LLM rumor triage. The
current project's events come pre-labelled from SEC 8-K item codes, so none of
that annotation machinery is needed.
