# Implementation Plan — Pre-Announcement Footprints Around SEC Filings

**Project Code 31 · NMAM Institute of Technology · Dept. of ISE**

This is the authoritative plan. It supersedes the Reddit-based plan, which is
kept at [`archive/implementation_plan_reddit.md`](archive/implementation_plan_reddit.md).

Source material: `docs/project-ideas-review.pdf` (Idea 1) and `docs/context.md`.
**Where those two disagree, `context.md` wins** — its corrections are folded in
below and flagged where they change the PDF's instructions.

---

## 1. The problem, and what we claim

When something material is about to happen at a company, a small number of
people know before the public. Some of them trade. The stock starts moving
before the announcement: volume picks up, the price drifts, and nothing is in
the news because officially nothing has happened. Then the 8-K lands and
everyone reads it at the same moment.

Nobody catches this while it is happening. Regulators find it months later, by
hand, usually only once a case is already suspicious.

**The system:** a monitor that watches a stock hour by hour, sees only past
prices and volumes, and raises a hand — *"something is coming for this
company"* — then measures honestly how many trading hours of warning it gives
and how often it cries wolf.

**The claim, stated so it survives a viva:**

> Pre-announcement price and volume anomalies are well documented in finance,
> and machine learning has been applied to insider trading detection using
> regulator-held investor-level data. Our contribution is different: we build a
> forward-running detector that uses only publicly available hourly price and
> volume, frames the problem as sequential change detection, and evaluates
> detection delay at a controlled false-alarm rate across all 8-K announcement
> categories.

Four restrictions do the work, each genuinely excluding the prior literature:
**forward not retrospective · public data only · a stopping decision rather
than post-hoc detection · detection delay at a fixed alert budget.**

**Do not claim to have discovered pre-announcement leakage.** It has been
documented since roughly 1981.

### Related work to engage with

| Prior work | Why it does not cover this |
|---|---|
| ML insider-trading detection (EPJ Data Science, Italian exchange) | Uses investor-level trading records — individually identified traders, regulator-only data. We use public price/volume. |
| ML on 8-K filings (Springer forecasting studies, Stanford CS224N, 34k-filing deep learning) | All predict returns **from filing text, after the filing**. Their clock starts where ours ends. |
| Augustin, Brenner & Subrahmanyam (Management Science, 2019); options open-interest studies | Options data, takeovers only, retrospective. Explains events after the fact. |
| Financial early-warning systems | Crisis and bankruptcy prediction, not per-stock announcement detection. |

### Terminology (held strictly, everywhere)

- **footprint** — never *insider trading*. Volume spikes have innocent causes.
- **t₀** — the moment news became public. **Not** simply the filing time (§4).
- **alert budget** — fixed false alarms per stock per month; the denominator
  for every headline number.
- **CUSUM** — the classical sequential change-detection baseline.
- **detection delay** — the optimal-stopping term for lead time; use it when
  positioning against the change-detection literature.

---

## 2. Framing: optimal stopping, not "we used PPO"

The obvious attack on a WAIT/FLAG agent is *"this is a threshold classifier
dressed up as reinforcement learning — why RL?"* That is a fair hit, and the
answer is not to defend RL. The answer is that the task genuinely **is**
sequential change detection / optimal stopping: deciding *when* enough evidence
has accumulated to declare a regime change, trading detection delay against
false-alarm rate.

That reframing buys three things at once:

1. a principled justification for hour-by-hour sequential decisions over
   independent per-hour classification;
2. **CUSUM** as a strong non-neural baseline;
3. existing literature to position against.

**Build CUSUM before the learned agent.** *"We compare a learned stopping
policy against CUSUM quickest-change-detection on identical features"* is a far
stronger sentence than *"we used PPO."*

---

## 3. Architecture

```
SEC EDGAR ──► filings ──┐
                        ├──► events (t₀ corrected, filtered) ──► features ──► policy ──► alerts
Finnhub/GDELT ──► news ─┘                                          ▲
                                                                   │
yfinance ──────────────► bars ─────────────────────────────────────┘
```

Storage: SQLite at `data/db/footprints.db`, six tables (`src/db.py`).
All timestamps are UTC epoch seconds. All writes are idempotent upserts.
Every knob lives in `config/config.yaml`.

### Data sources

| Source | Role | Cost / limits |
|---|---|---|
| SEC EDGAR `data.sec.gov/submissions/` | Events + item codes + exact acceptance times | Free, no key. Descriptive User-Agent with contact address; ≤10 req/s. That is the entire ToS. |
| yfinance | Hourly + daily OHLCV | Free. ~2 years of hourly history, **rolling**. |
| Finnhub `/company-news` | t₀ correction — exact per-ticker article timestamps | Free key, ~60 calls/min. |
| GDELT DOC API | Breadth: small and foreign outlets Finnhub misses | Free, unreliable under load. **Never a blocking dependency.** |

---

## 4. CORRECTION: t₀ is not the filing time

*Corrects the PDF, which treats `acceptanceDateTime` as t₀.*

Companies typically issue a **press release first**, then file the 8-K with
that release attached as an exhibit. The gap is often minutes, sometimes hours.
The PDF's own evidence log shows the problem without noticing it: acceptance
times cluster at 20:00–21:00 UTC, just after the 16:00 ET close — but earnings
press releases hit the wire at ~16:05 ET. The filing is accepted 30–90 minutes
*after* the market already knew.

**Consequence if uncorrected:** part of the measured warning window is time
when the news was already public and the price was moving for ordinary reasons.
Nothing crashes. Lead-time numbers come out looking excellent and are wrong. A
finance-literate examiner will ask about this.

**Fix:**

```
t₀ = min(8-K acceptanceDateTime, earliest news article timestamp for that ticker/event)
```

**Architectural consequence:** news collection is **core label infrastructure
and must run from week 1**, not an optional channel added in weeks 9–11 as the
PDF schedules it. `news.py` already exists and speaks Finnhub — this is a
scheduling change, not new code.

**Report both** filing-time t₀ and news-adjusted t₀, and show the difference.
That comparison is itself a small contribution and pre-empts the question
entirely.

---

## 5. CORRECTION: usable event count

*Corrects the PDF's ~30,000 labelled events.*

That is the raw filing count and it does not survive filtering. Beyond the
exclusions the PDF names (9.01, 2.02, 5.07), much of `8.01` and `1.01` is also
routine — dividend declarations, credit facility amendments, investor-day
notices.

Genuinely unscheduled and market-moving is plausibly **10–20% of filings**.
**Plan for ~3,000–5,000 usable positives.** Still ample; the project must not
be sized on 30,000.

**Add a materiality filter** the PDF does not specify: post-announcement
absolute return above a stated threshold, set in config
(`materiality.min_abs_return`, benchmark-relative).

### Item codes

| Item | Meaning | Treatment |
|---|---|---|
| 9.01 | Financial statements & exhibits | **Exclude.** Not an event type — it rides along with other items and would dominate the labels. |
| 2.02 | Results of operations (earnings) | Scheduled. Kept, always reported separately. |
| 5.07 | Shareholder vote results | Exclude — routine paperwork. |
| 1.01 | Entry into a material agreement | Deals — the best target. |
| 5.02 | Departure/appointment of officers | Unscheduled, high signal. |
| 8.01 | Other material events | Unscheduled, good target, partly routine. |
| 7.01 | Regulation FD disclosure | Mixed. |
| 2.03 | Financial obligation created | Mixed. |

**Every headline number is split scheduled vs unscheduled and never pooled.**
Item 2.02 is ~36% of the data with dates published weeks ahead; predicting them
is trivial and worthless. Pooling would inflate the headline number and any
examiner with a finance background will catch it. The scheduled half is a
*sanity check*: if the model cannot predict known earnings dates, something is
broken.

---

## 6. CORRECTION: a null result is not automatically a finding

*Corrects the PDF's claim that Idea 1 "cannot produce a null result".*

Only half true. If the model finds nothing, there are two explanations:

1. Nothing is there ← the interesting answer
2. The model was underpowered ← the embarrassing answer

An examiner assumes (2) unless it is ruled out. **A null result only counts as
a finding if respectable baselines also fail on the same data.** This is the
second reason CUSUM and gradient boosting must exist and be genuinely
well-tuned, not token comparisons.

Related: **report the learned agent's action distribution**, not only its
accuracy. With few events the most likely failure is a degenerate policy —
always wait, or always fire immediately. Check it and report it explicitly.

---

## 7. CORRECTION: download timing and effort

*Corrects the PDF's "freeze in week 6" and "market.py reusable as-is".*

**Freeze the price snapshot in week 3, not week 6.** yfinance's ~2-year hourly
history is a *rolling* window — bars available today silently disappear later.
Waiting until week 6 risks losing the earliest part of the study window. Record
the download date in the database (`meta` table, `--stamp-snapshot`).

**`market.py` needs no code changes, but the download is not a quick job.**
1,500 tickers of hourly history is large and throttle-prone, and **broad
coverage is required — not just event windows** — because negative sampling and
trailing z-scores both need continuous history. Budget several days of
wall-clock time and cache permanently.

---

## 8. Traps, and the fix for each

| Trap | Fix |
|---|---|
| **Most 8-Ks are filed outside market hours** (peaks 20:00–21:00 and 08:00–13:00 UTC; market hours are 13:30–20:00 UTC). "The hours before the filing" is usually the *previous trading session*. | Measure lead time in **trading hours**. Install `exchange_calendars`, add a market-hours helper to `timeutils.py` **in week 1**. Doing this later means rewriting every feature. Define the target as "before the next market close" and state the choice in the report. |
| **Brutal base rate** — ~12 filings/company/year against ~3,500 trading hours ≈ 0.3% positive. Always-quiet scores 99.7%. | Never report plain accuracy. Report **precision at a fixed alert budget** (2 alerts/stock/month). Decide this in week 1 and write the evaluation script **before** the model. Downsample negatives in training (20 quiet hours per positive) while keeping the true base rate in evaluation. |
| **Look-ahead leakage** — normalising with whole-dataset statistics; split-adjusted prices that retroactively rewrite history; building the universe from data that includes the test period. | Every statistic uses a **backward-looking window only**. Write the leakage test in **week 2, not week 12**: take a row at time *t*, rebuild every feature using only data up to *t*, assert the values match. |
| **Missing hourly bars** — gaps, half-days, holidays produce `NaN` z-scores that quietly poison a third of the rows. | Market-calendar helper first, then feature code. Add a test asserting no NaNs survive into the feature vector. |
| **Survivorship bias** — `company_tickers.json` is *today's* map. Companies acquired, renamed, or delisted during the window are missing, and those are exactly the dramatic events. | Build the universe from an SEC company list **dated at the start of the window**, not from today. Cheap to do; mentioning it shows you understand the bias. |
| **Negative sampling changes results more than the model does.** | Decide the sampling rule in week 1, write it in config, and **report results at two ratios** (`sampling.robustness_ratios`) so nobody can accuse you of tuning it. |
| **Price history rewrites itself** — yfinance auto-adjusts for splits/dividends, so historical prices change after a split. | Freeze a snapshot, record the download date, never re-download. |
| **Two-year ceiling** — hourly bars go back only ~2 years. | Accept it. If more is wanted later, add a daily-resolution arm going back 10 years as a robustness check. |
| **Silent failure** — the 2026 failure mode was not clean 403s but **HTTP 200 responses returning redirect HTML or empty JSON**; pipelines looked healthy while writing nothing for days. | **Any collector cycle that parses zero records must log loudly and fail visibly, never pass quietly.** Present in `market.py` and `news.py`; required in `edgar.py`. |
| **A "liquid universe" is not optional.** "Trading more than usual" is meaningless without a stable *usual*. On a thin stock one buyer doubles the day's volume. | Filter: average daily traded value > $5M, price > $5, listed on NYSE or Nasdaq, ≥1 year of history. Leaves ~1,000–1,500 stocks. Defensible in one sentence. |
| **Freeze the test period.** | Last 15% by date, sealed until week 14. Every threshold tuned against the test set is quiet overfitting. |

---

## 9. Build phases

Acceptance criteria are stated per phase. Do not start a phase until the
previous one passes.

### Phase 0 — Repointing (done)
Reddit code archived; schema, config, collectors, and tests rebuilt around
EDGAR. `pytest` passes.

### Phase 1 — Week 1: foundations
- Finnhub key in `.env`; real contact address in `http.user_agent`.
- `exchange_calendars` installed; market-hours helpers in `src/utils/timeutils.py`
  (`is_market_open`, `trading_hours_between`, `next_market_close`).
- Headline metric decided (precision @ 2 alerts/stock/month) and the
  **evaluation script skeleton written before any model**.
- **Start news collection now** (moved from weeks 9–11 — §4).

*Done means:* you can ask "was the market open at time *t*?" and get a correct
answer, and `src/eval/metrics.py` runs on synthetic input.

### Phase 2 — Week 2: EDGAR
- `src/collectors/edgar.py` (~120 lines): build the universe from SEC
  `company_tickers_exchange.json` dated at window start → `companies`; then one
  request per company to `data.sec.gov/submissions/CIK##########.json`, filter
  to form 8-K, store `accessionNumber`, `acceptanceDateTime`, `items`,
  `primaryDocument` → `filings`.
- Cache every raw response under `data/raw/edgar/`.
- Zero-record guard.
- **Write the leakage test now.**

*Done means:* tens of thousands of rows in `filings`; at 8 req/s, 1,500
companies takes about four minutes.

### Phase 3 — Week 3: prices, frozen
- `python -m src.collectors.market --universe` for hourly and daily bars, broad
  coverage across the whole window. **Budget several days of wall-clock time.**
- `--stamp-snapshot` to record the download date. Set `market.snapshot_frozen: true`.
- Liquidity filter applied and `companies.in_universe` populated.

*Done means:* bars cover every event with padding; no NaNs; snapshot date in `meta`.

### Phase 4 — Week 4: events
- `src/pipeline/t0.py` — the t₀ correction (§4). Report both variants.
- `src/pipeline/events.py` — item-code filtering, scheduled/unscheduled split,
  materiality filter. **Record the actual usable positive count** and compare it
  against the §5 estimate.
- `src/pipeline/features.py` — at each hour: returns over 1h/4h/1d/5d; volume
  z-score against its own trailing normal; realised volatility; the stock's move
  minus the benchmark's; trading hours to market close; days since the last 8-K;
  whether an earnings date is near.
- `src/pipeline/sampling.py` — negative windows, config-driven ratio.
- Leakage test passes. **Freeze the test period.**

*Done means:* leakage test green, test set sealed, positive count reported.

### Phase 5 — Week 5: baselines
Four, all genuinely tuned (§6):
1. **always-quiet** — the base-rate floor;
2. **volume z-score threshold** — the simple thing that might win;
3. **CUSUM** — classical quickest change detection, built *before* the agent;
4. **gradient boosting** — on identical features.

*Done means:* a real number to beat, at the fixed alert budget.
**If the simple threshold wins, that is a finding — report it.**

### Phase 6 — Weeks 6–8: the learned stopping policy
- Gymnasium environment: hourly steps, WAIT/FLAG, reward table from config.
- Small MLP policy via stable-baselines3. Trains in minutes on the RTX 3050.
- **Report the action distribution alongside accuracy** every time (§6).
- Scheduled vs unscheduled split maintained throughout.

*Done means:* the agent beats — or honestly does not beat — the baselines.

### Phase 7 — Week 8 onward: turn the live monitor on
Start it and leave it running until submission. This is the highest-value,
lowest-effort item in the plan: by submission you can say *"of the N alerts it
raised live, M were followed by a filing within 48 hours"* — genuine
forward-looking evidence that almost no student project has.

### Phase 8 — Weeks 9–11: the news channel as an ablation
Collection has been running since week 1, so this is with/without, not new
collection. Does adding news-coverage features change lead time, and by how much?

### Phase 9 — Weeks 12–13: dashboard
Streamlit, showing today's flagged tickers. **Every alert must surface its
reasons** — which features triggered it: volume multiple, sector-relative move,
time since last news — so a human can sanity-check rather than trust a black box.

### Phase 10 — Week 14: final evaluation
Unseal the test set. Run the evaluation **once**. Those are the numbers.

### Phase 11 — Weeks 15–16: report and viva
Write the limitations section around §8. Write down every trap you actually hit.

---

## 10. Intended users (the report and viva both need this)

- **Regulators / compliance — strongest.** A same-day triage queue with reasons
  attached. Current practice investigates months later, by hand, usually only
  after a complaint.
- **Researchers — solid.** The evaluation table is itself a measurement of
  market efficiency.
- **Retail investors — weakest; handle carefully.** Do **not** frame this as a
  trading edge. A ~30% detection rate with false alarms is not a strategy, and
  an examiner will immediately ask about transaction costs. Frame as
  *awareness* — "this stock is moving unusually with no public explanation" —
  never as a signal to act on.

**One-sentence version for the report:**

> A live surveillance dashboard that flags stocks showing pre-announcement
> trading footprints, plus the first systematic measurement of how much advance
> warning public market data gives across all categories of corporate disclosure.

---

## 11. What "done" looks like

- A table of precision / recall / median detection delay at a fixed alert
  budget, split by item type and by scheduled vs unscheduled.
- Comparison against four baselines, including the possibility that the simple
  threshold wins.
- Both t₀ variants reported, with the gap between them.
- Calibration curves (Brier, ECE).
- The learned policy's action distribution.
- A live monitor that has been running for weeks, with its alerts logged.
- A dashboard where every alert shows its reasons.
