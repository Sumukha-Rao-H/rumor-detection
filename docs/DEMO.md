# Demo runbook — Phase 3–5 + dashboard

Everything below is already built and checked in on this machine. These are the
commands to reproduce it from scratch, in order, and what to say about each.

> **The labels are provisional.** The dataset is built from the *machine's*
> verdicts (`--from-proposals`), not the human-reviewed ones — Phase 2 review is
> still open. The pipeline is real end to end; the accuracy numbers are not
> reportable. The dashboard states this on screen, and the plan's own rule is
> that no machine verdict becomes a label without review.

## 0. One-time setup

```bash
source .venv/bin/activate
pip install -r requirements.txt      # already done here
```

## 1. Build the dataset (2 s)

```bash
python -m src.pipeline.dataset --from-proposals
```

451 events, temporally split 317 / 66 / 68 with a 7-day ticker quarantine at
each boundary. Class balance is **9.3% TRUE** — remember this number, it
explains the model's behaviour.

## 2. Build the state tensors (5 s)

```bash
python -m src.pipeline.features --encoder hashed
```

One `(48, 418)` matrix per event in `data/processed/states/`. 48 hourly steps ×
418 features (395 text + 8 social + 14 market + 1 step fraction), exactly §7.

`--encoder hashed` swaps MiniLM/FinBERT for a deterministic hashed
bag-of-words so the demo needs no 2 GB model download. For a real run drop the
flag and it uses the frozen encoders the plan specifies.

## 3. Train (19 s for PPO, 12 s for DQN)

```bash
python -m src.rl.train --algo ppo --timesteps 30000
python -m src.rl.train --algo dqn --timesteps 15000
```

Trains on CPU. The heavy models were already spent offline in step 2 — what
trains here is only the decision rule (418 → 256 → 128 → 3, ~140k params),
which is why this is seconds rather than hours.

## 4. Run the dashboard

```bash
streamlit run app/dashboard.py
```

Three tabs:

- **Replay an event** — pick a rumor; see the price chart, the Reddit posts
  that raised the claim, the moment the agent commits, and the moment the
  official record catches up. The hourly tape underneath shows WAIT/WAIT/…/COMMIT.
- **Evaluation** — the agent against `random` and `immediate` baselines on the
  same split, with the Δ distribution and a confusion matrix.
- **Dataset** — composition, split boundaries, class balance.

## What the numbers currently say

| policy | accuracy | F1 | mean commit | Δ median |
|---|---|---|---|---|
| PPO | 0.809 | **0.000** | 0.1 h | 72 h |
| DQN | ~0.79 | ~0 | 5.7 h | 72 h |
| random | 0.529 | 0.304 | 0.4 h | 72 h |

**Say this out loud rather than hoping nobody notices:** PPO's 80.9% accuracy
with an F1 of 0.000 means it learned to answer FALSE every single time. That is
the correct optimum for a 9.3% positive class under this reward — it is a
dataset problem, not a bug in the agent, and it is exactly why Phase 2 review
matters. The env, the reward, the leakage guards and the metrics are all
working; they are reporting an honest negative result.

The three things that would move it, in order:
1. Finish the `moved_filed` review pass (144 rows) — that is where TRUE labels
   are recoverable.
2. Class weighting / reward reshaping for the minority class.
3. Real encoders instead of `--encoder hashed`.

## Checks worth running in front of the mentor

```bash
pytest tests/                     # 261 tests
pytest tests/test_features.py -k leakage -v
```

`tests/test_features.py::test_every_row_only_uses_its_own_past` rebuilds each
state row with all future data deleted from the database and asserts the row is
unchanged. That is the plan's #1 rule enforced mechanically rather than by
discipline.
