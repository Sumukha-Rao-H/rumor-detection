# NOTICE — what the licence covers, and what it cannot

The [MIT licence](LICENSE) covers the **source code** in this repository.

It does not, and cannot, license the **data** the code collects. That data
belongs to its providers, and each carries its own terms.

| source | terms |
|---|---|
| **SEC EDGAR** | Public domain as a US government work. Free to use, subject to the SEC's fair-access rules — a descriptive User-Agent with a real contact address, and no more than ten requests a second. Both are enforced in code (`src/collectors/edgar.py`, `config.edgar.min_interval_s`). |
| **Yahoo Finance** | Retrieved via `yfinance`, an unofficial client. Yahoo's terms restrict redistribution. Price bars are treated as a local cache, rebuildable from source. |
| **Finnhub** | Free tier, subject to Finnhub's terms. News articles are label infrastructure only. |

Consequently **`data/` is gitignored in full**. The one derived artifact kept
in version control is `live-log/alerts.csv` — detector outputs computed by this
code, not redistributed vendor data.

Anyone reproducing this collects their own copy from the providers above, under
their own contact details. `.env.example` and `config/config.yaml` say how; the
EDGAR client refuses to make a live request until a real contact address is
configured, precisely so a fork cannot inherit ours.

## Not investment advice

This is a research prototype produced as an academic project. It detects a
statistical **footprint** — unusual trading before a disclosure, which has many
innocent explanations: index rebalancing, an analyst note, a fund unwinding a
position. It does not identify a person, a fund, or an intent, and it is not
evidence of wrongdoing. Nothing here is investment advice or a recommendation
to trade.
