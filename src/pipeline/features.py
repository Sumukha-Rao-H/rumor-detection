"""Feature builders — everything the model is allowed to see.

One rule governs this module, and breaking it does not raise an error, it
produces a number that looks good and is wrong:

    every value at time t uses only data timestamped <= t,
    through backward-looking windows only.

`tests/test_leakage.py` was written in P2-08 against this module before any of
it existed, precisely so the check could not be shaped around the
implementation. It perturbs rows after `t`, recomputes, and flags any earlier
value that moved.

**Frame contract.** Each builder takes ONE ticker's bars: a DataFrame indexed by
`ts_utc` (UTC epoch seconds, ascending) with at least `close` and `volume`, and
returns a DataFrame on the same index. One ticker at a time is not a
convenience — a frame holding several would let a lag reach across a company
boundary and compute Apple's return from Microsoft's price, and no test of
"does it read the future" would catch that, because the timestamps would be
perfectly ordered.

**Horizons are in BARS, not wall-clock hours.** Bars exist only while the market
is open, so a positional lag is automatically a trading-time lag — which is what
this project asks for everywhere else. A wall-clock lag would reach across a
weekend and compare Monday's price with Friday's while calling it 24 hours. The
imprecision worth naming: a session yields 7 bars in 6.5 hours (six full hours
plus the 15:30 half-hour stub), so 24 bars is about 22.3 trading hours.
"""

from __future__ import annotations

import argparse
import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.timeutils import next_market_close, trading_hours_between

log = logging.getLogger(__name__)


def _assert_sorted(values, what: str) -> None:
    """Guard the module's one precondition: ascending timestamps.

    Every builder below computes positionally — `pct_change`, `rolling`,
    `shift`, `searchsorted` — and simply trusts that row order matches
    `ts_utc` order. `build_matrix` always feeds sorted SQL output, so this
    never fires in the real pipeline. But these are public, individually
    importable functions, and the module's own header warns that breaking the
    no-future rule "does not raise an error, it produces a number that looks
    good and is wrong" — that is equally true of an unsorted input, just via
    an unguarded precondition instead of the intended failure mode. A loud
    `ValueError` here is cheap; a silently wrong feature is not.
    """
    arr = np.asarray(values)
    if arr.size > 1 and not np.all(arr[:-1] <= arr[1:]):
        raise ValueError(f"{what} must be sorted ascending by timestamp")


def returns(frame: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Trailing returns over each configured horizon.

        ret_{h}h(t) = close(t) / close(t - h bars) - 1

    Both prices are at or before `t`, so the value is knowable at `t`.

    NaN where the horizon reaches before the series starts, or where a bar is
    missing. Deliberately not filled: a zero would be a fabricated observation
    of "no move", and forward-filling would invent a price that never traded.
    `pct_change` no longer fills by default on pandas 3, so this is the library
    behaviour rather than something layered on top.

    NaN, not infinity, when the prior close is zero or negative. `pct_change`
    would otherwise divide by that price and return +/-inf — the same
    zero-denominator failure `volume_zscore` guards against for its own
    baseline. An undefined return is the honest value; `+inf` would sail
    through `print_matrix_report`'s NaN-only audit and poison every
    downstream statistic that touches it.
    """
    cfg = cfg or load_config()
    close = frame["close"]
    _assert_sorted(frame.index, "frame index")
    return pd.DataFrame(
        {f"ret_{h}h": close.pct_change(h).where(close.shift(h) > 0)
         for h in cfg["features"]["return_horizons_h"]},
        index=frame.index,
    )


def volume_zscore(frame: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """How unusual this hour's volume is against the stock's own recent normal.

    The project's central feature. If the detector works at all, it works
    because of this number.

        base = volume.shift(1).rolling(window, min_periods=min_baseline_bars)
        z    = (volume - base.mean()) / base.std()

    **`.shift(1)` is the point.** Without it the current bar sits inside its own
    baseline, so a genuine spike inflates the mean and standard deviation it is
    measured against and damps the very signal we are hunting. A 10x volume
    hour would score lower than it should, and nothing would look wrong.

    **The leakage detector cannot catch that.** It perturbs rows AFTER t;
    including the current bar uses data AT t, which the no-future rule permits.
    So the property has its own test in `test_features_volume.py` rather than
    relying on `assert_no_lookahead`.

    Zero variance yields NaN, never infinity. If every baseline bar has the same
    volume the standard deviation is 0, and any differing current bar would
    divide to +/-inf — which propagates silently through every downstream
    statistic and blows up any model that meets it. The frozen snapshot holds
    34,709 zero-volume bars, so a flat baseline is real, not hypothetical.
    Undefined is the honest value.
    """
    cfg = cfg or load_config()
    fcfg = cfg["features"]
    volume = frame["volume"]
    _assert_sorted(frame.index, "frame index")

    base = volume.shift(1).rolling(fcfg["volume_zscore_window_h"],
                                   min_periods=fcfg["min_baseline_bars"])
    std = base.std()
    z = (volume - base.mean()) / std
    return pd.DataFrame({"volume_z": z.where(std > 0)}, index=frame.index)


def realised_volatility(frame: pd.DataFrame,
                        cfg: dict | None = None) -> pd.DataFrame:
    """How much this stock has been moving lately: the sd of one-bar returns.

    Context for the footprint — a 3% move means something different in a calm
    stock than in a jumpy one.

    **No `.shift(1)` here, and that is deliberate.** P4-07 made the shift the
    whole task, so its absence needs a reason. The z-score is a *comparison*:
    it asks how far THIS bar sits from its baseline, so the bar must not be in
    the baseline or it drags normal toward itself. Volatility is a
    *description*: it asks how much the stock has been moving, and the move
    realised at `t` is part of the honest answer. It is knowable at `t`, from
    `close(t)` and `close(t-1)`, both at or before `t`. Excluding it would not
    be safer, only staler — reporting the previous bar's volatility while
    labelling it the current one.

    `min_periods` equals the window, not `min_baseline_bars`: volatility "over
    120 bars" should mean 120 bars. Reusing `min_baseline_bars` would also set a
    trap, since it happens to equal the window today — shrinking the window
    would silently make the feature NaN forever.

    Not annualised. Annualising multiplies by a constant, which changes nothing
    a model can use, and the constant needs a bars-per-year figure the
    7-bars-per-6.5-hour session makes awkward to state honestly.
    """
    cfg = cfg or load_config()
    window = cfg["features"]["volatility_window_h"]
    _assert_sorted(frame.index, "frame index")
    one_bar = frame["close"].pct_change(1)
    return pd.DataFrame(
        {"volatility": one_bar.rolling(window, min_periods=window).std()},
        index=frame.index,
    )


def benchmark_relative(frame: pd.DataFrame, benchmark: pd.DataFrame,
                       cfg: dict | None = None) -> pd.DataFrame:
    """The stock's move minus the market's, over the same span.

    A stock up 4% on a day the whole market rose 4% did nothing in particular.
    Stripping the market out is what leaves something company-specific to
    detect.

    **Aligned before differencing, not after.** The benchmark is reindexed onto
    the stock's timestamps first, so its "h bars ago" is h bars in the STOCK's
    index and both returns cover an identical set of bars — which is what "over
    the same span" has to mean. Computing the benchmark's return on its own
    index and reindexing afterwards would compare the stock's last h bars with
    the benchmark's last h bars, the same thing only when the two series align
    perfectly.

    Measured across 60 sampled universe tickers: 37 match SPY's timestamps
    exactly and the worst case is 1 bar in 1,733. So the ordering changes almost
    nothing today; it is written this way so it stays correct if a future
    benchmark or window is less tidy.

    A missing benchmark bar gives NaN, never a forward-filled price:
    substituting an older market move and calling the difference
    company-specific is precisely the error this feature exists to avoid.

    A zero or negative close, in either leg, also gives NaN rather than the
    +/-inf `pct_change` would otherwise divide out to — the same guard
    `returns()` applies for the same reason.

    Simple excess return, not a beta-adjusted market-model residual. Beta would
    have to be estimated on yet another rolling window with its own minimum and
    its own leakage surface, and nothing has asked for it. The plain difference
    is what `config.yaml` describes and what `materiality.py` already uses for
    the label, so the feature and the label measure the same quantity.

    The benchmark frame is supplied by the caller: the BUILDERS do no I/O.
    Only `build_matrix` below touches the database, and only to feed them.
    """
    cfg = cfg or load_config()
    fcfg = cfg["features"]
    _assert_sorted(frame.index, "frame index")
    if not fcfg["include_benchmark_relative"]:
        return pd.DataFrame(index=frame.index)

    close = frame["close"]
    bench = benchmark["close"].reindex(frame.index)

    def rel_return(series: pd.Series, h: int) -> pd.Series:
        return series.pct_change(h).where(series.shift(h) > 0)

    return pd.DataFrame(
        {f"ret_rel_{h}h": rel_return(close, h) - rel_return(bench, h)
         for h in fcfg["return_horizons_h"]},
        index=frame.index,
    )


DAY_S = 86400


@lru_cache(maxsize=100_000)
def _hours_to_close(ts_utc: int, calendar: str) -> float:
    """Trading hours from `ts_utc` to the next close of that session.

    Cached because every ticker shares the same bar timestamps: the calendar
    lookup costs 0.67 ms, which is 1.2 s per ticker and roughly half an hour
    across 1,500 if recomputed. There are only ~1,733 distinct timestamps in
    the study, so this is computed once and free thereafter.

    Out of hours it returns the *next* session's remaining length, so a
    pre-open bar reports a full session ahead of it. A timestamp outside the
    exchange calendar raises, by P1-03's design — a silent fallback would let
    the Phase 7 live monitor conclude the market is permanently shut.
    """
    return trading_hours_between(ts_utc, next_market_close(ts_utc))


def _days_since(index: pd.Index, event_times: np.ndarray) -> pd.Series:
    """Calendar days from the most recent event at or before each timestamp.

    `side="right"` is the whole safety property: a filing accepted at exactly
    `t` counts, because it is public at `t`, and anything later does not. The
    naive alternative — "the most recent filing in the table" — would hand the
    model the answer.

    NaN where no prior event exists. "No previous filing" is not "a filing
    infinitely long ago", and 0 would be a lie in the opposite direction.
    """
    stamps = np.asarray(index, dtype=np.int64)
    if event_times.size == 0:
        return pd.Series(np.nan, index=index)
    _assert_sorted(event_times, "event_times")
    pos = np.searchsorted(event_times, stamps, side="right") - 1
    last = np.where(pos >= 0, event_times[np.clip(pos, 0, None)], np.nan)
    return pd.Series((stamps - last) / DAY_S, index=index)


def context_signals(frame: pd.DataFrame, filing_times: np.ndarray | None = None,
                    earnings_times: np.ndarray | None = None,
                    cfg: dict | None = None) -> pd.DataFrame:
    """Where we are in the session, and how long since this company spoke.

    `filing_times` and `earnings_times` are that ticker's 8-K acceptance times,
    sorted ascending. Supplied by the caller; this module does no I/O.

    ⚠ **The event being predicted IS an 8-K**, so `days_since_last_8k` falls to
    0 at exactly t0. A feature row at or after t0 does not merely leak — it
    announces the event. The decision window must be STRICTLY before t0, which
    is P4-11's boundary to enforce. The leakage detector cannot catch it: it
    perturbs rows after `t`, and this reads a filing AT `t`.

    Earnings proximity is days since the LAST results filing, never days until
    the next. We have no earnings calendar, only 2.02 filings recording that
    results happened, so "until the next" would mean reading a future filing.
    Earnings are quarterly, so "since the last" carries the same information
    legitimately, and any "expected days until" is a deterministic transform a
    model can make for itself.

    Units differ on purpose. Hours to close is in TRADING hours — it is a
    question about the session. Days since is in CALENDAR days — it measures how
    stale news is, and news ages over a weekend.
    """
    cfg = cfg or load_config()
    fcfg = cfg["features"]
    calendar = cfg["market"]["calendar"]
    _assert_sorted(frame.index, "frame index")

    out: dict[str, pd.Series] = {
        "trading_hours_to_close": pd.Series(
            [_hours_to_close(int(ts), calendar) for ts in frame.index],
            index=frame.index),
    }
    if fcfg["include_days_since_last_8k"]:
        out["days_since_last_8k"] = _days_since(
            frame.index, np.asarray(filing_times if filing_times is not None
                                    else [], dtype=np.int64))
    if fcfg["include_earnings_proximity"]:
        out["days_since_last_earnings"] = _days_since(
            frame.index, np.asarray(earnings_times if earnings_times is not None
                                    else [], dtype=np.int64))
    return pd.DataFrame(out, index=frame.index)


# --------------------------------------------------------------------------
# assembly — the only part of this module that touches the database
# --------------------------------------------------------------------------

#: Identifying columns, matching `src/eval/contract.py` exactly so a model's
#: output is this frame plus `score` and `action`, with no rename sitting
#: between the matrix and the metric.
ID_COLUMNS = ("window_id", "ticker", "ts_utc", "t0_utc", "is_scheduled",
              "item_code")


def _ticker_frame(conn, ticker: str, interval: str) -> pd.DataFrame:
    """A ticker's bars, indexed by `ts_utc`. Empty (not a crash) with zero bars.

    `pd.DataFrame([])` has no columns at all, so `.set_index("ts_utc")` would
    raise `KeyError` on a ticker with no rows for this interval — a plausible
    data gap (a collector miss, a delisted name, a benchmark not yet backfilled)
    rather than a hypothetical. Returning an empty-but-valid frame instead lets
    `build_matrix`'s own `if frame.empty: continue` guard actually run, rather
    than a raw `KeyError` taking down the whole assembly.
    """
    rows = conn.execute(
        "SELECT ts_utc, close, volume FROM bars WHERE ticker = ? AND "
        "interval = ? ORDER BY ts_utc", (ticker, interval)).fetchall()
    if not rows:
        return pd.DataFrame(columns=["close", "volume"]).rename_axis("ts_utc")
    return pd.DataFrame([dict(r) for r in rows]).set_index("ts_utc")


def ticker_features(frame: pd.DataFrame, benchmark: pd.DataFrame,
                    filing_times: np.ndarray, earnings_times: np.ndarray,
                    cfg: dict) -> pd.DataFrame:
    """Every feature for one ticker, over its WHOLE bar history.

    Computed on the full series and sliced afterwards, never computed on a
    48-bar slice. That ordering is correctness, not speed: `volume_z` needs 480
    prior bars of baseline and `volatility` 120, so a per-event implementation
    would return NaN for every row of every event — and would look perfectly
    reasonable while doing it.
    """
    return pd.concat([
        returns(frame, cfg),
        volume_zscore(frame, cfg),
        realised_volatility(frame, cfg),
        benchmark_relative(frame, benchmark, cfg),
        context_signals(frame, filing_times, earnings_times, cfg),
    ], axis=1)


def build_matrix(cfg: dict, conn) -> pd.DataFrame:
    """One row per (usable event, hour) over the decision window.

    **The window is strictly before t0**: `ts_utc < t0_utc`, never `<=`. The
    event being predicted is itself an 8-K, so at t0 `days_since_last_8k` is 0
    and the feature announces the event rather than merely leaking. This is
    stricter than `contract.py`, which forbids only hours AFTER t0 — the
    contract is the floor, not the target.

    The window is `decision.horizon_hours` BARS, not wall-clock hours. Bars
    exist only while the market is open, so 48 bars is 48 trading hours, and a
    stopping problem wants a fixed number of decision steps per episode, which
    wall-clock cannot give.
    """
    from src import db  # local: keeps the builders importable without the DB

    horizon = cfg["decision"]["horizon_hours"]
    interval = cfg["market"]["interval"]
    forms = cfg["edgar"]["forms"]
    scheduled_codes = cfg["items"]["scheduled"]

    events = conn.execute(
        "SELECT event_id, ticker, items, t0_utc, is_scheduled FROM events "
        "WHERE usable = 1 ORDER BY ticker, t0_utc").fetchall()
    if not events:
        raise SystemExit(
            "no usable events — run the Phase 4 filters before assembling "
            "features."
        )
    benchmark = _ticker_frame(conn, cfg["market"]["benchmark"], interval)

    by_ticker: dict[str, list] = {}
    for e in events:
        by_ticker.setdefault(e["ticker"], []).append(e)

    marks = ",".join("?" * len(forms))
    out = []
    for ticker, ticker_events in by_ticker.items():
        frame = _ticker_frame(conn, ticker, interval)
        if frame.empty:
            continue
        filings = np.array([r[0] for r in conn.execute(
            f"SELECT acceptance_utc FROM filings WHERE ticker = ? AND form IN "
            f"({marks}) AND acceptance_utc IS NOT NULL ORDER BY acceptance_utc",
            (ticker, *forms))], dtype=np.int64)
        if scheduled_codes:
            earnings = np.array([r[0] for r in conn.execute(
                "SELECT acceptance_utc FROM filings WHERE ticker = ? AND "
                "acceptance_utc IS NOT NULL AND (" +
                " OR ".join("items LIKE ?" for _ in scheduled_codes) +
                ") ORDER BY acceptance_utc",
                (ticker, *[f"%{c}%" for c in scheduled_codes]))], dtype=np.int64)
        else:
            # An empty `items.scheduled` means no filing can ever match — the
            # SQL fragment would otherwise be `AND ()`, a syntax error, for a
            # config that is a valid (if unusual) choice.
            earnings = np.array([], dtype=np.int64)

        feats = ticker_features(frame, benchmark, filings, earnings, cfg)
        stamps = feats.index.to_numpy()

        for e in ticker_events:
            t0 = e["t0_utc"]
            end = np.searchsorted(stamps, t0, side="left")   # strictly before
            window = feats.iloc[max(end - horizon, 0):end]
            if window.empty:
                continue
            block = window.copy()
            block.insert(0, "item_code", e["items"])
            block.insert(0, "is_scheduled", bool(e["is_scheduled"]))
            block.insert(0, "t0_utc", t0)
            block.insert(0, "ticker", ticker)
            block.insert(0, "window_id", e["event_id"])
            out.append(block.reset_index())

    if not out:
        # Every usable event's window came out empty (e.g. every t0 lands at
        # or before its ticker's very first bar) — `pd.concat([])` would raise
        # a raw `ValueError: No objects to concatenate` before the check below
        # ever ran. Same guard, same message, reached from both routes to it.
        raise SystemExit("feature matrix is empty — check bar coverage.")
    matrix = pd.concat(out, ignore_index=True)
    if matrix.empty:
        raise SystemExit("feature matrix is empty — check bar coverage.")
    return matrix


def write_matrix(cfg: dict, conn) -> pd.DataFrame:
    """Build and write to `paths.processed`. Overwrites, never appends."""
    from pathlib import Path

    matrix = build_matrix(cfg, conn)
    dest = Path(cfg["paths"]["processed"]) / "features.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(dest, index=False)
    log.info("feature matrix: %d rows x %d cols -> %s",
             len(matrix), matrix.shape[1], dest)
    return matrix


def print_matrix_report(cfg: dict, matrix: pd.DataFrame) -> None:
    """Row and NaN counts, each explained — the acceptance criterion."""
    horizon = cfg["decision"]["horizon_hours"]
    fcfg = cfg["features"]
    windows = matrix["window_id"].nunique()

    print(f"\n=== Feature matrix ===")
    print(f"rows            : {len(matrix):,}")
    print(f"windows (events): {windows:,}")
    print(f"rows per window : {len(matrix)/windows:.1f} (horizon {horizon} bars)")
    print(f"tickers         : {matrix['ticker'].nunique():,}")
    print(f"columns         : {matrix.shape[1]}")

    short = matrix.groupby("window_id").size()
    print(f"\nwindows shorter than {horizon} bars: {(short < horizon).sum():,}"
          f"  (history starts inside the window; not padded)")

    print(f"\nNaN by column, with the reason:")
    reasons = {
        "volume_z": f"needs {fcfg['volume_zscore_window_h']}-bar baseline, "
                    f"min {fcfg['min_baseline_bars']}",
        "volatility": f"needs {fcfg['volatility_window_h']} prior returns",
        "days_since_last_8k": "no earlier filing for that ticker",
        "days_since_last_earnings": "no earlier results filing",
    }
    for col in matrix.columns:
        n = int(matrix[col].isna().sum())
        if not n:
            continue
        why = reasons.get(col)
        if why is None and col.startswith("ret_rel_"):
            why = "benchmark bar missing at t or t-h"
        elif why is None and col.startswith("ret_"):
            why = f"horizon reaches before the ticker's first bar"
        print(f"  {col:<26}{n:>9,}  ({n/len(matrix):>5.1%})  {why or ''}")

    # NaN is the honest "undefined". +/-inf is not, and `.isna()` above would
    # never see it (a zero-price bar dividing out to infinity, say) — checked
    # separately so a builder's missing zero-denominator guard cannot hide in
    # a report that otherwise reads "every NaN explained".
    numeric = matrix.select_dtypes(include="number")
    inf_counts = {col: int(np.isinf(numeric[col].to_numpy(dtype=float)).sum())
                 for col in numeric.columns}
    inf_counts = {col: n for col, n in inf_counts.items() if n}
    assert not inf_counts, f"non-finite values found (a builder's guard is missing): {inf_counts}"

    assert (matrix["ts_utc"] < matrix["t0_utc"]).all()
    print(f"\nEvery row is strictly before its t0 — checked, not assumed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true",
                        help="assemble the matrix and write it to parquet")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    from src import db
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    matrix = write_matrix(cfg, conn) if args.build else build_matrix(cfg, conn)
    print_matrix_report(cfg, matrix)


if __name__ == "__main__":
    main()
