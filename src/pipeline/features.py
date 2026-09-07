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


def news_coverage(frame: pd.DataFrame,
                  article_times: np.ndarray | None = None,
                  publishers: list[str] | None = None,
                  cfg: dict | None = None) -> pd.DataFrame:
    """P8-01. How loudly the press was already covering this company.

    Three signals per configured window, all strictly backward-looking:

      news_count_<W>h      articles published in (t-W, t]
      news_breadth_<W>h    DISTINCT publishers over that same window
      hours_since_news     hours to the most recent article at or before t

    Count and breadth are separate on purpose. Twenty articles from one wire
    aggregator republishing itself is not the same event as five articles from
    five newsrooms, and a count alone cannot tell them apart — which matters
    here, because this corpus is dominated by a handful of aggregators.

    **The boundary is `<= t`, matching `_days_since`.** An article published at
    exactly t is public at t. Anything later is the future, and the leakage
    test perturbs it to prove this reads none of it.

    ⚠ **A zero means "nothing was published", not "nobody asked".** That holds
    only because P4-00b fetched every week of the window for every in-universe
    ticker, not merely the weeks containing a filing. Without it the negatives
    — which sit >=168 h from any filing, exactly the weeks a filing-only
    backfill never requests — would carry a structural zero while positives
    carried real counts, and the Phase 8 ablation would report a large delta
    caused by collection scope rather than by the market. See issue 23.

    `article_times` must be sorted ascending, with `publishers` parallel to it.
    Supplied by the caller; this module does no I/O.
    """
    cfg = cfg or load_config()
    windows = cfg["features"]["news_windows_h"]
    stamps = np.asarray(frame.index, dtype=np.int64)
    times = np.asarray(article_times if article_times is not None else [],
                       dtype=np.int64)
    names = list(publishers or [])
    out: dict[str, pd.Series] = {}

    if times.size:
        _assert_sorted(times, "article_times")
    if names and len(names) != times.size:
        raise ValueError(
            f"publishers has {len(names)} entries for {times.size} article "
            "times — they must be parallel, or breadth counts the wrong rows.")

    # Most recent article at or before t. NaN where none exists: "no coverage
    # yet" is not "coverage infinitely long ago", and a large number would be
    # a lie in the opposite direction — the same rule `_days_since` follows.
    if times.size:
        pos = np.searchsorted(times, stamps, side="right") - 1
        last = np.where(pos >= 0, times[np.clip(pos, 0, None)], np.nan)
        hours_since = (stamps - last) / 3600.0
    else:
        hours_since = np.full(stamps.size, np.nan)
    out["hours_since_news"] = pd.Series(hours_since, index=frame.index)

    for w in windows:
        span = int(w) * 3600
        hi = np.searchsorted(times, stamps, side="right")
        lo = np.searchsorted(times, stamps - span, side="right")
        out[f"news_count_{w}h"] = pd.Series(hi - lo, index=frame.index)

        # Distinct publishers over the same window. Two pointers with a running
        # tally rather than a set per row: `stamps` is ascending, so both edges
        # only ever move forward, which is O(bars + articles) instead of the
        # O(bars x articles) a per-row set would cost on a 2.4M-row matrix.
        breadth = np.zeros(stamps.size, dtype=np.int64)
        if names:
            tally: dict[str, int] = {}
            left = right = 0
            for i in range(stamps.size):
                while right < hi[i]:
                    tally[names[right]] = tally.get(names[right], 0) + 1
                    right += 1
                while left < lo[i]:
                    n = tally[names[left]] - 1
                    if n:
                        tally[names[left]] = n
                    else:
                        del tally[names[left]]
                    left += 1
                breadth[i] = len(tally)
        out[f"news_breadth_{w}h"] = pd.Series(breadth, index=frame.index)

    return pd.DataFrame(out, index=frame.index)


def ticker_features(frame: pd.DataFrame, benchmark: pd.DataFrame,
                    filing_times: np.ndarray, earnings_times: np.ndarray,
                    cfg: dict, article_times: np.ndarray | None = None,
                    publishers: list[str] | None = None) -> pd.DataFrame:
    """Every feature for one ticker, over its WHOLE bar history.

    Computed on the full series and sliced afterwards, never computed on a
    48-bar slice. That ordering is correctness, not speed: `volume_z` needs 480
    prior bars of baseline and `volatility` 120, so a per-event implementation
    would return NaN for every row of every event — and would look perfectly
    reasonable while doing it.
    """
    parts = [
        returns(frame, cfg),
        volume_zscore(frame, cfg),
        realised_volatility(frame, cfg),
        benchmark_relative(frame, benchmark, cfg),
        context_signals(frame, filing_times, earnings_times, cfg),
    ]
    # Off unless config says otherwise, because Phase 8 is an ABLATION: the
    # with/without switch has to be a config flag rather than a code edit, or
    # the two arms are not otherwise-identical and the delta means nothing.
    # Default false also keeps every Phase 5/6 number reproducible unchanged.
    if cfg["features"].get("include_news_coverage", False):
        parts.append(news_coverage(frame, article_times, publishers, cfg))
    return pd.concat(parts, axis=1)


def _event_times(conn, cfg: dict, ticker: str) -> tuple[np.ndarray, np.ndarray]:
    """(all filing times, scheduled-filing times) for one ticker.

    Shared by the positive and quiet builders so the two cannot drift: a quiet
    window whose `days_since_last_8k` were computed from a different filing set
    than a positive's would be distinguishable by something other than its
    label, which is precisely what `sampling.py` forbids.
    """
    forms = cfg["edgar"]["forms"]
    scheduled_codes = cfg["items"]["scheduled"]
    marks = ",".join("?" * len(forms))
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
        # An empty `items.scheduled` means no filing can ever match — the SQL
        # fragment would otherwise be `AND ()`, a syntax error, for a config
        # that is a valid (if unusual) choice.
        earnings = np.array([], dtype=np.int64)
    return filings, earnings


def _news_arrays(conn, cfg: dict, ticker: str
                 ) -> tuple[np.ndarray | None, list[str] | None]:
    """(article times, publishers) for one ticker, or (None, None) when off.

    Deliberately beside `_event_times` and returning the same shape of thing,
    so the four builders that call one and then the other cannot wire up half
    of it. When `include_news_coverage` is false this does no query at all —
    the "without" arm of the ablation must not pay for data it never reads.
    """
    if not cfg["features"].get("include_news_coverage", False):
        return None, None
    from src import db  # local: keeps this module importable without the DB

    times, names = db.news_times(
        conn, ticker, max_tier=cfg["features"].get("news_max_tier", 2))
    return np.asarray(times, dtype=np.int64), names


def build_quiet_matrix(cfg: dict, conn,
                       pairs: list[tuple[str, int]]) -> pd.DataFrame:
    """One row per (quiet window, hour), in the positives' exact shape.

    `pairs` are the `(ticker, anchor)` tuples `sampling.draw()` returns.

    `sampling.py` fixes the shape and the reason for it: *"Quiet windows are
    built in the same shape as positives — 48 bars ending strictly before an
    anchor — so nothing distinguishes the two except the label. Any structural
    difference would be something a model could learn instead of the market."*
    So this shares `_event_times`, `ticker_features` and the same
    `searchsorted(..., side="left")` boundary as `build_matrix`, rather than
    reimplementing them alongside.

    `t0_utc`, `is_scheduled` and `item_code` are null — the contract requires
    those three to be null together, and null `t0_utc` IS the negative label.

    Window ids are prefixed `quiet:` so a negative can never collide with an
    `event_id` when the two matrices are concatenated.

    Returns an empty frame (not a raise) when `pairs` is empty: drawing zero
    negatives is a legitimate configuration, unlike an events table with no
    usable rows.
    """
    from src import db  # local: keeps the builders importable without the DB

    horizon = cfg["decision"]["horizon_hours"]
    interval = cfg["market"]["interval"]
    if not pairs:
        return pd.DataFrame()

    benchmark = _ticker_frame(conn, cfg["market"]["benchmark"], interval)
    by_ticker: dict[str, list[int]] = {}
    for ticker, anchor in pairs:
        by_ticker.setdefault(ticker, []).append(int(anchor))

    out, skipped = [], 0
    for ticker, anchors in by_ticker.items():
        frame = _ticker_frame(conn, ticker, interval)
        if frame.empty:
            skipped += len(anchors)
            continue
        filings, earnings = _event_times(conn, cfg, ticker)
        articles, publishers = _news_arrays(conn, cfg, ticker)
        feats = ticker_features(frame, benchmark, filings, earnings, cfg,
                                article_times=articles,
                                publishers=publishers)
        stamps = feats.index.to_numpy()

        for anchor in sorted(anchors):
            end = np.searchsorted(stamps, anchor, side="left")  # strictly before
            window = feats.iloc[max(end - horizon, 0):end]
            if window.empty:
                skipped += 1
                continue
            block = window.copy()
            block.insert(0, "item_code", None)
            block.insert(0, "is_scheduled", None)
            block.insert(0, "t0_utc", None)
            block.insert(0, "ticker", ticker)
            block.insert(0, "window_id", f"quiet:{ticker}:{anchor}")
            out.append(block.reset_index())

    if skipped:
        # Reported, never silent: a shortfall that goes unmentioned turns a
        # 3:1 sample into some other ratio while still calling itself 3:1.
        print(f"build_quiet_matrix: {skipped} of {len(pairs)} quiet windows "
              f"skipped (no bars, or the anchor precedes the ticker's history)")
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


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

    out = []
    for ticker, ticker_events in by_ticker.items():
        frame = _ticker_frame(conn, ticker, interval)
        if frame.empty:
            continue
        filings, earnings = _event_times(conn, cfg, ticker)
        articles, publishers = _news_arrays(conn, cfg, ticker)

        feats = ticker_features(frame, benchmark, filings, earnings, cfg,
                                article_times=articles,
                                publishers=publishers)
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


def matrix_path(cfg: dict):
    """Where THIS configuration's feature matrix lives.

    The filename carries the ablation arm. Two things follow, and both are the
    point of P8-02: the arms cannot overwrite one another, and a comparison
    cannot silently read the arm it did not mean to. Flipping
    `include_news_coverage` moves the build target and the evaluation source
    together, so "identical everything else" holds by construction rather than
    by remembering to pass a matching pair of paths.
    """
    from pathlib import Path

    name = ("features-with-news.parquet"
            if cfg["features"].get("include_news_coverage", False)
            else "features.parquet")
    return Path(cfg["paths"]["processed"]) / name


def write_matrix(cfg: dict, conn) -> pd.DataFrame:
    """Build and write to `paths.processed`. Overwrites, never appends."""
    matrix = build_matrix(cfg, conn)
    dest = matrix_path(cfg)
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
