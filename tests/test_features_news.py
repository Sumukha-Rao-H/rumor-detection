"""P8-01 — the news-coverage features.

The ablation these feed is only meaningful if two things hold: the features
read no future (or the "with news" arm wins by cheating), and a zero means
"nothing was published" rather than "nobody asked" (or the arm wins because of
collection scope, which is issue 23's trap).

The first is tested here. The second cannot be — it is a property of the
COLLECTION, not of any code path, and these tests use a temp database as
`code-standards.md` requires. It was checked against the real one instead, on
2026-09-07 over 40 tickers:

    positive windows   mean news_count_168h 13.44   median 6.57   4.5% zero
    quiet windows      mean news_count_168h  9.70   median 4.14   5.2% zero

Near-identical zero rates are the evidence. Had P4-00b not fetched every week
of the window — rather than only the weeks containing a filing — the quiet
windows would sit near 100% zero, since `sampling.quiet_gap_hours` places them
>=168 h from any filing, exactly the weeks a filing-only backfill never
requests. The 38% coverage gap that remains is market signal, which is the
thing Phase 8 exists to measure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.features import news_coverage, ticker_features
from src.utils.config import load_config
# `assert_no_lookahead` is deliberately NOT used here — see `_articles_moved`.

HOUR = 3600
BASE = 1_760_000_000


@pytest.fixture
def cfg():
    c = load_config()
    c["features"]["include_news_coverage"] = True
    c["features"]["news_windows_h"] = [24, 168]
    return c


@pytest.fixture
def frame():
    idx = pd.Index([BASE + i * HOUR for i in range(300)], name="ts_utc")
    return pd.DataFrame({"close": 100.0, "volume": 1e6}, index=idx)


# --------------------------------------------------------------------------
# the boundary
# --------------------------------------------------------------------------
def test_an_article_at_exactly_t_counts_and_one_a_second_later_does_not(cfg, frame):
    """`<= t`, matching `_days_since`. An article published at t is public at t."""
    t = frame.index[100]
    at_t = news_coverage(frame, np.array([t], dtype=np.int64), ["wire"], cfg)
    after = news_coverage(frame, np.array([t + 1], dtype=np.int64), ["wire"], cfg)

    assert at_t.loc[t, "news_count_24h"] == 1
    assert at_t.loc[t, "hours_since_news"] == 0.0
    assert after.loc[t, "news_count_24h"] == 0, (
        "an article one second after the bar is the future and must not count")
    assert np.isnan(after.loc[t, "hours_since_news"])


def test_the_window_is_half_open_on_the_left(cfg, frame):
    """(t-W, t] — an article exactly W ago has aged out."""
    t = frame.index[200]
    edge = np.array([t - 24 * HOUR, t - 24 * HOUR + 1], dtype=np.int64)
    got = news_coverage(frame, edge, ["a", "b"], cfg)
    assert got.loc[t, "news_count_24h"] == 1, (
        "the article exactly 24h old should have left the 24h window")


# --------------------------------------------------------------------------
# count is not breadth — the reason both exist
# --------------------------------------------------------------------------
def test_one_publisher_repeating_itself_is_not_broad_coverage(cfg, frame):
    """Twenty republications from one aggregator are not twenty newsrooms.

    This corpus is dominated by a handful of aggregators, so a count alone
    would read a single story echoed twenty times as a major event.
    """
    t = frame.index[150]
    times = np.array([t - i * 60 for i in range(20)][::-1], dtype=np.int64)

    echo = news_coverage(frame, times, ["Yahoo"] * 20, cfg)
    broad = news_coverage(frame, times, [f"outlet{i}" for i in range(20)], cfg)

    assert echo.loc[t, "news_count_24h"] == 20
    assert echo.loc[t, "news_breadth_24h"] == 1
    assert broad.loc[t, "news_count_24h"] == 20
    assert broad.loc[t, "news_breadth_24h"] == 20


def test_breadth_falls_as_publishers_age_out_of_the_window(cfg, frame):
    """The sliding tally must retract, not only accumulate."""
    t_early, t_late = frame.index[50], frame.index[250]
    times = np.array([t_early - HOUR, t_early], dtype=np.int64)
    got = news_coverage(frame, times, ["alpha", "beta"], cfg)

    assert got.loc[t_early, "news_breadth_24h"] == 2
    assert got.loc[t_late, "news_breadth_24h"] == 0, (
        "publishers must leave the window as it slides forward")


# --------------------------------------------------------------------------
# absence
# --------------------------------------------------------------------------
def test_no_coverage_is_nan_hours_and_zero_counts(cfg, frame):
    """"Never covered" is not "covered infinitely long ago"."""
    got = news_coverage(frame, np.array([], dtype=np.int64), [], cfg)
    assert got["hours_since_news"].isna().all()
    assert (got["news_count_24h"] == 0).all()
    assert (got["news_breadth_168h"] == 0).all()


# --------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------
def test_unsorted_article_times_are_refused(cfg, frame):
    with pytest.raises(ValueError, match="sorted ascending"):
        news_coverage(frame, np.array([BASE + 10, BASE], dtype=np.int64),
                      ["a", "b"], cfg)


def test_publishers_must_be_parallel_to_the_times(cfg, frame):
    with pytest.raises(ValueError, match="parallel"):
        news_coverage(frame, np.array([BASE, BASE + HOUR], dtype=np.int64),
                      ["only-one"], cfg)


# --------------------------------------------------------------------------
# leakage — the same experiment P2-08 built, applied to this builder
# --------------------------------------------------------------------------
def _articles_moved(builder, frame, times, pubs, cut, cfg):
    """Columns at or before `cut` that move when ARTICLES after `cut` change.

    `assert_no_lookahead` cannot be reused here, and the reason is worth
    stating: it perturbs the BAR FRAME, and these features never read a bar's
    values — only its index and an external article array. Pointed at this
    builder it would pass no matter what the code did, which is a check that
    cannot fail rather than a clean bill of health. So the perturbation has to
    be applied to the articles instead.
    """
    clean = builder(frame, times, pubs, cfg)
    keep = times <= cut
    rng = np.random.default_rng(3)
    future = np.sort(rng.choice(np.arange(cut + 1, cut + 400 * HOUR), 300,
                                replace=False)).astype(np.int64)
    t2 = np.concatenate([times[keep], future])
    p2 = ([pubs[i] for i in range(int(keep.sum()))]
          + [f"FUTURE{i % 13}" for i in range(future.size)])
    after = builder(frame, t2, p2, cfg)

    past = clean.index <= cut
    return [c for c in clean.columns
            if not clean.loc[past, c].equals(after.loc[past, c])]


def test_news_features_read_no_future(cfg, frame):
    """Adding, removing and reshuffling articles AFTER t moves nothing at t."""
    rng = np.random.default_rng(11)
    times = np.sort(rng.choice(
        np.arange(BASE - 200 * HOUR, BASE + 300 * HOUR), 400, replace=False)
    ).astype(np.int64)
    pubs = [f"p{i % 7}" for i in range(times.size)]

    moved = _articles_moved(news_coverage, frame, times, pubs,
                            int(frame.index[200]), cfg)
    assert not moved, f"these read the future: {moved}"


def test_the_article_perturbation_is_capable_of_failing(cfg, frame):
    """Guard on the guard, twice over.

    A centred window is leakage by construction, so the same perturbation must
    catch it — otherwise the test above proves nothing.
    """
    rng = np.random.default_rng(11)
    times = np.sort(rng.choice(
        np.arange(BASE - 200 * HOUR, BASE + 300 * HOUR), 400, replace=False)
    ).astype(np.int64)
    pubs = [f"p{i % 7}" for i in range(times.size)]

    def leaky(f, ts, ps, c):
        stamps = np.asarray(f.index, dtype=np.int64)
        hi = np.searchsorted(ts, stamps + 24 * HOUR, side="right")
        lo = np.searchsorted(ts, stamps - 24 * HOUR, side="right")
        return pd.DataFrame({"news_count_24h": hi - lo}, index=f.index)

    moved = _articles_moved(leaky, frame, times, pubs,
                            int(frame.index[200]), cfg)
    assert moved, "a centred window is leakage and the check must say so"


# --------------------------------------------------------------------------
# the ablation switch
# --------------------------------------------------------------------------
def test_the_flag_governs_whether_the_columns_exist_at_all(frame):
    """P8-02 flips one config flag; nothing else may differ between arms."""
    bench = frame.copy()
    empty = np.array([], dtype=np.int64)

    off = load_config()
    off["features"]["include_news_coverage"] = False
    on = load_config()
    on["features"]["include_news_coverage"] = True

    cols_off = set(ticker_features(frame, bench, empty, empty, off).columns)
    cols_on = set(ticker_features(frame, bench, empty, empty, on).columns)

    news_cols = {c for c in cols_on if c.startswith("news_") or c == "hours_since_news"}
    assert news_cols, "the flag did not add any news column"
    assert not (cols_off & news_cols), "news columns leaked into the without arm"
    assert cols_on - news_cols == cols_off, (
        "the two arms must differ ONLY by the news columns")
