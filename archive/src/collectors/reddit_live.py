"""Live Reddit collector — polls the Arctic Shift API (plan §5.2, revised).

Reddit's own channels are unavailable for this project: the research/commercial
API application was rejected, and unauthenticated `.json` scraping has
returned 403 on every host (www./old./oauth.) since Reddit tightened
unauthenticated access in 2026. Arctic Shift (arctic-shift.photon-reddit.com)
is an independent, continuously-updated mirror — not a Reddit-owned channel —
observed at ~10 minute ingestion lag on busy subreddits, which is inside the
finest checkpoint (5 min) used by the early-rumor-detection literature's own
definition of "real-time." PullPush.io is used as a fallback when Arctic
Shift is unreachable; it shares the same flat record schema.

Non-negotiable rules implemented here:
  - every poll cycle the raw search-result payload is snapshotted to
    data/raw/live_json/ BEFORE any filtering (the DB only keeps ticker-bearing
    posts)
  - polling is incremental per subreddit: the max created_utc already stored
    for that subreddit is the low-water mark; each cycle asks for
    after=<that mark>, sort=asc
  - exponential backoff on non-200/error responses; after repeated failure
    on Arctic Shift for a subreddit, that cycle falls back to PullPush.io
  - posts are re-fetched once at t0+6h and t0+24h (via a narrow windowed
    search matched by id — Arctic Shift has no by-id lookup) to capture
    score/comment growth into score_6h / score_24h. Caveat: Arctic Shift
    itself only re-scrapes a post around t0+36h (observed), so these columns
    may reflect the same value as `score` until that has happened upstream.

Usage:
  python -m src.collectors.reddit_live           # loop forever, poll every 5 min
  python -m src.collectors.reddit_live --once    # single cycle (testing)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests

from src import db
from src.collectors.arctic_shift import record_to_post
from src.pipeline.tickers import TickerExtractor
from src.utils.config import load_config
from src.utils.ratelimit import Backoff, RateLimiter
from src.utils.timeutils import utc_now, utc_now_ts

log = logging.getLogger(__name__)


class LivePoller:
    def __init__(self, cfg: dict, conn):
        self.cfg = cfg
        self.conn = conn
        self.extractor = TickerExtractor.from_config(cfg)
        self.subreddits = cfg["subreddits"]

        acfg = cfg["arctic_shift"]
        self.primary_url = f"{acfg['api_base']}/posts/search"
        self.page_limit = acfg["page_limit"]
        self.limiter = RateLimiter(acfg["min_interval_s"])

        pcfg = cfg.get("pullpush") or {}
        self.fallback_url = pcfg.get("api_base")
        self.fallback_limiter = RateLimiter(pcfg.get("min_interval_s", 4.0))

        self.backoff = Backoff(base_s=15)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg["reddit"]["user_agent"]
        self.snapshot_dir = Path(cfg["paths"]["live_json"])
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.lookback_s = cfg["reddit"].get("live_lookback_s", 900)
        self._high_water = {s: self._init_high_water(s) for s in self.subreddits}

    def _init_high_water(self, subreddit: str) -> int:
        row = self.conn.execute(
            "SELECT MAX(created_utc) FROM posts WHERE subreddit = ? AND source = 'live'",
            (subreddit,),
        ).fetchone()
        if row and row[0]:
            return row[0]
        return utc_now_ts() - self.lookback_s

    # -- HTTP ----------------------------------------------------------
    def _search(self, base_url: str, limiter: RateLimiter, subreddit: str,
                after: int, before: int | None = None) -> list[dict] | None:
        """One rate-limited search against a flat-schema API; None on failure."""
        params = {"subreddit": subreddit, "after": after,
                  "limit": self.page_limit, "sort": "asc"}
        if before is not None:
            params["before"] = before
        for _ in range(4):
            limiter.wait()
            try:
                resp = self.session.get(base_url, params=params, timeout=30)
                if resp.status_code != 200:
                    self.backoff.sleep(f"HTTP {resp.status_code} from {base_url}")
                    continue
                payload = resp.json()
                if payload.get("error"):
                    self.backoff.sleep(f"API error: {payload['error']}")
                    continue
                self.backoff.reset()
                return payload.get("data") or []
            except requests.RequestException as exc:
                self.backoff.sleep(f"request failed: {exc}")
        return None

    def _fetch(self, subreddit: str, after: int, before: int | None = None) -> list[dict]:
        records = self._search(self.primary_url, self.limiter, subreddit, after, before)
        if records is not None:
            return records
        if not self.fallback_url:
            return []
        log.warning("Arctic Shift unreachable for r/%s; falling back to PullPush", subreddit)
        records = self._search(self.fallback_url, self.fallback_limiter, subreddit, after, before)
        return records or []

    # -- poll cycle ------------------------------------------------------
    def poll_once(self) -> int:
        """One cycle: per subreddit, fetch since the high-water mark, snapshot,
        filter by ticker, insert; then run any due score refetches."""
        now = utc_now_ts()
        total_fresh = 0
        for subreddit in self.subreddits:
            records = self._fetch(subreddit, self._high_water[subreddit])
            if not records:
                continue

            # Archive the raw payload before any filtering — you will thank yourself
            stamp = utc_now().strftime("%Y%m%dT%H%M%S_%fZ")
            (self.snapshot_dir / f"{stamp}_{subreddit}.json").write_text(
                json.dumps(records), encoding="utf-8"
            )

            posts = [p for p in (record_to_post(r) for r in records) if p]
            for post in posts:
                post["source"] = "live"
                post["fetched_utc"] = now
            self._high_water[subreddit] = max(
                [self._high_water[subreddit]] + [p["created_utc"] for p in posts]
            )

            known = db.known_post_ids(self.conn, [p["id"] for p in posts])
            fresh, links = [], []
            for post in posts:
                if post["id"] in known:
                    continue
                tickers = self.extractor.extract(f"{post['title']} {post['selftext']}")
                if not tickers:
                    continue
                fresh.append(post)
                links.extend((post["id"], t) for t in tickers)
            db.upsert_posts(self.conn, fresh)
            db.link_post_tickers(self.conn, links)
            total_fresh += len(fresh)

        n_refetched = self.refetch_due(now)
        log.info("cycle: %d new ticker posts across %d subs, %d snapshots refetched",
                  total_fresh, len(self.subreddits), n_refetched)
        return total_fresh

    # -- delayed engagement snapshots -------------------------------------
    def refetch_due(self, now: int) -> int:
        """Fill score_6h / score_24h for live posts whose window has arrived.
        Arctic Shift has no by-id lookup, so this re-queries a narrow
        [min_created-1, max_created+1] window per subreddit and matches by id."""
        refetched = 0
        hours_6, hours_24 = self.cfg["reddit"]["refetch_hours"]
        for column, hours in (("score_6h", hours_6), ("score_24h", hours_24)):
            rows = self.conn.execute(
                f"""SELECT id, subreddit, created_utc FROM posts
                    WHERE source='live' AND {column} IS NULL AND created_utc <= ?
                    LIMIT ?""",
                (now - hours * 3600, self.page_limit),
            ).fetchall()
            if not rows:
                continue
            by_sub: dict[str, list[str]] = {}
            windows: dict[str, list[int]] = {}
            for pid, sub, created in rows:
                by_sub.setdefault(sub, []).append(pid)
                windows.setdefault(sub, [created, created])
                windows[sub][0] = min(windows[sub][0], created)
                windows[sub][1] = max(windows[sub][1], created)
            for sub, pids in by_sub.items():
                lo, hi = windows[sub]
                records = self._fetch(sub, lo - 1, hi + 1)
                scores = {r["id"]: r.get("score", 0) for r in records if r.get("id")}
                for pid in pids:
                    db.set_post_score_snapshot(self.conn, pid, column, scores.get(pid, 0))
                    refetched += 1
        return refetched

    # -- main loop ---------------------------------------------------------
    def run(self) -> None:
        interval = self.cfg["reddit"]["live_poll_seconds"]
        log.info("Live poller started: %d subs via Arctic Shift, every %ds",
                  len(self.subreddits), interval)
        while True:
            start = time.monotonic()
            try:
                self.poll_once()
            except Exception:
                log.exception("poll cycle failed; continuing")
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, interval - elapsed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run a single cycle")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    poller = LivePoller(cfg, conn)
    if args.once:
        poller.poll_once()
    else:
        poller.run()


if __name__ == "__main__":
    main()
