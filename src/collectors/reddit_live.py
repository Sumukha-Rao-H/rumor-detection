"""Live Reddit collector — public .json endpoints, no auth (plan §5.2).

Non-negotiable rules implemented here:
  - custom User-Agent (default python-requests UA gets blocked instantly)
  - >=7s between requests (config reddit.live_min_interval_s)
  - exponential backoff on 429/403: 60s, 120s, 240s, ... never hammer
  - every poll cycle the raw listing JSON is snapshotted to data/raw/live_json/
    BEFORE any filtering (the DB only keeps ticker-bearing posts)
  - posts are re-fetched once at t0+6h and t0+24h (batched via /api/info.json)
    to capture score/comment growth into score_6h / score_24h

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
from src.pipeline.tickers import TickerExtractor
from src.utils.config import load_config
from src.utils.ratelimit import Backoff, RateLimiter
from src.utils.timeutils import utc_now, utc_now_ts

log = logging.getLogger(__name__)

INFO_BATCH = 100  # /api/info.json accepts up to 100 fullnames


def listing_to_posts(listing: dict, fetched_utc: int) -> list[dict]:
    """Normalize a Reddit listing payload to `posts` rows (source='live')."""
    posts = []
    for child in listing.get("data", {}).get("children", []):
        rec = child.get("data", {})
        if not rec.get("id"):
            continue
        posts.append({
            "id": rec["id"],
            "subreddit": rec.get("subreddit"),
            "title": rec.get("title") or "",
            "selftext": (rec.get("selftext") or "")[:10000],
            "author": rec.get("author"),
            "created_utc": int(float(rec.get("created_utc", 0))),
            "score": rec.get("score", 0),
            "upvote_ratio": rec.get("upvote_ratio"),
            "num_comments": rec.get("num_comments", 0),
            "flair": rec.get("link_flair_text"),
            "url": rec.get("url"),
            "source": "live",
            "fetched_utc": fetched_utc,
            "score_6h": None,
            "score_24h": None,
        })
    return posts


class LivePoller:
    def __init__(self, cfg: dict, conn):
        self.cfg = cfg
        self.conn = conn
        self.extractor = TickerExtractor.from_config(cfg)
        rcfg = cfg["reddit"]
        self.limiter = RateLimiter(rcfg["live_min_interval_s"])
        self.backoff = Backoff(base_s=60)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = rcfg["user_agent"]
        self.snapshot_dir = Path(cfg["paths"]["live_json"])
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.multi_sub = "+".join(cfg["subreddits"])
        self.base_urls = rcfg["base_urls"]
        self._base_i = 0

    @property
    def base_url(self) -> str:
        return self.base_urls[self._base_i]

    # -- HTTP ----------------------------------------------------------
    def _get(self, url: str, params: dict) -> dict | None:
        """One rate-limited GET with backoff; None if it keeps failing."""
        for _ in range(4):
            self.limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code in (429, 403):
                    self.backoff.sleep(f"HTTP {resp.status_code} from reddit")
                    continue
                resp.raise_for_status()
                self.backoff.reset()
                return resp.json()
            except requests.RequestException as exc:
                self.backoff.sleep(f"reddit request failed: {exc}")
        # Persistent failure: rotate to the next base host (www -> old.reddit)
        self._base_i = (self._base_i + 1) % len(self.base_urls)
        log.error("Giving up on %s; rotating base host to %s", url, self.base_url)
        return None

    # -- poll cycle ------------------------------------------------------
    def poll_once(self) -> int:
        """One cycle: fetch multi-sub new.json, snapshot, insert, refetch due."""
        now = utc_now_ts()
        listing = self._get(
            f"{self.base_url}/r/{self.multi_sub}/new.json",
            {"limit": self.cfg["reddit"]["listing_limit"], "raw_json": 1},
        )
        if listing is None:
            return 0

        # Archive the raw payload before any filtering — you will thank yourself
        stamp = utc_now().strftime("%Y%m%dT%H%M%S_%fZ")
        (self.snapshot_dir / f"{stamp}_new.json").write_text(
            json.dumps(listing), encoding="utf-8"
        )

        posts = listing_to_posts(listing, fetched_utc=now)
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

        n_refetched = self.refetch_due(now)
        log.info("cycle: %d listed, %d new ticker posts, %d snapshots refetched",
                 len(posts), len(fresh), n_refetched)
        return len(fresh)

    # -- delayed engagement snapshots -------------------------------------
    def refetch_due(self, now: int) -> int:
        """Fill score_6h / score_24h for live posts whose window has arrived."""
        refetched = 0
        hours_6, hours_24 = self.cfg["reddit"]["refetch_hours"]
        for column, hours in (("score_6h", hours_6), ("score_24h", hours_24)):
            rows = self.conn.execute(
                f"""SELECT id FROM posts WHERE source='live' AND {column} IS NULL
                    AND created_utc <= ? LIMIT ?""",
                (now - hours * 3600, INFO_BATCH),
            ).fetchall()
            if not rows:
                continue
            fullnames = ",".join(f"t3_{r[0]}" for r in rows)
            payload = self._get(f"{self.base_url}/api/info.json",
                                {"id": fullnames, "raw_json": 1})
            if payload is None:
                continue
            scores = {
                c["data"]["id"]: c["data"].get("score", 0)
                for c in payload.get("data", {}).get("children", [])
            }
            for (pid,) in rows:
                # Deleted/removed posts: record 0 so they aren't re-queried forever
                db.set_post_score_snapshot(self.conn, pid, column,
                                           scores.get(pid, 0))
                refetched += 1
        return refetched

    # -- main loop ---------------------------------------------------------
    def run(self) -> None:
        interval = self.cfg["reddit"]["live_poll_seconds"]
        log.info("Live poller started: r/%s every %ds", self.multi_sub, interval)
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
