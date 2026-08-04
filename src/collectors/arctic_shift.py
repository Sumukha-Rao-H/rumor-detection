"""Historical Reddit collector — Arctic Shift (plan §5.1). Replaces PRAW entirely.

Two modes:
  dumps  — stream dump files from data/raw/arctic_dumps/: .zst (newline-delimited
           JSON inside zstandard, never fully decompressed), .jsonl and .jsonl.gz
           as produced by the Arctic Shift download tool.
  api    — Arctic Shift REST API, paginating by created_utc. Free community
           service: throttled to <=1 req/sec with exponential backoff.

Every record is passed through tickers.extract(); only posts that map to at
least one ticker are inserted (posts + post_tickers tables, source='arctic').

Usage:
  python -m src.collectors.arctic_shift --mode dumps
  python -m src.collectors.arctic_shift --mode api --start 2025-01-01 --end 2025-02-01
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path

import requests
import zstandard as zstd

from src import db
from src.pipeline.tickers import TickerExtractor
from src.utils.config import load_config
from src.utils.ratelimit import Backoff, RateLimiter
from src.utils.timeutils import date_str_to_ts, ts_to_iso, utc_now_ts

log = logging.getLogger(__name__)

BATCH_SIZE = 5000  # posts per DB flush in dump mode
PROGRESS_EVERY = 500_000  # lines between progress logs in dump mode
DUMP_PATTERNS = ("*.zst", "*.jsonl", "*.jsonl.gz", "*.ndjson")

# Cheap pre-parse timestamp probe: full-history dumps are mostly outside the
# backtest window, and json.loads on every one of ~10M lines is the bottleneck.
CREATED_UTC_RE = re.compile(rb'"created_utc"\s*:\s*"?(\d+)')


def iter_raw_lines(path: str | Path) -> Iterator[bytes]:
    """Yield raw newline-delimited JSON lines from a .zst / .gz / plain dump.

    Compressed dumps are streamed, never fully decompressed to disk or memory.
    """
    name = str(path)
    if name.endswith(".zst"):
        with open(path, "rb") as fh:
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(fh) as reader:
                yield from io.BufferedReader(reader, buffer_size=1 << 20)
    elif name.endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            yield from fh
    else:
        with open(path, "rb") as fh:
            yield from fh


def stream_records(path: str | Path) -> Iterator[dict]:
    """Yield parsed JSON records from a dump file, skipping malformed lines."""
    for line in iter_raw_lines(path):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def record_to_post(rec: dict) -> dict | None:
    """Normalize a raw Reddit submission record to a `posts` row."""
    try:
        created = int(float(rec["created_utc"]))
    except (KeyError, TypeError, ValueError):
        return None
    if not rec.get("id"):
        return None
    return {
        "id": rec["id"],
        "subreddit": rec.get("subreddit"),
        "title": rec.get("title") or "",
        "selftext": (rec.get("selftext") or "")[:10000],
        "author": rec.get("author"),
        "created_utc": created,
        "score": rec.get("score", 0),
        "upvote_ratio": rec.get("upvote_ratio"),
        "num_comments": rec.get("num_comments", 0),
        "flair": rec.get("link_flair_text"),
        "url": rec.get("url"),
        "source": "arctic",
        "fetched_utc": utc_now_ts(),
        "score_6h": None,
        "score_24h": None,
    }


class _Ingestor:
    """Shared filter-and-insert path for both modes."""

    def __init__(self, conn, extractor: TickerExtractor, start_ts: int, end_ts: int):
        self.conn = conn
        self.extractor = extractor
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.seen = 0
        self.kept = 0
        self.out_of_window = 0
        self._posts: list[dict] = []
        self._links: list[tuple[str, str]] = []

    def offer_raw(self, line: bytes) -> None:
        """Dump-mode entry point: reject out-of-window lines without parsing JSON."""
        match = CREATED_UTC_RE.search(line)
        if match and not (self.start_ts <= int(match.group(1)) < self.end_ts):
            self.seen += 1
            self.out_of_window += 1
            self._log_progress()
            return
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            self.seen += 1
            return
        self.offer(rec)

    def offer(self, rec: dict) -> None:
        self.seen += 1
        self._log_progress()
        post = record_to_post(rec)
        if post is None or not (self.start_ts <= post["created_utc"] < self.end_ts):
            return
        tickers = self.extractor.extract(f"{post['title']} {post['selftext']}")
        if not tickers:
            return
        self.kept += 1
        self._posts.append(post)
        self._links.extend((post["id"], t) for t in tickers)
        if len(self._posts) >= BATCH_SIZE:
            self.flush()

    def _log_progress(self) -> None:
        if self.seen % PROGRESS_EVERY == 0:
            log.info("progress: %d records seen, %d kept", self.seen, self.kept)

    def flush(self) -> None:
        if self._posts:
            db.upsert_posts(self.conn, self._posts)
            db.link_post_tickers(self.conn, self._links)
            self._posts, self._links = [], []
        log.info("progress: %d records seen, %d kept", self.seen, self.kept)


def find_dumps(dump_dir: str | Path) -> list[Path]:
    """Dump files in `dump_dir`, any supported format, deduped and sorted."""
    dump_dir = Path(dump_dir)
    found = {p for pattern in DUMP_PATTERNS for p in dump_dir.glob(pattern)}
    return sorted(found)


def ingest_dumps(cfg: dict, conn, ingestor: _Ingestor) -> None:
    dump_dir = Path(cfg["paths"]["arctic_dumps"])
    files = find_dumps(dump_dir)
    if not files:
        log.error("No dumps (%s) in %s — download per-subreddit dumps from "
                  "https://arctic-shift.photon-reddit.com/download-tool",
                  "/".join(DUMP_PATTERNS), dump_dir)
        return
    for path in files:
        before_seen, before_kept = ingestor.seen, ingestor.kept
        log.info("Streaming %s (%.1f GB) ...", path.name,
                 path.stat().st_size / 1e9)
        for line in iter_raw_lines(path):
            if line.strip():
                ingestor.offer_raw(line)
        ingestor.flush()
        log.info("%s: %d records scanned, %d kept", path.name,
                 ingestor.seen - before_seen, ingestor.kept - before_kept)


def ingest_api(cfg: dict, conn, ingestor: _Ingestor, subreddits: list[str]) -> None:
    acfg = cfg["arctic_shift"]
    limiter = RateLimiter(acfg["min_interval_s"])
    session = requests.Session()
    session.headers["User-Agent"] = cfg["reddit"]["user_agent"]

    for subreddit in subreddits:
        log.info("API: r/%s from %s to %s", subreddit,
                 ts_to_iso(ingestor.start_ts), ts_to_iso(ingestor.end_ts))
        after = ingestor.start_ts
        backoff = Backoff(base_s=15)
        while True:
            limiter.wait()
            try:
                resp = session.get(
                    f"{acfg['api_base']}/posts/search",
                    params={
                        "subreddit": subreddit,
                        "after": after,
                        "before": ingestor.end_ts,
                        "limit": acfg["page_limit"],
                        "sort": "asc",
                    },
                    timeout=90,
                )
                # 422/429/5xx from this API are typically transient overload
                if resp.status_code != 200:
                    backoff.sleep(f"HTTP {resp.status_code} from Arctic Shift API")
                    continue
                payload = resp.json()
                if payload.get("error"):  # in-body errors come with HTTP 200
                    backoff.sleep(f"API error: {payload['error']}")
                    continue
                records = payload.get("data") or []
            except requests.RequestException as exc:
                backoff.sleep(f"request failed: {exc}")
                continue
            backoff.reset()
            if not records:
                break
            for rec in records:
                ingestor.offer(rec)
            last_created = int(float(records[-1]["created_utc"]))
            if last_created <= after:  # no forward progress — stop
                break
            after = last_created
        ingestor.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["dumps", "api"], required=True)
    parser.add_argument("--start", help="YYYY-MM-DD (default: backtest_window.start)")
    parser.add_argument("--end", help="YYYY-MM-DD (default: backtest_window.end)")
    parser.add_argument("--subreddits", help="comma-separated override (api mode)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    start_ts = date_str_to_ts(args.start or cfg["backtest_window"]["start"])
    end_ts = date_str_to_ts(args.end or cfg["backtest_window"]["end"])

    conn = db.get_conn(cfg["paths"]["db"])
    extractor = TickerExtractor.from_config(cfg)
    ingestor = _Ingestor(conn, extractor, start_ts, end_ts)

    if args.mode == "dumps":
        ingest_dumps(cfg, conn, ingestor)
    else:
        subs = (args.subreddits.split(",") if args.subreddits else cfg["subreddits"])
        ingest_api(cfg, conn, ingestor, subs)

    ingestor.flush()
    total = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    log.info("Done. %d records scanned (%d outside the backtest window), "
             "%d kept this run; posts table now has %d rows.",
             ingestor.seen, ingestor.out_of_window, ingestor.kept, total)


if __name__ == "__main__":
    main()
