"""Rebuild post_tickers for every stored post under the current rules.

The collectors write ticker links as they ingest, so links reflect whatever
`config/tickers.csv` looked like at collection time. Whenever the universe is
rebuilt (`python -m src.pipeline.build_universe`) the stored links go stale —
this script re-runs extraction over the posts already in SQLite and replaces
them, so the DB always matches the current config.

Posts that no longer map to any ticker are deleted by default: the posts table
is defined as "posts that mention at least one ticker" (plan §5.1), and after a
prune the leftovers are exactly the false positives that motivated it.

Usage:
  python -m src.pipeline.relink [--keep-orphans] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging

from src import db
from src.pipeline.tickers import TickerExtractor
from src.utils.config import load_config

log = logging.getLogger(__name__)

BATCH_SIZE = 20_000


def relink(conn, extractor: TickerExtractor, dry_run: bool = False) -> dict[str, int]:
    """Recompute every post's tickers. Returns before/after counts."""
    stats = {
        "posts": conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0],
        "links_before": conn.execute("SELECT COUNT(*) FROM post_tickers").fetchone()[0],
    }
    links: list[tuple[str, str]] = []
    matched = 0
    for offset in range(0, stats["posts"], BATCH_SIZE):
        rows = conn.execute(
            "SELECT id, title, selftext FROM posts ORDER BY id LIMIT ? OFFSET ?",
            (BATCH_SIZE, offset),
        ).fetchall()
        for post_id, title, selftext in rows:
            tickers = extractor.extract(f"{title or ''} {selftext or ''}")
            if tickers:
                matched += 1
                links.extend((post_id, t) for t in tickers)
        log.info("extracted %d/%d posts", min(offset + BATCH_SIZE, stats["posts"]),
                 stats["posts"])

    stats["links_after"] = len(links)
    stats["orphans"] = stats["posts"] - matched
    if dry_run:
        return stats

    with conn:  # one transaction: never leave the DB half-relinked
        conn.execute("DELETE FROM post_tickers")
        conn.executemany(
            "INSERT OR IGNORE INTO post_tickers (post_id, ticker) VALUES (?, ?)",
            links,
        )
    return stats


def delete_orphans(conn) -> int:
    with conn:
        cur = conn.execute(
            "DELETE FROM posts WHERE id NOT IN (SELECT post_id FROM post_tickers)"
        )
    return cur.rowcount


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-orphans", action="store_true",
                        help="keep posts that no longer map to any ticker")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    stats = relink(conn, TickerExtractor.from_config(cfg), dry_run=args.dry_run)

    deleted = 0
    if not args.dry_run and not args.keep_orphans:
        deleted = delete_orphans(conn)

    log.info(
        "%s%d posts: links %d -> %d, %d posts now match nothing%s",
        "DRY RUN — " if args.dry_run else "",
        stats["posts"], stats["links_before"], stats["links_after"],
        stats["orphans"],
        f" ({deleted} deleted)" if deleted else "",
    )


if __name__ == "__main__":
    main()
