"""Event construction — clusters rumor-candidate posts into events (plan §6.3).

This implements the deterministic clustering rule from the plan: same
ticker, posts within a rolling `gap_hours` window of each other are the same
event. The rumor pre-filter is the keyword step from plan §6.2 (step 1);
the LLM triage refinement (§6.2 step 2, e.g. a real `claim_summary` and a
model-picked `claim_type`) isn't wired in yet — here `claim_summary` is just
the earliest post's title and `claim_type` is the first keyword it matched.
That's enough to get real, working events now; upgrading the claim
extraction later doesn't require touching the schema or this clustering.

Usage:
  python -m src.pipeline.events
"""

from __future__ import annotations

import argparse
import logging

from src import db
from src.utils.config import load_config

log = logging.getLogger(__name__)


def matched_keyword(text: str, keywords: list[str]) -> str | None:
    lowered = text.lower()
    for kw in keywords:
        if kw.lower() in lowered:
            return kw
    return None


def is_rumor_candidate(post: dict, keywords: list[str]) -> str | None:
    """Returns the matched keyword (used as claim_type) or None."""
    text = f"{post['title']} {post['selftext'] or ''}"
    return matched_keyword(text, keywords)


def cluster_ticker_events(
    posts: list[dict], gap_hours: int, min_posts: int, min_solo_score: int,
) -> list[list[dict]]:
    """posts: rumor-candidate posts for ONE ticker, each already carrying a
    'claim_type' key, sorted by created_utc ascending. A new cluster starts
    after `gap_hours` of silence on that ticker."""
    if not posts:
        return []
    gap_s = gap_hours * 3600
    clusters, cluster = [], [posts[0]]
    for post in posts[1:]:
        if post["created_utc"] - cluster[-1]["created_utc"] > gap_s:
            clusters.append(cluster)
            cluster = [post]
        else:
            cluster.append(post)
    clusters.append(cluster)

    return [c for c in clusters if len(c) >= min_posts or c[0]["score"] >= min_solo_score]


def build_events(conn, cfg: dict) -> int:
    """Scans every ticker's posts, clusters rumor candidates into events,
    and inserts any not already stored. Returns the number newly inserted."""
    ecfg, keywords = cfg["event"], cfg["rumor_keywords"]
    n_new = 0
    for ticker in db.distinct_post_tickers(conn):
        rows = conn.execute(
            """SELECT p.* FROM posts p JOIN post_tickers pt ON pt.post_id = p.id
               WHERE pt.ticker = ? ORDER BY p.created_utc ASC""",
            (ticker,),
        ).fetchall()

        candidates = []
        for row in rows:
            post = dict(row)
            claim_type = is_rumor_candidate(post, keywords)
            if claim_type is None:
                continue
            post["claim_type"] = claim_type
            candidates.append(post)

        clusters = cluster_ticker_events(
            candidates, ecfg["gap_hours"], ecfg["min_posts"], ecfg["min_solo_score"]
        )
        for cluster in clusters:
            t0 = cluster[0]["created_utc"]
            event = {
                "event_id": f"{ticker}_{t0}",
                "ticker": ticker,
                "t0_utc": t0,
                "claim_summary": cluster[0]["title"][:280],
                "claim_type": cluster[0]["claim_type"],
                "post_ids": ",".join(p["id"] for p in cluster),
                "label": None,
                "t_official_utc": None,
                "label_source": None,
                "human_reviewed": 0,
            }
            n_new += db.upsert_event(conn, event)
    return n_new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    n_new = build_events(conn, cfg)
    total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    log.info("Done. %d new events; events table now has %d rows.", n_new, total)


if __name__ == "__main__":
    main()
