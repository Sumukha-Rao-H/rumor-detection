"""LLM rumor triage — plan §6.2 stage 2, prompt from Appendix D.1.

The keyword pre-filter in events.py is deliberately high-recall: it lets
through price-target chatter, memes and questions that merely contain the word
"merger". This pass asks a model, once per (post, ticker) pair, whether the post
states a *checkable* claim about that company, and extracts a one-sentence
claim summary and a claim type for the event that contains it.

Design notes:

  budget    only posts inside kept events are judged (~11k pairs, not the whole
            corpus), ordered so the events most likely to survive labeling come
            first: biggest events, then highest-scoring. Free-tier quota runs
            out mid-job by design — the run is resumable and every answer is
            cached, so tomorrow's run continues rather than restarts.
  events    an event's claim_summary/claim_type come from its *seed post*: the
            highest-scoring post the model called a rumor. n_rumor_posts records
            how many of its posts qualified, so §6.4 can skip the 0s.
  cost      re-running after a prompt change is opt-in via llm.prompt_version.

Usage:
  python -m src.pipeline.triage --limit 50        # pilot
  python -m src.pipeline.triage                   # everything outstanding
  python -m src.pipeline.triage --events-only     # just refresh event rollups
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass

from src import db
from src.utils.config import load_config
from src.utils.llm import LLMClient, LLMError
from src.utils.timeutils import utc_now_ts

log = logging.getLogger(__name__)

CLAIM_TYPES = ("merger", "bankruptcy", "regulatory", "earnings", "contract",
               "legal", "offering", "other")
BODY_CHARS = 1500  # Appendix D.1: selftext[:1500]

PROMPT = """You are labeling Reddit posts about stocks for a research dataset.
A "rumor" is an unverified factual claim about a specific company that could \
later be confirmed or denied by official news (merger, acquisition, bankruptcy, \
FDA decision, earnings leak, major contract, investigation, delisting, offering).
NOT rumors: opinions, price predictions, memes, questions, technical analysis, \
general DD without a specific checkable claim.

POST TITLE: {title}
POST BODY: {body}
TICKER: {ticker}

Respond ONLY with JSON:
{{"is_rumor": true/false,
 "claim_summary": "one sentence stating the checkable claim, or empty",
 "claim_type": "merger|bankruptcy|regulatory|earnings|contract|legal|offering|other"}}"""


@dataclass
class Pair:
    """A (post, ticker) pair awaiting judgement, with its event context."""
    post_id: str
    ticker: str
    title: str
    selftext: str
    score: int
    event_id: str
    n_posts: int


def build_prompt(pair: Pair) -> str:
    return PROMPT.format(title=pair.title or "",
                         body=(pair.selftext or "")[:BODY_CHARS],
                         ticker=pair.ticker)


def content_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()


def pending_pairs(conn, cfg: dict, prompt_version: str) -> list[Pair]:
    """Untriaged (post, ticker) pairs inside kept events, most promising first.

    Ordered *seed-first*: every event's highest-scoring post comes before any
    event's second post. An event only needs one rumor post to become usable,
    so this maximises the number of distinct events covered by any prefix of
    the run — which matters because free-tier quota decides where the run stops.
    """
    done = db.triaged_pairs(conn, prompt_version)
    rows = conn.execute(
        "SELECT event_id, ticker, n_posts, post_ids FROM events "
        "ORDER BY n_posts DESC, t0_utc"
    ).fetchall()

    by_rank: dict[int, list[Pair]] = {}
    for event_id, ticker, n_posts, post_ids in rows:
        ids = json.loads(post_ids)
        placeholders = ",".join("?" * len(ids))
        posts = conn.execute(
            f"SELECT id, title, selftext, score FROM posts WHERE id IN ({placeholders})"
            " ORDER BY score DESC, id",
            ids,
        ).fetchall()
        for rank, (post_id, title, selftext, score) in enumerate(posts):
            if (post_id, ticker) not in done:
                by_rank.setdefault(rank, []).append(
                    Pair(post_id, ticker, title, selftext, score or 0,
                         event_id, n_posts))
    return [pair for rank in sorted(by_rank) for pair in by_rank[rank]]


def normalize(data: dict) -> dict:
    """Coerce a model reply into the three fields we store."""
    claim_type = str(data.get("claim_type") or "other").strip().lower()
    if claim_type not in CLAIM_TYPES:
        claim_type = "other"
    is_rumor = data.get("is_rumor")
    if isinstance(is_rumor, str):
        is_rumor = is_rumor.strip().lower() in ("true", "yes", "1")
    summary = (data.get("claim_summary") or "").strip()
    return {
        "is_rumor": int(bool(is_rumor)),
        # A "rumor" with no claim is a contradiction; treat it as not a rumor.
        "claim_summary": summary or None,
        "claim_type": claim_type,
    }


def triage(conn, cfg: dict, client: LLMClient, limit: int | None = None) -> dict:
    """Judge outstanding pairs, writing each verdict as it arrives."""
    prompt_version = cfg["llm"]["prompt_version"]
    pairs = pending_pairs(conn, cfg, prompt_version)
    if limit is not None:
        pairs = pairs[:limit]
    log.info("%d pairs to triage under prompt %s", len(pairs), prompt_version)

    stats = Counter()
    for i, pair in enumerate(pairs, 1):
        prompt = build_prompt(pair)
        try:
            # Cache on prompt content, not post id: WSB reposts the same text
            # under many ids (8.5% of pairs), and an identical prompt has an
            # identical answer. Resumability still comes from post_triage rows.
            reply = client.complete_json(prompt, cache_key=content_key(prompt))
        except LLMError as exc:
            log.error("giving up at pair %d/%d: %s", i, len(pairs), exc)
            stats["aborted"] = 1
            break
        fields = normalize(reply.data)
        if fields["is_rumor"] and not fields["claim_summary"]:
            fields["is_rumor"] = 0
        db.upsert_triage(conn, [{
            **fields, "post_id": pair.post_id, "ticker": pair.ticker,
            "provider": reply.provider, "model": reply.model,
            "prompt_version": prompt_version, "created_utc": utc_now_ts(),
        }])
        stats["done"] += 1
        stats["rumor"] += fields["is_rumor"]
        stats[f"type:{fields['claim_type']}"] += fields["is_rumor"]
        if i % 25 == 0:
            log.info("%d/%d judged (%d rumors, %d cached, %d api calls)",
                     i, len(pairs), stats["rumor"], client.cache_hits, client.calls)
    return dict(stats)


def refresh_events(conn) -> dict:
    """Roll triage verdicts up to their events (seed post = top-scoring rumor)."""
    rows = conn.execute("SELECT event_id, ticker, post_ids FROM events").fetchall()
    updates = []
    for event_id, ticker, post_ids in rows:
        ids = json.loads(post_ids)
        placeholders = ",".join("?" * len(ids))
        verdicts = conn.execute(
            f"""
            SELECT t.claim_summary, t.claim_type, p.score
            FROM post_triage t JOIN posts p ON p.id = t.post_id
            WHERE t.ticker = ? AND t.is_rumor = 1 AND t.post_id IN ({placeholders})
            ORDER BY p.score DESC
            """,
            [ticker, *ids],
        ).fetchall()
        if not verdicts:
            updates.append((0, None, None, event_id))
            continue
        summary, claim_type, _ = verdicts[0]
        updates.append((len(verdicts), summary, claim_type, event_id))

    with conn:
        conn.executemany(
            """UPDATE events SET n_rumor_posts = ?, claim_summary = ?,
               claim_type = ? WHERE event_id = ?""",
            updates,
        )
    rumor_events = sum(1 for u in updates if u[0])
    return {"events": len(updates), "rumor_events": rumor_events}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="judge at most N pairs")
    parser.add_argument("--events-only", action="store_true",
                        help="skip the API entirely, just refresh event rollups")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])

    if not args.events_only:
        client = LLMClient(cfg)
        stats = triage(conn, cfg, client, limit=args.limit)
        log.info("triaged %d pairs: %d rumors (%.0f%%), %d api calls, %d cached",
                 stats.get("done", 0), stats.get("rumor", 0),
                 100 * stats.get("rumor", 0) / max(stats.get("done", 0), 1),
                 client.calls, client.cache_hits)
        types = {k[5:]: v for k, v in stats.items() if k.startswith("type:")}
        if types:
            log.info("claim types: %s", ", ".join(
                f"{k}={v}" for k, v in sorted(types.items(), key=lambda x: -x[1])))

    rollup = refresh_events(conn)
    log.info("events: %d total, %d with >=1 rumor post", rollup["events"],
             rollup["rumor_events"])


if __name__ == "__main__":
    main()
