"""Per-event state tensors — plan §7. The last step before the RL env.

Each event becomes `X ∈ ℝ^(T×D)`, one row per hourly step from t0, saved to
`data/processed/states/{event_id}.npz`. The env then just indexes the array, so
training does no NLP and no SQL at all.

    text     395  MiniLM embedding of title + claim (384), FinBERT sentiment
                  (3), claim-type one-hot (8). Static per event, repeated each
                  step — the claim does not change as the hours pass.
    social     8  how the crowd is reacting, recomputed each step
    market    14  what the tape is doing, recomputed each step
    step       1  t / T_max
    -------------
             418

**The rule that matters**: the row for step t is built only from posts and bars
timestamped `<= t0 + t hours`. Labeling is allowed to look at the future (it is
the oracle); features never are. `tests/test_leakage.py` asserts this by
rebuilding a row with the future truncated away and comparing.

Text encoders are frozen and run once here, never fine-tuned — that is what
keeps training feasible on a 3050. If sentence-transformers is not installed
the text block falls back to a hashed bag-of-words of the same width, which
makes the pipeline runnable end to end without a 2 GB download. The fallback is
loud, deterministic, and not a substitute for a real run: it is there so the
env, the training loop and the dashboard can be exercised before anyone waits
on a model download.

Usage:
  python -m src.pipeline.features                 # every event in events.parquet
  python -m src.pipeline.features --limit 20      # a quick subset
  python -m src.pipeline.features --encoder hashed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import db
from src.utils.config import load_config

log = logging.getLogger(__name__)

HOUR = 3600
DAY = 86400

EMBED_DIM = 384
SENT_DIM = 3
CLAIM_TYPES = ("merger", "bankruptcy", "regulatory", "earnings",
               "contract", "legal", "offering", "other")
TEXT_DIM = EMBED_DIM + SENT_DIM + len(CLAIM_TYPES)   # 395
SOCIAL_DIM = 8
MARKET_DIM = 14
STATE_DIM = TEXT_DIM + SOCIAL_DIM + MARKET_DIM + 1   # 418


@dataclass
class Post:
    created_utc: int
    author: str | None
    score: int | None
    upvote_ratio: float | None
    num_comments: int | None


# --------------------------------------------------------------------- text

def _hashed_embedding(text: str, dim: int) -> np.ndarray:
    """Deterministic bag-of-words fallback, unit-normalised like a real one."""
    vec = np.zeros(dim, dtype=np.float32)
    for token in (text or "").lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        slot = int.from_bytes(digest, "big") % dim
        vec[slot] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class TextEncoder:
    """MiniLM + FinBERT, loaded once. Falls back to hashing when unavailable."""

    def __init__(self, cfg: dict, prefer: str = "minilm"):
        self.mode = "hashed"
        self._embed = self._sent = None
        if prefer != "hashed":
            try:
                from sentence_transformers import SentenceTransformer
                from transformers import (AutoModelForSequenceClassification,
                                          AutoTokenizer)
                self._embed = SentenceTransformer(cfg["state"]["embed_model"])
                name = cfg["state"]["sent_model"]
                self._tok = AutoTokenizer.from_pretrained(name)
                self._sent = AutoModelForSequenceClassification.from_pretrained(name)
                self._sent.eval()
                self.mode = "minilm"
            except Exception as exc:            # noqa: BLE001 - any import/download failure
                log.warning("text encoders unavailable (%s) — falling back to "
                            "hashed features. Install sentence-transformers and "
                            "transformers for a real run.", type(exc).__name__)

    def embed(self, text: str) -> np.ndarray:
        if self._embed is None:
            return _hashed_embedding(text, EMBED_DIM)
        vec = self._embed.encode(text or "", convert_to_numpy=True,
                                 normalize_embeddings=True)
        return vec.astype(np.float32)

    def sentiment(self, text: str) -> np.ndarray:
        """(positive, negative, neutral) probabilities."""
        if self._sent is None:
            # A neutral prior: the fallback must not invent a signal.
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        import torch
        with torch.no_grad():
            batch = self._tok(text or "", return_tensors="pt", truncation=True,
                              max_length=256)
            probs = torch.softmax(self._sent(**batch).logits, dim=-1)[0]
        return probs.numpy().astype(np.float32)


def claim_type_onehot(claim_type: str | None) -> np.ndarray:
    vec = np.zeros(len(CLAIM_TYPES), dtype=np.float32)
    key = (claim_type or "other").strip().lower()
    vec[CLAIM_TYPES.index(key) if key in CLAIM_TYPES else len(CLAIM_TYPES) - 1] = 1.0
    return vec


def text_block(encoder: TextEncoder, title: str, claim: str,
               claim_type: str | None) -> np.ndarray:
    text = f"{title or ''} {claim or ''}".strip()
    return np.concatenate([encoder.embed(text), encoder.sentiment(text),
                           claim_type_onehot(claim_type)])


# ------------------------------------------------------------------- social

def social_block(posts: list[Post], t0: int, now: int, t_max: int) -> np.ndarray:
    """Crowd reaction using only posts at or before `now`."""
    seen = [p for p in posts if p.created_utc <= now]
    if not seen:
        return np.zeros(SOCIAL_DIM, dtype=np.float32)

    recent = [p for p in seen if p.created_utc > now - 6 * HOUR]
    hours = max((now - t0) / HOUR, 1.0)
    scores = [p.score or 0 for p in seen]
    ratios = [p.upvote_ratio for p in seen if p.upvote_ratio is not None]
    return np.array([
        math.log1p(len(recent) / 6.0),                       # velocity, posts/hr
        math.log1p(len({p.author for p in seen if p.author})),
        float(np.mean(ratios)) if ratios else 0.0,
        math.log1p(max(sum(scores), 0)),
        math.log1p(sum(p.num_comments or 0 for p in seen) / hours),
        0.0,                        # young-account share: not in the schema
        math.log1p(max(max(scores), 0)),
        (now - t0) / (t_max * HOUR),
    ], dtype=np.float32)


# ------------------------------------------------------------------- market

def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def market_block(bars: list, spy: list, t0: int, now: int) -> np.ndarray:
    """Tape features from bars at or before `now`. Missing data reads as zero."""
    past = [b for b in bars if b["ts_utc"] <= now]
    out = np.zeros(MARKET_DIM, dtype=np.float32)
    if not past:
        return out

    closes = [b["close"] for b in past if b["close"]]
    # 0-5: last six hourly log returns, oldest first.
    rets = [math.log(closes[i] / closes[i - 1])
            for i in range(max(1, len(closes) - 6), len(closes))
            if closes[i - 1] > 0]
    for i, r in enumerate(rets[-6:]):
        out[6 - len(rets[-6:]) + i] = r

    # 6: volume z-score against the trailing 20-day mean for this hour of day.
    hour = (now % DAY) // HOUR
    same_hour = [b["volume"] or 0.0 for b in past
                 if (b["ts_utc"] % DAY) // HOUR == hour
                 and b["ts_utc"] > now - 20 * DAY]
    if len(same_hour) >= 3:
        mean, std = float(np.mean(same_hour[:-1] or same_hour)), float(np.std(same_hour[:-1] or same_hour))
        out[6] = _safe_div(same_hour[-1] - mean, std)

    # 7: realised vol over 24h against a 20-day baseline.
    day_rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
                if closes[i - 1] > 0]
    if len(day_rets) > 24:
        out[7] = _safe_div(float(np.std(day_rets[-24:])), float(np.std(day_rets)))

    # 8: cumulative return since t0.
    at_t0 = [b["close"] for b in past if b["ts_utc"] <= t0 and b["close"]]
    if at_t0 and closes:
        out[8] = _safe_div(closes[-1], at_t0[-1]) - 1.0

    # 9: gap against the previous session's close.
    prev_day = [b["close"] for b in past
                if b["ts_utc"] < now - (now % DAY) and b["close"]]
    if prev_day and closes:
        out[9] = _safe_div(closes[-1], prev_day[-1]) - 1.0

    # 10-11: is the tape live, and how stale is the last print.
    age = now - past[-1]["ts_utc"]
    out[10] = 1.0 if age <= 2 * HOUR else 0.0
    out[11] = min(age / DAY, 5.0)

    # 12: the market control — SPY over the same window.
    spy_past = [b["close"] for b in spy if b["ts_utc"] <= now and b["close"]]
    spy_t0 = [b["close"] for b in spy if b["ts_utc"] <= t0 and b["close"]]
    if spy_past and spy_t0:
        out[12] = _safe_div(spy_past[-1], spy_t0[-1]) - 1.0

    # 13: spread proxy from the last bar.
    last = past[-1]
    if last["close"]:
        out[13] = _safe_div((last["high"] or 0) - (last["low"] or 0), last["close"])
    return out


# ------------------------------------------------------------- assembling it

def event_posts(conn, post_ids: list[str]) -> list[Post]:
    if not post_ids:
        return []
    marks = ",".join("?" * len(post_ids))
    rows = conn.execute(
        f"""SELECT created_utc, author, score, upvote_ratio, num_comments
            FROM posts WHERE id IN ({marks}) ORDER BY created_utc""",
        post_ids).fetchall()
    return [Post(r["created_utc"], r["author"], r["score"],
                 r["upvote_ratio"], r["num_comments"]) for r in rows]


def seed_title(conn, post_ids: list[str]) -> str:
    if not post_ids:
        return ""
    marks = ",".join("?" * len(post_ids))
    row = conn.execute(
        f"SELECT title FROM posts WHERE id IN ({marks}) ORDER BY score DESC LIMIT 1",
        post_ids).fetchone()
    return (row["title"] if row else "") or ""


def build_event(conn, cfg: dict, encoder: TextEncoder, row: pd.Series) -> dict:
    """One event -> (T, 418) matrix plus the metadata the env needs."""
    t_max = int(cfg["reward"]["T_max"])
    t0 = int(row["t0_utc"])
    interval = cfg["market"]["interval"]
    horizon = t0 + t_max * HOUR

    posts = event_posts(conn, list(row["post_ids"]))
    title = seed_title(conn, list(row["post_ids"]))
    text = text_block(encoder, title, row.get("claim_summary") or "",
                      row.get("claim_type"))

    bars = db.bars_in_range(conn, row["ticker"], t0 - 25 * DAY, horizon, interval)
    spy = db.bars_in_range(conn, cfg["market"]["benchmark"],
                           t0 - 25 * DAY, horizon, interval)

    steps = np.zeros((t_max, STATE_DIM), dtype=np.float32)
    for t in range(t_max):
        now = t0 + t * HOUR
        steps[t] = np.concatenate([
            text,
            social_block(posts, t0, now, t_max),
            market_block(bars, spy, t0, now),
            np.array([t / t_max], dtype=np.float32),
        ])
    return {
        "X": steps,
        "label": np.int64(row["label"]),
        "t0_utc": np.int64(t0),
        "t_official_utc": np.int64(row["t_official_utc"] or (t0 + 72 * HOUR)),
        "T": np.int64(t_max),
    }


def write_states(conn, cfg: dict, frame: pd.DataFrame, out_dir: Path,
                 encoder: TextEncoder) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for _, row in frame.iterrows():
        data = build_event(conn, cfg, encoder, row)
        if not np.isfinite(data["X"]).all():
            log.warning("%s produced non-finite features — clipping", row["event_id"])
            data["X"] = np.nan_to_num(data["X"], posinf=0.0, neginf=0.0)
        np.savez_compressed(out_dir / f"{row['event_id']}.npz", **data)
        written += 1
        if written % 50 == 0:
            log.info("  %d/%d events", written, len(frame))
    return written


def fit_scaler(frame: pd.DataFrame, out_dir: Path) -> dict:
    """Mean/std of the non-embedding columns, fit on train events only.

    Fitting on anything wider would leak val/test distribution into training,
    which is the same mistake as a random split wearing different clothes.
    """
    train = frame[frame["split"] == "train"]
    stack = []
    for event_id in train["event_id"]:
        path = out_dir / f"{event_id}.npz"
        if path.exists():
            stack.append(np.load(path)["X"][:, TEXT_DIM:])
    if not stack:
        return {}
    block = np.concatenate(stack)
    mean, std = block.mean(axis=0), block.std(axis=0)
    std[std < 1e-6] = 1.0
    np.savez(out_dir / "_scaler.npz", mean=mean, std=std, offset=TEXT_DIM)
    return {"columns": int(block.shape[1]), "rows": int(block.shape[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", help="parquet path (default: paths.events)")
    parser.add_argument("--out", help="state dir (default: paths.states)")
    parser.add_argument("--limit", type=int, help="only the first N events")
    parser.add_argument("--encoder", choices=["minilm", "hashed"], default="minilm",
                        help="hashed skips the model download (demo only)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config()
    frame = pd.read_parquet(args.events or cfg["paths"]["events"])
    if args.limit:
        frame = frame.head(args.limit)
    out_dir = Path(args.out or cfg["paths"]["states"])

    conn = db.get_conn(cfg["paths"]["db"])
    encoder = TextEncoder(cfg, prefer=args.encoder)
    log.info("building %d states with the %s encoder (D=%d)",
             len(frame), encoder.mode, STATE_DIM)

    written = write_states(conn, cfg, frame, out_dir, encoder)
    scaler = fit_scaler(frame, out_dir)
    (out_dir / "_manifest.json").write_text(json.dumps({
        "encoder": encoder.mode, "state_dim": STATE_DIM, "events": written,
        "text_dim": TEXT_DIM, "scaler": scaler,
        "label_source": sorted(set(frame.get("label_source", ["unknown"]))),
    }, indent=2))
    log.info("wrote %d state files to %s", written, out_dir)


if __name__ == "__main__":
    main()
