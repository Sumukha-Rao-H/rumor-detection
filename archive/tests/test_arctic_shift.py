"""Arctic Shift collector tests: zst streaming, normalization, ingest filter."""

import json

import zstandard as zstd

from src import db
from src.collectors.arctic_shift import _Ingestor, record_to_post, stream_zst
from src.pipeline.tickers import TickerExtractor
from src.utils.config import load_config


def make_record(pid, title, created=1_750_000_000, **over):
    rec = {
        "id": pid, "subreddit": "wallstreetbets", "title": title,
        "selftext": "", "author": "u1", "created_utc": created, "score": 12,
        "upvote_ratio": 0.88, "num_comments": 3, "link_flair_text": "DD",
        "url": "https://reddit.com/x",
    }
    rec.update(over)
    return rec


def write_zst(path, records):
    payload = "\n".join(json.dumps(r) for r in records).encode()
    path.write_bytes(zstd.ZstdCompressor().compress(payload))


def test_stream_zst_roundtrip(tmp_path):
    records = [make_record("p1", "a"), make_record("p2", "b")]
    path = tmp_path / "dump.zst"
    write_zst(path, records)
    out = list(stream_zst(path))
    assert [r["id"] for r in out] == ["p1", "p2"]


def test_record_to_post_normalization():
    post = record_to_post(make_record("p1", "hello", created="1750000000.0"))
    assert post["created_utc"] == 1_750_000_000
    assert post["source"] == "arctic"
    assert post["flair"] == "DD"
    assert record_to_post({"title": "no id"}) is None
    assert record_to_post({"id": "x", "created_utc": None}) is None


def test_ingestor_filters_by_ticker_and_window(tmp_path):
    cfg = load_config()
    conn = db.get_conn(tmp_path / "t.db")
    ingestor = _Ingestor(conn, TickerExtractor.from_config(cfg),
                         start_ts=1_700_000_000, end_ts=1_800_000_000)
    ingestor.offer(make_record("keep1", "$TSLA merger rumor"))
    ingestor.offer(make_record("drop_no_ticker", "I like the stock"))
    ingestor.offer(make_record("drop_old", "$TSLA old news", created=1_600_000_000))
    ingestor.flush()

    ids = {r[0] for r in conn.execute("SELECT id FROM posts")}
    assert ids == {"keep1"}
    links = conn.execute("SELECT post_id, ticker FROM post_tickers").fetchall()
    assert [(r[0], r[1]) for r in links] == [("keep1", "TSLA")]
    assert ingestor.seen == 3 and ingestor.kept == 1
