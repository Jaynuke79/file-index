"""Tier-2 dedup: identical files (same hash) reuse deep results instead of
re-running the models."""

from pathlib import Path
from unittest.mock import MagicMock

from file_index.index import PENDING_DEEP
from file_index.queue import Tier2Worker


def _add_video(index, path: Path, hash_: str) -> int:
    path.write_bytes(b"fake video bytes")
    fid = index.upsert_file(str(path), hash_, 16, 1000.0, "video/mp4", "video")
    index.enqueue(fid, 2, PENDING_DEEP, "video", 1000.0)
    return fid


def test_duplicate_video_reuses_results(tmp_env):
    cfg, index, root = tmp_env
    src_id = _add_video(index, root / "a.mp4", "samehash")
    dup_id = _add_video(index, root / "copy.mp4", "samehash")

    # src was fully processed: content + timestamped chunks with embeddings
    cid = index.store_content(
        src_id, "video_scenes", "video-1.0", "[00:00 - 00:10] a red car drives by",
        meta={"scenes": [{"start": 0, "end": 10}]},
    )
    index.store_chunks(src_id, cid, "video_scenes", [
        {"text": "a red car drives by", "ts_start": 0.0, "ts_end": 10.0,
         "embedding": [0.1, 0.2, 0.3]},
    ])
    q = index.db.execute(
        "SELECT id FROM queue WHERE file_id=? AND tier=2", (src_id,)
    ).fetchone()
    index.mark_done(q["id"])
    index.commit()

    client = MagicMock()  # any model call would blow up the assertion below
    worker = Tier2Worker(cfg, index, client=client)
    item = index.next_pending(tier=2)
    assert item["file_id"] == dup_id
    worker._process(item, Path(item["path"]))

    client.generate.assert_not_called()
    rows = index.get_content(dup_id, "video_scenes")
    assert rows and "red car" in rows[0]["body"]
    ch = index.db.execute(
        "SELECT * FROM chunks WHERE file_id=?", (dup_id,)
    ).fetchone()
    assert ch["ts_start"] == 0.0 and ch["embedding"] is not None
    # FTS entry points at the duplicate's own path
    hits = index.fts_search("red car")
    assert any(h.file_id == dup_id for h in hits)


def test_no_duplicate_falls_through(tmp_env):
    cfg, index, root = tmp_env
    _add_video(index, root / "only.mp4", "uniquehash")
    worker = Tier2Worker(cfg, index, client=MagicMock())
    item = index.next_pending(tier=2)
    assert worker._reuse_duplicate(item, Path(item["path"])) is False
