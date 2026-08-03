"""Purge hard-deletes soft-deleted files' data, including the FTS and vector
shadow tables and the thumbnail cache."""

import time

from typer.testing import CliRunner

from file_index import cli
from file_index.web import prune_thumbs


def _indexed(index, path, fid_hash, mtime=1000.0, body="secret tax return 2024"):
    path.write_text(body)
    fid = index.upsert_file(str(path), fid_hash, len(body), mtime, "text/plain", "text")
    cid = index.store_content(fid, "text", "text-1.0", body)
    index.store_chunks(fid, cid, "text", [
        {"text": body, "ts_start": None, "ts_end": None, "embedding": [0.1, 0.2, 0.3]},
    ])
    index.commit()
    return fid


def _counts(index):
    q = lambda sql: index.db.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "files": q("SELECT count(*) FROM files"),
        "content": q("SELECT count(*) FROM content"),
        "chunks": q("SELECT count(*) FROM chunks"),
        "fts": q("SELECT count(*) FROM content_fts"),
    }


def test_purge_erases_soft_deleted_content_everywhere(tmp_env):
    cfg, index, root = tmp_env
    keep = _indexed(index, root / "keep.txt", "h1", body="keep me around")
    drop = _indexed(index, root / "drop.txt", "h2")
    index.mark_deleted(drop)
    index.commit()

    before = _counts(index)
    assert before["files"] == 2 and before["fts"] == 2

    stats = index.purge_deleted()

    assert stats == {"files": 1, "content": 1, "chunks": 1}
    after = _counts(index)
    assert after == {"files": 1, "content": 1, "chunks": 1, "fts": 1}
    # the surviving file is the one we kept, and its data is intact
    assert index.get_file_by_path(str(root / "keep.txt"))["id"] == keep
    assert index.fts_search("keep")[0].file_id == keep
    # the purged body is gone from the keyword index
    assert index.fts_search("secret") == []
    if index._vec_table:
        left = index.db.execute("SELECT count(*) FROM chunk_vec").fetchone()[0]
        assert left == 1


def test_purge_respects_older_than_cutoff(tmp_env):
    cfg, index, root = tmp_env
    old = _indexed(index, root / "old.txt", "h1")
    recent = _indexed(index, root / "recent.txt", "h2")
    index.mark_deleted(old)
    index.db.execute("UPDATE files SET updated_at=? WHERE id=?",
                     (time.time() - 30 * 86400, old))
    index.mark_deleted(recent)
    index.commit()

    stats = index.purge_deleted(older_than=time.time() - 7 * 86400)

    assert stats["files"] == 1
    remaining = {r["id"] for r in index.db.execute("SELECT id FROM files")}
    assert remaining == {recent}


def test_purge_of_nothing_is_a_noop(tmp_env):
    cfg, index, root = tmp_env
    _indexed(index, root / "a.txt", "h1")
    assert index.purge_deleted() == {"files": 0, "content": 0, "chunks": 0}
    assert _counts(index)["files"] == 1


def test_prune_thumbs_removes_stale_and_superseded(tmp_env):
    cfg, index, root = tmp_env
    thumbs = cfg.data_dir / "thumbs"
    thumbs.mkdir()
    live = _indexed(index, root / "live.txt", "h1", mtime=1234.0)
    dead = _indexed(index, root / "dead.txt", "h2", mtime=5678.0)
    index.mark_deleted(dead)
    index.commit()

    (thumbs / f"{live}-1234.jpg").write_bytes(b"current")
    (thumbs / f"{live}-1000.jpg").write_bytes(b"superseded version")
    (thumbs / f"{dead}-5678.jpg").write_bytes(b"deleted file")

    removed = prune_thumbs(thumbs, index.live_thumb_keys())

    assert removed == 2
    assert {p.name for p in thumbs.glob("*.jpg")} == {f"{live}-1234.jpg"}


def test_purge_cli_requires_confirmation(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    fid = _indexed(index, root / "drop.txt", "h1")
    index.mark_deleted(fid)
    index.commit()
    monkeypatch.setattr(cli, "_load", lambda: (cfg, index))

    result = CliRunner().invoke(cli.app, ["purge"], input="n\n")
    assert "aborted" in result.output
    assert _counts(index)["files"] == 1  # still there

    result = CliRunner().invoke(cli.app, ["purge", "--yes"])
    assert result.exit_code == 0
    assert "purged" in result.output
    assert _counts(index)["files"] == 0


def test_purge_cli_reports_when_nothing_matches(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    _indexed(index, root / "a.txt", "h1")
    monkeypatch.setattr(cli, "_load", lambda: (cfg, index))
    result = CliRunner().invoke(cli.app, ["purge", "--yes"])
    assert "nothing to purge" in result.output
