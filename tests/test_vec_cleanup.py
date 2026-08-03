"""Replacing content must clean chunk_vec even in a fresh process.

Regression for the orphaned-vector bug: _delete_content_row skipped the vec
table until the first embedding insert of the process set _vec_dim, so a
`scan`/`deep` run that started by replacing a modified file's stage left the
old embeddings behind — polluting KNN results and, once chunk rowids were
reused, colliding with new inserts.
"""

import pytest

pytest.importorskip("sqlite_vec")

from file_index.index import Index


def _store(index, file_id, text, vec):
    cid = index.store_content(file_id, "text", "text-1.0", text)
    index.store_chunks(
        file_id, cid, "text",
        [{"text": text, "ts_start": None, "ts_end": None, "embedding": vec}],
    )
    index.commit()


def test_replacement_in_fresh_process_leaves_no_orphans(tmp_env):
    cfg, index, root = tmp_env
    if not index._has_vec:
        pytest.skip("sqlite-vec extension failed to load")
    f = root / "a.txt"
    f.write_text("v1")
    fid = index.upsert_file(str(f), "h1", 2, 1.0, "text/plain", "text")
    _store(index, fid, "version one", [1.0, 0.0, 0.0])

    # fresh process: new Index, _vec_dim starts as None
    index2 = Index(cfg.db_path)
    _store(index2, fid, "version two", [0.0, 1.0, 0.0])

    orphans = index2.db.execute(
        "SELECT count(*) FROM chunk_vec WHERE rowid NOT IN (SELECT id FROM chunks)"
    ).fetchone()[0]
    assert orphans == 0

    # KNN sees only the current chunk
    hits = index2.vector_search([0.0, 1.0, 0.0], limit=10)
    assert [h.snippet for h in hits] == ["version two"]
    index2.close()


def test_repeated_replacement_survives_rowid_reuse(tmp_env):
    """Deleting the highest chunk rows frees their rowids for reuse; the next
    insert into chunk_vec must not collide with a stale row."""
    cfg, index, root = tmp_env
    if not index._has_vec:
        pytest.skip("sqlite-vec extension failed to load")
    f = root / "b.txt"
    f.write_text("x")
    fid = index.upsert_file(str(f), "h2", 1, 1.0, "text/plain", "text")
    _store(index, fid, "first", [1.0, 0.0, 0.0])

    for i in range(3):  # each replacement reuses the freed max rowid
        fresh = Index(cfg.db_path)
        _store(fresh, fid, f"rewrite {i}", [0.0, 0.0, 1.0])
        fresh.close()

    check = Index(cfg.db_path)
    assert check.db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
    assert check.db.execute("SELECT count(*) FROM chunk_vec").fetchone()[0] == 1
    check.close()
