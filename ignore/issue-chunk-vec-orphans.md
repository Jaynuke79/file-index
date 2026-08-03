Replacing content early in a run leaves orphaned sqlite-vec rows; rowid reuse can then fail chunk inserts

**Labels:** `type:bug`, `priority:p1`  

---

## Problem
`Index._delete_content_row()` only deletes matching `chunk_vec` rows when
`self._vec_dim` is set (`file_index/index.py:268`), but `_vec_dim` is only populated
by `_ensure_vec_table()`, which runs on the first embedding *insert* (or vector
search) of the process. The common re-index path — `scan` or `deep` starts, and the
first thing it does is `store_content()` for a modified file, which replaces the old
stage — therefore deletes the old `chunks` rows but leaves their embeddings behind in
`chunk_vec`.

Consequences:
- Stale vectors keep matching in KNN. `vector_search()` drops them when the `chunks`
  join fails (`file_index/index.py:484-489`), but each stale hit silently consumes
  one of the `k` result slots, degrading recall as churn accumulates.
- `chunks.id` is a plain `INTEGER PRIMARY KEY` (rowid alias, no AUTOINCREMENT), so
  after the highest chunk rows are deleted, new inserts can reuse those rowids. The
  subsequent `INSERT INTO chunk_vec(rowid, …)` (`file_index/index.py:346`) then hits
  an existing rowid — either failing the insert (marking the file failed) or
  associating a new chunk with a wrong stale vector, depending on vec0's conflict
  behavior.

## Proposed Solution
Make vec cleanup unconditional on `_vec_dim`: track whether the `chunk_vec` table
exists (e.g. query `sqlite_master` once at startup when `_has_vec`), and delete from
it whenever it exists — not only after the first insert of the process. Alternatives:
switch `chunks` to `AUTOINCREMENT` (prevents rowid reuse but not stale-hit pollution),
or add a startup reconciliation that deletes `chunk_vec` rows with no matching
`chunks` row.

## Acceptance Criteria
- [ ] Re-indexing a modified file in a fresh process leaves zero `chunk_vec` rows
      whose rowid has no matching `chunks` row.
- [ ] A regression test covers: store chunks with embeddings → close/reopen Index →
      `store_content()` same stage again → vector search returns only current chunks
      and re-insert succeeds.
