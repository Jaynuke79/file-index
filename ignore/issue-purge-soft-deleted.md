No purge/lifecycle for soft-deleted data: excluded files' content persists in the DB forever; thumbnail cache never pruned

**Labels:** `type:chore`, `priority:p2`  

---

## Problem
Deletion is soft everywhere (`deleted=1`, `file_index/index.py:208`), and nothing
ever reclaims the underlying data: `content` bodies, FTS rows, `chunks`, and
`chunk_vec` embeddings of deleted or excluded files remain in `index.db`
indefinitely. Two consequences:

- **Privacy expectation mismatch:** `file-index exclude <sensitive-dir>` reports
  "N removed from index" (`file_index/crawler.py:91`), but every extracted text,
  caption, and transcript of those files remains readable in the DB (and keeps
  consuming FTS/vec space). Reversibility is documented, but "excluded yet fully
  retained forever" deserves an explicit opt-out.
- **Unbounded growth:** high-churn roots accumulate dead rows; the thumbnail cache
  (`thumbs/<id>-<mtime>.jpg`, `file_index/web.py:349`) similarly accumulates a new
  file per content change and is never pruned.

## Proposed Solution
Add a `purge` command (or `exclude --purge` / `scan --gc`) that hard-deletes rows for
files with `deleted=1` older than a threshold — including FTS and `chunk_vec`
mirrors — and prunes thumbs whose `<id>-<mtime>` no longer matches a live file.
Foreign keys with `ON DELETE CASCADE` already cover `content`/`chunks` if `files`
rows are deleted; the FTS and vec mirrors need explicit cleanup (see the related
chunk_vec orphan issue).

## Acceptance Criteria
- [ ] A documented command removes all stored content of soft-deleted files from
      `index.db` (verified by querying content/chunks/FTS/vec afterwards).
- [ ] Thumbnail files for deleted/re-modified files are removed by the same or a
      related sweep.
