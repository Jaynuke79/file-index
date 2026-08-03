browse endpoint edge cases: deleted files inflate the captioned count, invalid Range yields a negative Content-Length, dead DB-exists check

**Labels:** `type:bug`, `priority:p2`  

---

## Problem
Three small correctness defects in the browse path:

1. `Store.summary()` counts `captioned` from `content` with no join to
   `files.deleted=0` (`file_index/web.py:69-72`), while the per-kind counts do
   filter. Once a captioned file is deleted/excluded, the "captioned" figure exceeds
   reality (it can even exceed the kind totals). The existing test doesn't catch this
   because its deleted fixture file has no content rows.
2. `_serve_media` accepts `Range: bytes=500-100` (start > end): `length` becomes
   negative and is emitted as a negative `Content-Length`
   (`file_index/web.py:371-393`), producing a broken response instead of 416/200.
3. `cli.browse` checks `cfg.db_path.exists()` *after* `_load()`
   (`file_index/cli.py:417`), but `Index.__init__` has already created the DB file —
   so the "run scan first" error is unreachable and a fresh machine gets an empty
   gallery instead of guidance.

## Proposed Solution
(1) Add `JOIN files f ON f.id=content.file_id AND f.deleted=0` to the captioned
count. (2) After parsing, if `start > end`, either ignore the header (serve 200 full)
or return 416 per RFC 7233. (3) Check `db_path.exists()` before constructing `Index`
(e.g. before `_load()`, using `load_config()` directly).

## Acceptance Criteria
- [ ] A deleted file with caption content no longer counts toward
      `summary().captioned`.
- [ ] `Range: bytes=500-100` gets a well-formed 416 or 200 response with a
      non-negative Content-Length.
- [ ] `browse` without an existing index prints the "run scan first" error and does
      not create an empty DB.
