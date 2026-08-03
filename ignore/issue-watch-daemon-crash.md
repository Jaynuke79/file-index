watch daemon dies on any exception outside OSError/PermissionError

**Labels:** `type:bug`, `priority:p2`  

---

## Problem
`Watcher._flush` catches only `(OSError, PermissionError)` per path
(`file_index/watch.py:113-114`). Any other exception — e.g.
`sqlite3.OperationalError: database is locked` when a `scan`/`deep` run holds the
write lock longer than the 5 s default busy timeout, or a `ValueError` from an
extractor via `Tier1Worker.run()` in the `on_event` callback
(`file_index/cli.py:398-401`) — propagates out of `run()`, stops the observer, and
exits the daemon. Under systemd (`Restart=on-failure`) this causes a silent restart
and loses the debounce map; run interactively, the watcher just dies. This
contradicts the README's "…never abort a run" robustness claim.

## Proposed Solution
Broaden the per-path handler to `except Exception` with a logged warning (the crawler
already treats per-file processing as best-effort), and/or wrap the whole `_flush`
body so one bad path can't take down the loop. Consider an explicit `busy_timeout`
pragma and a retry-once for `OperationalError` since concurrent `scan`/`deep` while
watching is a supported scenario.

## Acceptance Criteria
- [ ] An exception raised while processing one watched path is logged and the daemon
      continues watching (test with a handler that raises a non-OSError).
- [ ] A locked database during `_flush` does not terminate the watcher.
