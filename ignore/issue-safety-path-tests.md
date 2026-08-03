Safety-critical paths are untested: agent whitelist, organize apply guards, watcher

**Labels:** `type:chore`, `priority:p2`  

---

## Problem
The test suite covers the pipeline well (crawler, queue resume, dedup, deferred
summaries, VLM JSON, web store) but has zero coverage of the code enforcing the
project's safety model, which is its main selling point:
- `AgentTools._check_path` (`file_index/agent.py:120` — whitelist enforcement for
  every agent tool) — untested; a regression could silently let the LLM read
  arbitrary paths.
- `propose_organization`'s action filter (`file_index/agent.py:294-313`) that
  discards model-proposed paths escaping the target directory — untested against
  traversal (`../`, symlink, absolute-elsewhere) inputs.
- `cli.organize --apply` flow: `_assert_inside` (`file_index/cli.py:335`), the
  target-exists skip, and index path updates after moves — untested.
- `file_index/watch.py` — entirely untested (debounce, delete handling, exclusion
  filtering).

## Proposed Solution
Add focused unit tests: whitelist rejection/acceptance for each agent tool (no model
needed — call `AgentTools` directly); `propose_organization` with a mocked client
returning malicious action lists; an `organize --apply` test via `typer.testing`
with a scripted plan; watcher `_flush` tests driving `pending` directly (no real
inotify needed).

## Acceptance Criteria
- [ ] Tests fail if `_check_path` or the organize action filter stops rejecting
      out-of-scope paths (including `..` traversal and symlinks out of a root).
- [ ] Watcher create/modify/delete handling has direct test coverage.
