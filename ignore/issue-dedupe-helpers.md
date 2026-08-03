Duplicated helpers are drifting: two _fts_escape implementations differ, strip_think and timestamp formatting each exist twice

**Labels:** `type:chore`, `priority:p2`  

---

## Problem
Three helpers are copy-pasted and already diverging:
- `_fts_escape`: `file_index/index.py:553` splits on `str.split()`,
  `file_index/web.py:38` on `re.split(r"\s+")` — currently equivalent-ish, but a fix
  in one (e.g. handling `-`/`*` prefixes) won't reach the other.
- `strip_think` (`file_index/agent.py:185`) and `_strip_think`
  (`file_index/queue.py:640`) are identical qwen3 output scrubbers.
- `_ts` (`file_index/extractors/audio.py:69`) and `format_ts`
  (`file_index/search.py:23`) are the same formatter, and `file_index/queue.py:615`
  reaches into another module's private helper (`audio_ex._ts`).

## Proposed Solution
Create a small shared module (e.g. `file_index/util.py`) holding `fts_escape`,
`strip_think`, and `format_ts`; update the six call sites; make `format_transcript`
use the shared formatter. Pure mechanical refactor.

## Acceptance Criteria
- [ ] One definition of each helper, imported everywhere it's used; no
      private-cross-module imports remain.
- [ ] Existing tests (`test_fts_escape_neutralizes_syntax`, chunking/transcript
      tests) pass unchanged.
