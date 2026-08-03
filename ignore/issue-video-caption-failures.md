Transient Ollama failures during video captioning produce permanently empty results marked done

**Labels:** `type:bug`, `priority:p2`  

---

## Problem
Every frame caption call is wrapped in `except Exception` that logs and continues
(`file_index/extractors/video.py:246-247`). If Ollama crashes or the model OOMs
mid-run, *all* frames of the current video fail, `process_video` returns scenes with
empty caption lists, and the worker stores an empty `video_scenes` body and advances
the item toward done (`file_index/queue.py:581`). The video is now permanently
uncaptioned: the queue item is `done`, the stage row exists, and nothing ever retries
it (unlike a raised exception, which gets the 3-retry treatment, or a failed deferred
summary, which returns to `pending_summary`). The same swallow exists for the inline
(non-deferred) summary path (`file_index/extractors/video.py:279-284`), while the
deferred sweep correctly lets failures raise.

## Proposed Solution
Distinguish "frame undecodable" from "model call failed": count caption exceptions in
`process_video`, and if every attempted VLM call failed (or a failure ratio exceeds a
threshold), raise instead of returning empty results so the normal retry path
engages. A cheaper variant: after `_process_video`, if no scene has captions but
scenes were detected and the VLM was attempted, treat it as a failure. Consider a
`client.ping()` check between files to fail fast when Ollama has died.

## Acceptance Criteria
- [ ] A video whose every caption call raises (mock client) ends the run as
      failed/retryable, not `done` with an empty `video_scenes` body.
- [ ] A video with genuinely unreadable frames (extraction returns None) still
      completes as today.
