Audio pipeline still has the VRAM-contention patterns the video pipeline was reworked to avoid

**Labels:** `type:chore`, `priority:p1`  

---

## Problem
Recent work (commits fef4c92…ffccb62) carefully removed two costs from the video
path: Whisper grabbing VRAM while the vision model is pinned (background CPU
transcription) and per-file vision↔agent model swaps (deferred summary sweep). The
audio path has both problems untouched: `_process_audio` transcribes inline with the
configured `whisper_device` (default `cuda`, loading float16 large-v3 next to the
pinned ~20 GB vision model — the documented ~30x caption slowdown scenario), then
immediately calls the *agent* model for a summary (`file_index/queue.py:562-567`),
forcing a vision↔agent swap per audio file. With the default priority
`["image", "audio", "video"]`, every audio file interrupts the vision-resident phase
of the run.

## Proposed Solution
Apply the same treatment: route audio transcription through the background CPU
transcriber (`pending_transcript` state already exists and is kind-agnostic in the
queue schema), and defer audio summaries into the existing `_summary_sweep`
(`pending_summary`), which already runs with the agent model loaded once. This is
mostly wiring — `_pump_transcripts`/`_summary_sweep` currently assume video stages
(`video_transcript`/`video_scenes`), so they need a per-kind stage mapping.

## Acceptance Criteria
- [ ] During the captioning phase of a `deep` run containing audio files, the vision
      model is never evicted/offloaded by Whisper or the agent model.
- [ ] Audio transcripts/summaries survive interruption and resume, matching the video
      pipeline's checkpoint semantics.
