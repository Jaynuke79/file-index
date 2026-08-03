# file-index

Fully local filesystem content indexer: crawls whitelisted directories, extracts
semantic content from text, code, PDFs, office docs, images, audio, and video,
builds a hybrid keyword (FTS5) + vector (sqlite-vec) index, and exposes an agent
CLI for natural-language search and organization proposals.

**No data leaves the machine.** All models run locally via Ollama and
faster-whisper (CUDA).

## Requirements

- Linux, Python 3.11+, NVIDIA GPU recommended (Whisper + big models)
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- `ffmpeg` (`sudo apt install ffmpeg`) for audio/video
- Models (pulled on demand by `init`): `qwen3:30b-a3b` (agent),
  `qwen2.5vl:32b` (vision), `nomic-embed-text` (embeddings).
  Whisper `large-v3` weights are downloaded by faster-whisper on first use.

## Install

```bash
cd file-index
python3 -m venv .venv
.venv/bin/pip install -e ".[deep,dev]"
```

First run, end to end:

```bash
file-index init                 # pick roots, pull models
file-index scan                 # minutes: text + metadata, search works after this
file-index search "tax 2024"
file-index deep                 # hours: images, audio, video (ctrl-c safe, resumable)
file-index browse               # look at what it found
```

To keep the index current, run `file-index watch` or install the user unit in
[`systemd/file-index-watch.service`](systemd/file-index-watch.service); `deep`
stays manual so it never competes for the GPU unattended.

## Usage

```bash
file-index init              # interactive: choose roots/excludes, verify Ollama+models
file-index scan              # tier 1: crawl + text extraction + keyword/vector index
file-index search "tax 2024" # hybrid search (works right after scan)
file-index deep              # tier 2: VLM images, scanned PDFs, Whisper, video pipeline
file-index ask "where are my insurance documents?"
file-index organize ~/Downloads            # prints a plan, changes nothing
file-index organize ~/Downloads --apply    # applies moves/renames after confirmation
file-index browse            # local web gallery: files + captions, search, filters
file-index exclude PATH      # remove files from the index + future scans (disk untouched)
file-index reindex           # re-extract stages whose extractor or model changed
file-index purge             # erase indexed data of removed/excluded files for good
file-index status            # queue stats, per-type counts, failures
file-index watch             # incremental watcher (or install the systemd unit)
```

`deep` is safe to interrupt (ctrl-c or `kill`): it finishes the current file,
checkpoints, and unloads all Ollama models so the GPU is immediately free for
other work (send the signal twice to skip the current file). It resumes
exactly where it left off. Re-running `scan` with no changes is
near-instant (size+mtime short-circuit, content-hash verification on change).

`deep` overlaps CPU and GPU work: while the GPU runs the current file's models,
background threads prepare the next files (image transcode/downscale, video
scene detection — `deep.prefetch_files`, default 4), and within a video, ffmpeg
frame/audio extraction runs on `deep.video_frame_workers` threads (default 4)
while the VLM captions. Note that a stop may additionally wait for an in-flight
prefetch (at most one scene detection) to finish.

Whisper transcription runs on a background thread: after a video's captions
are stored it moves to a `pending_transcript` queue state, the GPU immediately
starts the next video's captions, and the transcript is stored as it finishes.
Audio files use the same path — they do no work on the critical path at all,
so a folder of podcasts never interrupts the vision-resident phase of a run.
Background transcription always uses CPU (int8) regardless of
`deep.whisper_device` — it is off the critical path, and Whisper grabbing VRAM
first pushes the vision model into partial CPU offload, slowing every caption
~30x. Ollama models are pinned in VRAM for the whole run (`keep_alive`) and
unloaded at the end.

Video and audio summaries are deferred to an end-of-run sweep
(`deep.defer_video_summaries`, default true): every video's scene captions and
transcript are stored (and searchable) immediately, then all summaries run with
the agent model loaded once — avoiding a ~15 s vision↔agent VRAM swap per
file. Interrupting mid-run is still safe: pending transcripts and summaries
persist in the queue and the next `deep` run picks them up; a failed summary
retries without redoing captions.

VLM work per video is bounded: scene-heavy videos are sampled down to
`deep.video_max_scenes` scenes (default 40, evenly spread; 0 = no cap), and
frames that are near-duplicates of already-captioned ones (perceptual hash —
common in gameplay/screen recordings) are skipped
(`deep.video_dedup_frames`). Every frame is downscaled to the vision model's
input limit during extraction; beyond it the encoder OOMs on a 32 GB GPU.

### Tuning captioning throughput

Captioning dominates `deep` — roughly `scenes x frames_per_scene x ~9 s` per
video — so these knobs trade visual detail for wall-clock. Measured over 3532
already-captioned videos (32747 captions, 9.3/video):

| Setting | Effect |
|---|---|
| `deep.video_frames_per_scene: 1` | 6.2 captions/video — **33% less VLM work**. Not 50%, because dedup already drops many second frames and scenes under 2 s take one frame anyway. |
| `deep.video_dedup_distance` (default 6) | Perceptual-hash distance under which a frame counts as a near-duplicate. 8–10 skips noticeably more on screen recordings and gameplay; 0 skips only identical frames. |
| `deep.video_max_scenes` (default 40) | Caps the tail. One video in the sample detected 754 scenes. |

Lowering detail is safe to revisit later: bump the stage version or change the
model and `file-index reindex` re-captions only the affected files.

When a video sits in a folder with already-summarized siblings, the captioner
and summarizer get those summaries as background (`deep.neighbor_context`,
default 3 siblings, 0 to disable) — so the 40th replay in your Smite folder is
described knowing the other 39 were Smite matches, and recurring
people/activities carry across clips.

## Configuration

`~/.config/file-index/config.yaml` — created by `init`. Everything is swappable
without code changes: model names, Ollama URL, roots, exclude globs, chunk
sizes, tier-2 priority order (default: images, then audio, then video; newest
first), Whisper device/precision. After changing a model name, run
[`file-index reindex`](#architecture) so existing files are re-extracted with
it.

Data lives in `~/.local/share/file-index/` (`index.db`, logs, audit log,
`thumbs/` cache for the browse UI).

`browse` serves a gallery at `http://127.0.0.1:8765/` (localhost only unless
`--host` says otherwise): thumbnails for images/videos/PDFs, the VLM caption or
video/audio summary on each card, full-text search over everything indexed,
kind filters, and a detail view with the original media, OCR text, EXIF,
transcripts, and scene lists. It opens the DB read-only, so it is safe to keep
running while `scan`/`deep` work.

On a loopback bind the server accepts only loopback `Host` headers and answers
403 otherwise, so a website you visit cannot DNS-rebind to `127.0.0.1:8765` and
read your index. `--host` beyond loopback disables that check (the machine's
external names are unknowable) and serves every indexed file unauthenticated to
anyone who can reach the address — `browse` prints a warning when you do it.

## Safety model

- Only whitelisted roots are ever read; every agent/organize path is checked.
- v1 performs **no destructive operations**. `organize` is read-only by default;
  `--apply` performs moves/renames only, after an interactive confirmation.
  Deletion is not implemented anywhere (proposals may only *flag* candidates).
- Removal from the index is soft by default (`exclude` and vanished files keep
  their extractions, so re-adding is free). `purge` makes it permanent —
  erasing bodies, captions, transcripts, embeddings and cached thumbnails of
  removed files — for when the point of excluding was privacy, not tidiness.
  Use `--older-than-days N` to keep a grace period. Files on disk are never
  touched by either.
- Every applied write is recorded in the audit log (DB + `audit.log`) with
  before/after paths.
- Permission errors, broken symlinks, and files disappearing mid-processing are
  marked failed (max 3 retries) and never abort a run.
- Partial results are labelled rather than passed off as complete. If the vision
  model is unreachable the file is retried; if its video stream cannot be decoded
  at all, extraction stops early and the stored scenes are marked `degraded`
  (visible in `browse`) while the audio transcript and summary still land.

## Architecture

```
crawler → work queue (SQLite) → extractors (type-routed) → index (SQLite: FTS5 + sqlite-vec)
                                                              ↑
watchdog daemon ──────────────────────────────────────────────┘
agent CLI ← search/organize tools ← index
```

- **Tier 1** (fast): text/code/PDF/office extraction, EXIF, stats → keyword
  search available within the first pass.
- **Tier 2** (deep, resumable queue): VLM image analysis (strict-JSON with
  validation + one retry), scanned-PDF page OCR via VLM, Whisper transcripts
  with timestamps, and the video pipeline (PySceneDetect scenes → frame
  captions → transcript → agent-model summary). Video/audio chunks carry
  timestamp ranges so search hits resolve to a moment.
- Extractions are versioned per stage (`content.extractor_version`), and the
  version records the model for stages that depend on one
  (`image-1.2+qwen2.5vl:32b`). `file-index reindex` compares those against
  what the current code and config would produce and re-queues only the files
  that differ — so pointing `models.vision` at a better VLM re-captions your
  images without touching transcripts, text, or anything else. Use
  `--dry-run` to preview, `--stage` to narrow, `--force` to redo regardless.
  Rows written before versions carried model names are judged on the
  extractor version alone, so upgrading doesn't re-run everything once.

## Tests

```bash
.venv/bin/pytest
```

139 tests, no GPU or network required — models and Whisper are mocked, and the
handful of tests that need real ffmpeg generate their own clips and skip when it
is absent.

- **Indexing**: crawler incremental logic (unchanged/modified/moved/deleted,
  excludes, symlinks), chunking, queue kill/resume, tier-2 priority ordering,
  duplicate reuse, and sqlite-vec cleanup on re-extraction.
- **Deep pass**: VLM JSON validation and degradation, caption-failure retry,
  frame downscaling to the VLM input limit, undecodable-stream handling, scene
  sampling and tunable frame dedup, CPU/GPU overlap, background transcription,
  deferred summaries for video and audio, Whisper cache thread-safety, model
  unloading.
- **Safety**: the root whitelist across every agent tool (including `..`
  traversal and symlinks leaving a root), `organize` plan filtering and
  `--apply` guards, audit-log integrity, and `purge`.
- **Interfaces**: the browse Store/HTTP endpoints (search, ranges, thumbnails,
  Host validation), `reindex` version comparison and routing, and the watcher.
