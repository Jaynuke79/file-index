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
scene detection — `deep.prefetch_files`, default 2), and within a video, ffmpeg
frame/audio extraction runs on `deep.video_frame_workers` threads (default 4)
while the VLM captions. Whisper still runs after the VLM per file so both never
compete for VRAM. Note that a stop may additionally wait for an in-flight
prefetch (at most one scene detection) to finish.

Video summaries are deferred to an end-of-run sweep
(`deep.defer_video_summaries`, default true): every video's scene captions and
transcript are stored (and searchable) immediately, then all summaries run with
the agent model loaded once — avoiding a ~15 s vision↔agent VRAM swap per
video. Interrupting mid-run is still safe: pending summaries persist in the
queue and the next `deep` run picks them up; a failed summary retries without
redoing captions.

When a video sits in a folder with already-summarized siblings, the captioner
and summarizer get those summaries as background (`deep.neighbor_context`,
default 3 siblings, 0 to disable) — so the 40th replay in your Smite folder is
described knowing the other 39 were Smite matches, and recurring
people/activities carry across clips.

## Configuration

`~/.config/file-index/config.yaml` — created by `init`. Everything is swappable
without code changes: model names, Ollama URL, roots, exclude globs, chunk
sizes, tier-2 priority order (default: images before video, newest first),
Whisper device/precision.

Data lives in `~/.local/share/file-index/` (`index.db`, logs, audit log,
`thumbs/` cache for the browse UI).

`browse` serves a gallery at `http://127.0.0.1:8765/` (localhost only unless
`--host` says otherwise): thumbnails for images/videos/PDFs, the VLM caption or
video/audio summary on each card, full-text search over everything indexed,
kind filters, and a detail view with the original media, OCR text, EXIF,
transcripts, and scene lists. It opens the DB read-only, so it is safe to keep
running while `scan`/`deep` work.

## Safety model

- Only whitelisted roots are ever read; every agent/organize path is checked.
- v1 performs **no destructive operations**. `organize` is read-only by default;
  `--apply` performs moves/renames only, after an interactive confirmation.
  Deletion is not implemented anywhere (proposals may only *flag* candidates).
- Every applied write is recorded in the audit log (DB + `audit.log`) with
  before/after paths.
- Permission errors, broken symlinks, and files disappearing mid-processing are
  marked failed (max 3 retries) and never abort a run.

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
- Extractions are versioned per stage (`content.extractor_version`) so any
  stage can be re-run later with a better model without touching the others.

## Tests

```bash
.venv/bin/pytest
```

Covers crawler incremental logic (unchanged/modified/moved/deleted), chunking,
queue kill/resume, tier-2 priority ordering, and VLM JSON validation (mocked model).
