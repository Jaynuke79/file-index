# Codebase audit — file-index (whole repo) (2026-08-03)

Audited all 21 source files (~5,600 lines: indexer core, queue workers, extractors,
agent, web UI, watcher) across all four classes. 12 findings: the headline risks are
stale sqlite-vec rows corrupting/failing vector indexing after content replacement,
a thread-unsafe Whisper model cache shared between the CPU background transcriber and
the CUDA inline path, and the browse server being readable cross-origin via DNS
rebinding. No data-loss defects found; the safety model (whitelist, no-delete) held
up under review, though it is untested.

Findings: 12 · bugs 6 · debt 3 · tests 1 · arch 2

---

## [1] Replacing content early in a run leaves orphaned sqlite-vec rows; rowid reuse can then fail chunk inserts

**Class:** bugs · **Suggested labels:** `type:bug`, `priority:p1` · **Files:** `file_index/index.py:266`, `file_index/index.py:158`

### Problem
`Index._delete_content_row()` only deletes matching `chunk_vec` rows when
`self._vec_dim` is set (`index.py:268`), but `_vec_dim` is only populated by
`_ensure_vec_table()`, which runs on the first embedding *insert* (or vector search)
of the process. The common re-index path — `scan` or `deep` starts, and the first
thing it does is `store_content()` for a modified file, which replaces the old stage —
therefore deletes the old `chunks` rows but leaves their embeddings behind in
`chunk_vec`.

Consequences:
- Stale vectors keep matching in KNN. `vector_search()` drops them when the `chunks`
  join fails (`index.py:484-489`), but each stale hit silently consumes one of the
  `k` result slots, degrading recall as churn accumulates.
- `chunks.id` is a plain `INTEGER PRIMARY KEY` (rowid alias, no AUTOINCREMENT), so
  after the highest chunk rows are deleted, new inserts can reuse those rowids. The
  subsequent `INSERT INTO chunk_vec(rowid, …)` (`index.py:346`) then hits an existing
  rowid — either failing the insert (marking the file failed) or associating a new
  chunk with a wrong stale vector, depending on vec0's conflict behavior.

### Proposed Solution
Make vec cleanup unconditional on `_vec_dim`: track whether the `chunk_vec` table
exists (e.g. query `sqlite_master` once at startup when `_has_vec`), and delete from
it whenever it exists — not only after the first insert of the process. Alternatives:
switch `chunks` to `AUTOINCREMENT` (prevents rowid reuse but not stale-hit pollution),
or add a startup reconciliation that deletes `chunk_vec` rows with no matching
`chunks` row.

### Acceptance Criteria
- [ ] Re-indexing a modified file in a fresh process leaves zero `chunk_vec` rows
      whose rowid has no matching `chunks` row.
- [ ] A regression test covers: store chunks with embeddings → close/reopen Index →
      `store_content()` same stage again → vector search returns only current chunks
      and re-insert succeeds.

---

## [2] Whisper model cache is not thread-safe and thrashes between concurrent CPU/GPU users

**Class:** bugs · **Suggested labels:** `type:bug`, `priority:p1` · **Files:** `file_index/extractors/audio.py:21`, `file_index/queue.py:285`, `file_index/queue.py:543`

### Problem
`audio._get_model()` caches one model in module globals (`_model`, `_model_key`) with
a check-then-set and no lock. During a `deep` run two threads use it concurrently
with *different* keys: the background transcriber thread always requests
`("large-v3", "cpu", "int8")` (`queue.py:285-289`), while the main thread's inline
audio processing requests the configured `("large-v3", "cuda", "float16")`
(`queue.py:543-551`).

- **Race:** interleaved calls can leave `_model` from one thread and `_model_key`
  from the other. From then on, callers silently get the wrong device — e.g. the
  "CPU-only" background transcriber running a CUDA model, which is exactly the
  VRAM-grab the design (README, commit ffccb62) exists to prevent: the pinned vision
  model gets pushed into partial CPU offload, slowing every caption ~30x.
- **Thrash:** even without the race, a queue that interleaves audio files and video
  transcripts reloads large-v3 on every key flip (tens of seconds each), and both
  models can be resident at once (CPU RAM + VRAM).

### Proposed Solution
Guard `_get_model` with a `threading.Lock`, and cache per-key (dict keyed by
`(name, device, compute_type)`) instead of a single slot so CPU and CUDA instances
coexist without reloading. Optionally cap to one instance per device. Note
`WhisperModel.transcribe` itself is safe to call from one thread per instance —
per-key caching also ensures the two threads never share an instance.

### Acceptance Criteria
- [ ] Concurrent `transcribe()` calls with different device keys never return a model
      whose device differs from the requested key (test with threads + a fake
      WhisperModel recording constructor args).
- [ ] An interleaved audio/video queue loads each (device, compute_type) model at
      most once per run.

---

## [3] browse server is readable via DNS rebinding and fully exposed when bound beyond localhost

**Class:** bugs · **Suggested labels:** `type:bug`, `type:security`, `priority:p1` · **Files:** `file_index/web.py:268`, `file_index/cli.py:408`

### Problem
The browse server authenticates nothing and never validates the `Host` header. Two
consequences:

1. **DNS rebinding against the default localhost bind.** A malicious website the user
   visits can rebind its hostname to `127.0.0.1` and issue same-origin requests to
   `http://127.0.0.1:8765/`. Because the handler ignores `Host`, the site can read
   `api/files`, `api/file/<id>`, and `media/<id>` — i.e. exfiltrate the extracted
   text, captions, transcripts, and raw bytes of everything indexed (the user's
   whitelisted home directories). This defeats the project's core "no data leaves
   the machine" guarantee without any misconfiguration by the user.
2. **`--host` exposes everything unauthenticated.** `browse --host 0.0.0.0` serves
   every indexed file's original bytes to anyone on the network, with no warning at
   startup.

### Proposed Solution
Validate the `Host` header against an allowlist (`127.0.0.1[:port]`, `localhost[:port]`,
and the explicitly bound host) and reject others with 403 — this is the standard
localhost-server rebinding defense and is a few lines in `Handler._route`. For
non-localhost binds, either print a prominent warning or require a `--token` that
must be present as a query parameter/cookie. Setting
`Access-Control-Allow-Origin` is not needed (its absence already blocks normal CORS
reads); rebinding is the gap.

### Acceptance Criteria
- [ ] Requests with a `Host` header not matching the bound/allowed hosts receive 403
      and no body (covered by a test).
- [ ] `browse --host <non-loopback>` prints an explicit exposure warning (or refuses
      without an auth token, if the token approach is chosen).

---

## [4] Transient Ollama failures during video captioning produce permanently empty results marked done

**Class:** bugs · **Suggested labels:** `type:bug`, `priority:p2` · **Files:** `file_index/extractors/video.py:243`, `file_index/queue.py:581`

### Problem
Every frame caption call is wrapped in `except Exception` that logs and continues
(`video.py:246-247`). If Ollama crashes or the model OOMs mid-run, *all* frames of
the current video fail, `process_video` returns scenes with empty caption lists, and
the worker stores an empty `video_scenes` body and advances the item toward done.
The video is now permanently uncaptioned: the queue item is `done`, the stage row
exists, and nothing ever retries it (unlike a raised exception, which gets the
3-retry treatment, or a failed deferred summary, which returns to `pending_summary`).
The same swallow exists for the inline (non-deferred) summary path
(`video.py:279-284`), while the deferred sweep correctly lets failures raise.

### Proposed Solution
Distinguish "frame undecodable" from "model call failed": count caption exceptions in
`process_video`, and if every attempted VLM call failed (or a failure ratio exceeds a
threshold), raise instead of returning empty results so the normal retry path
engages. A cheaper variant: after `_process_video`, if no scene has captions but
scenes were detected and the VLM was attempted, treat it as a failure. Consider a
`client.ping()` check between files to fail fast when Ollama has died.

### Acceptance Criteria
- [ ] A video whose every caption call raises (mock client) ends the run as
      failed/retryable, not `done` with an empty `video_scenes` body.
- [ ] A video with genuinely unreadable frames (extraction returns None) still
      completes as today.

---

## [5] browse endpoint edge cases: deleted files inflate the captioned count, invalid Range yields a negative Content-Length, dead DB-exists check

**Class:** bugs · **Suggested labels:** `type:bug`, `priority:p2` · **Files:** `file_index/web.py:69`, `file_index/web.py:371`, `file_index/cli.py:417`

### Problem
Three small correctness defects in the browse path:

1. `Store.summary()` counts `captioned` from `content` with no join to
   `files.deleted=0` (`web.py:69-72`), while the per-kind counts do filter. Once a
   captioned file is deleted/excluded, the "captioned" figure exceeds reality (it can
   even exceed the kind totals). The existing test doesn't catch this because its
   deleted fixture file has no content rows.
2. `_serve_media` accepts `Range: bytes=500-100` (start > end): `length` becomes
   negative and is emitted as a negative `Content-Length` (`web.py:371-393`),
   producing a broken response instead of 416/200.
3. `cli.browse` checks `cfg.db_path.exists()` *after* `_load()`, but
   `Index.__init__` has already created the DB file — so the "run scan first" error
   is unreachable and a fresh machine gets an empty gallery instead of guidance.

### Proposed Solution
(1) Add `JOIN files f ON f.id=content.file_id AND f.deleted=0` to the captioned
count. (2) After parsing, if `start > end`, either ignore the header (serve 200 full)
or return 416 per RFC 7233. (3) Check `db_path.exists()` before constructing `Index`
(e.g. before `_load()`, using `load_config()` directly).

### Acceptance Criteria
- [ ] A deleted file with caption content no longer counts toward `summary().captioned`.
- [ ] `Range: bytes=500-100` gets a well-formed 416 or 200 response with a
      non-negative Content-Length.
- [ ] `browse` without an existing index prints the "run scan first" error and does
      not create an empty DB.

---

## [6] watch daemon dies on any exception outside OSError/PermissionError

**Class:** bugs · **Suggested labels:** `type:bug`, `priority:p2` · **Files:** `file_index/watch.py:91`, `file_index/cli.py:398`

### Problem
`Watcher._flush` catches only `(OSError, PermissionError)` per path
(`watch.py:113-114`). Any other exception — e.g. `sqlite3.OperationalError:
database is locked` when a `scan`/`deep` run holds the write lock longer than the 5 s
default busy timeout, or a `ValueError` from an extractor via
`Tier1Worker.run()` in the `on_event` callback — propagates out of `run()`, stops the
observer, and exits the daemon. Under systemd (`Restart=on-failure`) this causes a
silent restart and loses the debounce map; run interactively, the watcher just dies.
This contradicts the README's "…never abort a run" robustness claim.

### Proposed Solution
Broaden the per-path handler to `except Exception` with a logged warning (the crawler
already treats per-file processing as best-effort), and/or wrap the whole `_flush`
body so one bad path can't take down the loop. Consider an explicit
`busy_timeout` pragma and a retry-once for `OperationalError` since concurrent
`scan`/`deep` while watching is a supported scenario.

### Acceptance Criteria
- [ ] An exception raised while processing one watched path is logged and the daemon
      continues watching (test with a handler that raises a non-OSError).
- [ ] A locked database during `_flush` does not terminate the watcher.

---

## [7] Audio pipeline still has the VRAM-contention patterns the video pipeline was reworked to avoid

**Class:** arch · **Suggested labels:** `type:chore`, `priority:p1` · **Files:** `file_index/queue.py:543`

### Problem
Recent work (commits fef4c92…ffccb62) carefully removed two costs from the video
path: Whisper grabbing VRAM while the vision model is pinned (background CPU
transcription) and per-file vision↔agent model swaps (deferred summary sweep). The
audio path has both problems untouched: `_process_audio` transcribes inline with the
configured `whisper_device` (default `cuda`, loading float16 large-v3 next to the
pinned ~20 GB vision model — the documented ~30x caption slowdown scenario), then
immediately calls the *agent* model for a summary (`queue.py:562-567`), forcing a
vision↔agent swap per audio file. With the default priority `["image", "audio",
"video"]`, every audio file interrupts the vision-resident phase of the run.

### Proposed Solution
Apply the same treatment: route audio transcription through the background CPU
transcriber (`pending_transcript` state already exists and is kind-agnostic in the
queue schema), and defer audio summaries into the existing `_summary_sweep`
(`pending_summary`), which already runs with the agent model loaded once. This is
mostly wiring — `_pump_transcripts`/`_summary_sweep` currently assume video stages
(`video_transcript`/`video_scenes`), so they need a per-kind stage mapping.

### Acceptance Criteria
- [ ] During the captioning phase of a `deep` run containing audio files, the vision
      model is never evicted/offloaded by Whisper or the agent model.
- [ ] Audio transcripts/summaries survive interruption and resume, matching the video
      pipeline's checkpoint semantics.

---

## [8] extractor_version is stored but nothing ever uses it — the documented re-extraction capability has no mechanism

**Class:** arch · **Suggested labels:** `type:feature-enhancement`, `priority:p2` · **Files:** `file_index/index.py:57`, `README.md:131`

### Problem
The README promises "Extractions are versioned per stage (`content.extractor_version`)
so any stage can be re-run later with a better model without touching the others."
The version is written on every `store_content` call, but no code path ever *reads*
or compares it: nothing re-enqueues files whose stored version differs from the
current extractor VERSION constant, and there is no CLI to trigger it. Bumping
`image.VERSION` or swapping the vision model does nothing for already-processed
files. Related: `audio.VERSION = "whisper-large-v3-1.0"` bakes a model name into the
constant while the actual model is config-driven, so a model change isn't even
representable in the version.

### Proposed Solution
Add a `reindex` CLI command (or a `scan --refresh-stage <stage>` flag) that compares
`content.extractor_version` against current constants (and, for model-dependent
stages, the configured model name) and re-enqueues mismatched files at the right tier
with the right queue `kind`. Derive model-dependent versions from config (e.g.
`f"vlm-{model}-1.2"`). Alternatively, drop the claim from the README until built.

### Acceptance Criteria
- [ ] After bumping a stage's version (or changing its model), a documented command
      re-processes exactly the files whose stored version mismatches, leaving other
      stages untouched.
- [ ] README matches the implemented behavior.

---

## [9] No purge/lifecycle for soft-deleted data: excluded files' content persists in the DB forever; thumbnail cache never pruned

**Class:** debt · **Suggested labels:** `type:chore`, `priority:p2` · **Files:** `file_index/index.py:208`, `file_index/crawler.py:91`, `file_index/web.py:349`

### Problem
Deletion is soft everywhere (`deleted=1`), and nothing ever reclaims the underlying
data: `content` bodies, FTS rows, `chunks`, and `chunk_vec` embeddings of deleted or
excluded files remain in `index.db` indefinitely. Two consequences:

- **Privacy expectation mismatch:** `file-index exclude <sensitive-dir>` reports
  "N removed from index", but every extracted text, caption, and transcript of those
  files remains readable in the DB (and keeps consuming FTS/vec space). Reversibility
  is documented, but "excluded yet fully retained forever" deserves an explicit
  opt-out.
- **Unbounded growth:** high-churn roots accumulate dead rows; the thumbnail cache
  (`thumbs/<id>-<mtime>.jpg`) similarly accumulates a new file per content change and
  is never pruned.

### Proposed Solution
Add a `purge` command (or `exclude --purge` / `scan --gc`) that hard-deletes rows for
files with `deleted=1` older than a threshold — including FTS and `chunk_vec` mirrors
— and prunes thumbs whose `<id>-<mtime>` no longer matches a live file. Foreign keys
with `ON DELETE CASCADE` already cover `content`/`chunks` if `files` rows are deleted;
the FTS and vec mirrors need explicit cleanup (see finding [1]).

### Acceptance Criteria
- [ ] A documented command removes all stored content of soft-deleted files from
      `index.db` (verified by querying content/chunks/FTS/vec afterwards).
- [ ] Thumbnail files for deleted/re-modified files are removed by the same or a
      related sweep.

---

## [10] Duplicated helpers are drifting: two _fts_escape implementations differ, strip_think and timestamp formatting each exist twice

**Class:** debt · **Suggested labels:** `type:chore`, `priority:p2` · **Files:** `file_index/index.py:553`, `file_index/web.py:38`, `file_index/agent.py:185`, `file_index/queue.py:640`, `file_index/extractors/audio.py:69`, `file_index/search.py:23`

### Problem
Three helpers are copy-pasted and already diverging:
- `_fts_escape`: `index.py:553` splits on `str.split()`, `web.py:38` on `re.split(r"\s+")`
  — currently equivalent-ish, but a fix in one (e.g. handling `-`/`*` prefixes) won't
  reach the other.
- `strip_think` (`agent.py:185`) and `_strip_think` (`queue.py:640`) are identical
  qwen3 output scrubbers.
- `_ts` (`audio.py:69`) and `format_ts` (`search.py:23`) are the same formatter, and
  `queue.py:615` reaches into another module's private helper (`audio_ex._ts`).

### Proposed Solution
Create a small shared module (e.g. `file_index/util.py`) holding `fts_escape`,
`strip_think`, and `format_ts`; update the six call sites; make `format_transcript`
use the shared formatter. Pure mechanical refactor.

### Acceptance Criteria
- [ ] One definition of each helper, imported everywhere it's used; no
      private-cross-module imports remain.
- [ ] Existing tests (`test_fts_escape_neutralizes_syntax`, chunking/transcript
      tests) pass unchanged.

---

## [11] Safety-critical paths are untested: agent whitelist, organize apply guards, watcher

**Class:** tests · **Suggested labels:** `type:chore`, `priority:p2` · **Files:** `file_index/agent.py:120`, `file_index/agent.py:294`, `file_index/cli.py:335`, `file_index/watch.py`

### Problem
The test suite covers the pipeline well (crawler, queue resume, dedup, deferred
summaries, VLM JSON, web store) but has zero coverage of the code enforcing the
project's safety model, which is its main selling point:
- `AgentTools._check_path` (whitelist enforcement for every agent tool) — untested;
  a regression could silently let the LLM read arbitrary paths.
- `propose_organization`'s action filter (`agent.py:294-313`) that discards
  model-proposed paths escaping the target directory — untested against traversal
  (`../`, symlink, absolute-elsewhere) inputs.
- `cli.organize --apply` flow: `_assert_inside`, the target-exists skip, and index
  path updates after moves — untested.
- `watch.py` — entirely untested (debounce, delete handling, exclusion filtering).

### Proposed Solution
Add focused unit tests: whitelist rejection/acceptance for each agent tool (no model
needed — call `AgentTools` directly); `propose_organization` with a mocked client
returning malicious action lists; an `organize --apply` test via `typer.testing`
with a scripted plan; watcher `_flush` tests driving `pending` directly (no real
inotify needed).

### Acceptance Criteria
- [ ] Tests fail if `_check_path` or the organize action filter stops rejecting
      out-of-scope paths (including `..` traversal and symlinks out of a root).
- [ ] Watcher create/modify/delete handling has direct test coverage.

---

## [12] Every organize run re-appends the last 200 audit rows to audit.log

**Class:** debt · **Suggested labels:** `type:bug`, `priority:p2` · **Files:** `file_index/cli.py:340`

### Problem
`_append_audit_file` appends the most recent 200 `audit_log` DB rows to
`audit.log` on every `organize --apply`, with no memory of what was already written.
The second and every later run re-appends rows from earlier runs, so the plain-text
log accumulates duplicates and its ordering no longer reflects when operations
happened — undermining the file's purpose as a trustworthy audit trail.

### Proposed Solution
Track the last mirrored `audit_log.id` in the `meta` table and append only rows with
a greater id; or simpler, write the audit line at the moment each operation is
recorded (inside `Index.audit`) instead of batch-mirroring afterwards.

### Acceptance Criteria
- [ ] Running `organize --apply` twice produces each audit entry exactly once in
      `audit.log`.

---

## Not filed

- **Per-request SQLite connections in the web Store** — `ThreadingHTTPServer` spawns
  a thread (and thus a connection) per HTTP connection; HTTP/1.1 keep-alive amortizes
  it and connections are read-only. Not worth a ticket.
- **Oversized files never refresh in the index** — files above `max_file_size` keep
  stale metadata; intentional consequence of the documented skip.
- **Watcher move-event handling depends on dest-before-src ordering** — works because
  `on_moved` inserts dest first and dicts preserve insertion order; fragile but
  correct today, and finding [11]'s watcher tests would pin it.
- **`OllamaClient.pull` uses `timeout=None`** — interactive init path only; a hang is
  visible to the user.
- **`chat()` ignores `self.keep_alive`** — only `ask` uses chat, outside deep runs;
  harmless today.
