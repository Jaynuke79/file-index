"""Queue workers: pull pending items, run the right extractor, store content +
chunks + embeddings, and checkpoint after every file (single commit per file,
so kill/restart resumes exactly where it left off).
"""

from __future__ import annotations

import logging
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .config import Config
from .embed import Embedder, chunk_segments, chunk_text
from .extractors import image as image_ex
from .extractors import office as office_ex
from .extractors import pdf as pdf_ex
from .extractors import text as text_ex
from .index import PENDING_SUMMARY, PENDING_TRANSCRIPT, Index
from .ollama_client import OllamaClient
from .util import format_ts, strip_think

log = logging.getLogger("file_index.queue")

# Per-kind stage names for the shared transcribe/summarize machinery.
TRANSCRIPT_STAGE = {"audio": "whisper", "video": "video_transcript"}
SUMMARY_STAGE = {"audio": "audio_summary", "video": "video_summary"}

AUDIO_SUMMARY_PROMPT = (
    "Summarize this audio transcript in a short paragraph. /no_think\n\n"
)


def _item_kind(item) -> str:
    """Queue routing kind for a row ('audio', 'video', 'image', 'pdf_scan')."""
    return item["kind"] or item["file_kind"]


class Tier1Worker:
    """Fast pass: text/code/pdf/office extraction + EXIF. No model calls except
    embeddings (skipped gracefully if Ollama is down — FTS still works)."""

    def __init__(self, config: Config, index: Index, embedder: Embedder | None = None):
        self.config = config
        self.index = index
        self.embedder = embedder
        self._embed_available: bool | None = None

    def _embed(self, chunks: list[dict]) -> list[dict]:
        if self.embedder is None:
            return chunks
        if self._embed_available is None:
            self._embed_available = self.embedder.client.ping()
            if not self._embed_available:
                log.warning("Ollama unreachable — indexing without embeddings (FTS only)")
        if self._embed_available:
            return self.embedder.embed_chunks(chunks)
        return chunks

    def run(self, progress_cb=None) -> dict:
        done = failed = 0
        while True:
            item = self.index.next_pending(tier=1)
            if item is None:
                break
            path = Path(item["path"])
            if progress_cb:
                progress_cb(str(path), done, failed)
            try:
                self._process(item, path)
                self.index.mark_done(item["id"])
                self.index.set_tier_status(item["file_id"], 1, "done")
                done += 1
            except FileNotFoundError:
                self.index.mark_failed(item["id"], "file disappeared during processing",
                                       self.config.limits.max_retries)
                self.index.mark_deleted(item["file_id"])
                failed += 1
            except Exception as e:  # noqa: BLE001 — never let one file kill the run
                log.exception("tier1 failed on %s", path)
                self.index.mark_failed(item["id"], f"{type(e).__name__}: {e}",
                                       self.config.limits.max_retries)
                row = self.index.db.execute(
                    "SELECT status FROM queue WHERE id=?", (item["id"],)
                ).fetchone()
                if row and row["status"] == "failed":
                    self.index.set_tier_status(item["file_id"], 1, "failed")
                    failed += 1
            self.index.commit()  # checkpoint after every file
        return {"done": done, "failed": failed}

    def _process(self, item, path: Path) -> None:
        kind = item["file_kind"]
        cap = self.config.limits.text_size_cap
        file_id = item["file_id"]

        if kind in ("text", "code"):
            body = text_ex.extract(path, cap)
            cid = self.index.store_content(file_id, "text", text_ex.VERSION, body)
            self._store_text_chunks(file_id, cid, "text", body)
        elif kind == "pdf":
            body, scanned_pages = pdf_ex.extract(path, cap)
            cid = self.index.store_content(
                file_id, "pdf_text", pdf_ex.VERSION, body,
                meta={"scanned_pages": scanned_pages},
            )
            if body:
                self._store_text_chunks(file_id, cid, "pdf_text", body)
            if scanned_pages:
                # queue for tier-2 VLM OCR of scanned pages
                self.index.enqueue(file_id, 2, "pending_deep", "pdf_scan", item["mtime"])
        elif kind == "office":
            body = office_ex.extract(path, cap)
            cid = self.index.store_content(file_id, "office", office_ex.VERSION, body)
            self._store_text_chunks(file_id, cid, "office", body)
        elif kind == "image":
            exif = image_ex.extract_exif(path)
            exif_text = " ".join(f"{k}:{v}" for k, v in exif.items() if k != "gps")
            self.index.store_content(file_id, "exif", image_ex.VERSION, exif_text or None, meta=exif)
        # audio/video/other: nothing cheap to do in tier 1 beyond stat metadata

    def _store_text_chunks(self, file_id: int, content_id: int, stage: str, body: str) -> None:
        if not body.strip():
            return
        chunks = chunk_text(
            body, self.config.limits.chunk_tokens, self.config.limits.chunk_overlap_tokens
        )
        self.index.store_chunks(file_id, content_id, stage, self._embed(chunks))


def _prepare_cpu(kind: str, path: Path, fallback_interval_s: float) -> dict | None:
    """CPU-only preparation for an upcoming queue item, run on a background
    thread while the GPU processes the current file: image transcode/downscale,
    video scene detection. Must not touch the DB (SQLite conn is not shared).
    """
    if kind == "image":
        tdir = tempfile.TemporaryDirectory(prefix="file-index-prep-")
        try:
            vlm_path = image_ex.prepare_for_vlm(path, Path(tdir.name))
        except Exception:
            tdir.cleanup()
            raise
        return {"tmpdir": tdir, "vlm_path": vlm_path}
    if kind == "video":
        from .extractors import video as video_ex

        return {"scenes": video_ex.detect_scenes(path, fallback_interval_s)}
    return None


class Tier2Worker:
    """Deep pass: VLM images, scanned PDFs, Whisper audio, full video pipeline.

    While the GPU works on the current file, `deep.prefetch_files` background
    threads run the CPU-heavy prep of the next queue items (image decode and
    downscale, video scene detection) so neither processor waits on the other.
    """

    def __init__(self, config: Config, index: Index, client: OllamaClient | None = None):
        self.config = config
        self.index = index
        self.client = client or OllamaClient(config.models.ollama_url)
        self.embedder = Embedder(config, self.client)

    def _version(self, stage: str) -> str:
        """Version to stamp on a stage written now — includes the configured
        model for model-dependent stages so `reindex` can spot a model swap."""
        from .reindex import stage_version

        return stage_version(stage, self.config)

    def pending_count(self) -> int:
        row = self.index.db.execute(
            "SELECT COUNT(*) n FROM queue WHERE tier=2 "
            "AND status IN ('pending_deep', 'pending_transcript', 'pending_summary')"
        ).fetchone()
        return row["n"]

    def run(self, progress_cb=None, stop_check=None) -> dict:
        self.client.require()
        # Pin models in VRAM for the whole run: a long CPU stretch (Whisper,
        # scene detection) must not let Ollama idle-evict a ~20 GB model.
        # The CLI unloads everything when the run ends.
        self.client.keep_alive = -1
        done = failed = 0
        priority = list(self.config.deep.priority)
        if "pdf_scan" not in priority:
            priority.insert(0, "pdf_scan")
        start_time = time.time()
        prefetch_n = max(0, self.config.deep.prefetch_files)
        executor = (
            ThreadPoolExecutor(max_workers=prefetch_n, thread_name_prefix="prefetch")
            if prefetch_n
            else None
        )
        prefetched: dict[int, Future] = {}
        transcriber = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcribe")
        in_flight: dict[int, tuple[Future, dict]] = {}  # queue_id -> (future, item)
        try:
            while True:
                if stop_check and stop_check():
                    break
                self._drain_transcripts(in_flight, wait=False)
                self._pump_transcripts(transcriber, in_flight)
                item = self.index.next_pending(
                    tier=2, kind_priority=priority, newest_first=self.config.deep.newest_first
                )
                if item is None:
                    # Captions are done; wait out the remaining transcriptions.
                    if in_flight:
                        self._drain_transcripts(in_flight, wait=True)
                        continue
                    break
                if executor:
                    self._top_up_prefetch(executor, prefetched, priority,
                                          current_id=item["id"], depth=prefetch_n)
                prep = self._take_prep(prefetched, item["id"])
                path = Path(item["path"])
                if progress_cb:
                    elapsed = time.time() - start_time
                    remaining = self.pending_count()
                    rate = done / elapsed if elapsed > 3 and done else None
                    eta = remaining / rate if rate else None
                    progress_cb(str(path), done, remaining, eta)
                try:
                    outcome = self._process(item, path, prep)
                    if outcome == "defer_transcript":
                        self.index.set_queue_status(item["id"], PENDING_TRANSCRIPT)
                    elif outcome == "defer_summary":
                        self.index.set_queue_status(item["id"], PENDING_SUMMARY)
                    else:
                        self.index.mark_done(item["id"])
                        self.index.set_tier_status(item["file_id"], 2, "done")
                    done += 1
                except FileNotFoundError:
                    self.index.mark_failed(item["id"], "file disappeared during processing",
                                           self.config.limits.max_retries)
                    self.index.mark_deleted(item["file_id"])
                    failed += 1
                except Exception as e:  # noqa: BLE001
                    log.exception("tier2 failed on %s", path)
                    self.index.mark_failed(item["id"], f"{type(e).__name__}: {e}",
                                           self.config.limits.max_retries)
                    row = self.index.db.execute(
                        "SELECT status FROM queue WHERE id=?", (item["id"],)
                    ).fetchone()
                    if row and row["status"] == "failed":
                        self.index.set_tier_status(item["file_id"], 2, "failed")
                        failed += 1
                finally:
                    if prep and prep.get("tmpdir"):
                        prep["tmpdir"].cleanup()
                self.index.commit()  # checkpoint after every file
        finally:
            if executor:
                executor.shutdown(wait=False, cancel_futures=True)
                for fut in prefetched.values():
                    if fut.done() and not fut.cancelled() and fut.exception() is None:
                        prep = fut.result()
                        if prep and prep.get("tmpdir"):
                            prep["tmpdir"].cleanup()
            transcriber.shutdown(wait=False, cancel_futures=True)
            # A queued-but-cancelled job stays pending_transcript for the next
            # run; a running one is about to block interpreter exit anyway, so
            # wait for it and keep its result.
            while in_flight:
                for qid in [q for q, (f, _) in in_flight.items() if f.cancelled()]:
                    in_flight.pop(qid)
                if in_flight:
                    self._drain_transcripts(in_flight, wait=True)
        # All vision work is done — summarize every deferred video with the
        # agent model loaded once, instead of swapping models per video.
        s_failed = self._summary_sweep(progress_cb, stop_check, done)
        failed += s_failed
        return {"done": done, "failed": failed}

    def _pump_transcripts(
        self,
        transcriber: ThreadPoolExecutor,
        in_flight: dict[int, tuple[Future, dict]],
        max_in_flight: int = 2,
    ) -> None:
        """Feed the background transcription thread from pending_transcript
        rows (captions already stored, this run or an interrupted earlier one).

        Whisper runs on CPU here regardless of deep.whisper_device: it is off
        the GPU's critical path, and letting it grab VRAM first pushes the
        (pinned) vision model into partial CPU offload, slowing every caption
        ~30x. The configured device still applies to inline transcription.
        """
        from .extractors import audio as audio_ex
        from .extractors import video as video_ex

        if len(in_flight) >= max_in_flight:
            return
        rows = self.index.peek_pending(
            tier=2, status=PENDING_TRANSCRIPT,
            newest_first=self.config.deep.newest_first,
            limit=max_in_flight + len(in_flight),
        )
        for it in rows:
            if len(in_flight) >= max_in_flight:
                break
            if it["id"] in in_flight:
                continue
            # Audio files are transcribed directly; videos need their audio
            # track demuxed first.
            fn = (
                audio_ex.transcribe
                if _item_kind(it) == "audio"
                else video_ex.transcribe_video
            )
            fut = transcriber.submit(
                fn, Path(it["path"]), self.config.models.whisper, "cpu", "int8",
            )
            in_flight[it["id"]] = (fut, it)

    def _drain_transcripts(
        self, in_flight: dict[int, tuple[Future, dict]], wait: bool
    ) -> None:
        """Store finished transcriptions and advance their queue items to
        pending_summary. With wait=True, block until at least one finishes."""
        if not in_flight:
            return
        if wait:
            from concurrent.futures import FIRST_COMPLETED
            from concurrent.futures import wait as futures_wait

            futures_wait([f for f, _ in in_flight.values()], return_when=FIRST_COMPLETED)
        for qid in [q for q, (f, _) in in_flight.items() if f.done() and not f.cancelled()]:
            fut, item = in_flight.pop(qid)
            path = Path(item["path"])
            try:
                transcript = fut.result()
            except Exception as e:  # noqa: BLE001 — a lost transcript is not fatal
                log.warning("whisper failed for %s: %s", path, e)
                transcript = None
            if transcript and transcript["segments"]:
                self._store_transcript(item["file_id"], transcript, _item_kind(item))
            self.index.set_queue_status(qid, PENDING_SUMMARY)
            self.index.commit()

    def _store_transcript(self, file_id: int, t: dict, kind: str = "video") -> None:
        from .extractors import audio as audio_ex

        stage = TRANSCRIPT_STAGE[kind]
        tcid = self.index.store_content(
            file_id, stage, self._version(stage),
            audio_ex.format_transcript(t["segments"]),
            meta={"language": t["language"], "duration": t["duration"],
                  "segments": t["segments"]},
        )
        tchunks = chunk_segments(t["segments"], self.config.limits.chunk_tokens)
        self.index.store_chunks(file_id, tcid, stage,
                                self.embedder.embed_chunks(tchunks))

    def _summary_sweep(self, progress_cb, stop_check, done_so_far: int) -> int:
        failed = 0
        while True:
            if stop_check and stop_check():
                break
            item = self.index.next_pending(
                tier=2, status=PENDING_SUMMARY,
                newest_first=self.config.deep.newest_first,
            )
            if item is None:
                break
            path = Path(item["path"])
            if progress_cb:
                progress_cb(str(path), done_so_far, self.pending_count(), None)
            try:
                self._summarize_deferred(item, path)
                self.index.mark_done(item["id"])
                self.index.set_tier_status(item["file_id"], 2, "done")
            except Exception as e:  # noqa: BLE001 — never let one file kill the sweep
                log.exception("deferred summary failed on %s", path)
                self.index.mark_failed(item["id"], f"{type(e).__name__}: {e}",
                                       self.config.limits.max_retries,
                                       retry_status=PENDING_SUMMARY)
                row = self.index.db.execute(
                    "SELECT status FROM queue WHERE id=?", (item["id"],)
                ).fetchone()
                if row and row["status"] == "failed":
                    self.index.set_tier_status(item["file_id"], 2, "failed")
                    failed += 1
            self.index.commit()  # checkpoint after every summary
        return failed

    def _summarize_deferred(self, item, path: Path) -> None:
        """Generate + store the summary for a file whose captions/transcript
        were stored earlier in the run (or a previous, interrupted run)."""
        from .extractors import video as video_ex

        file_id = item["file_id"]
        if _item_kind(item) == "audio":
            return self._summarize_deferred_audio(item, path)
        scenes = self.index.get_content(file_id, "video_scenes")
        transcript = self.index.get_content(file_id, "video_transcript")
        captions_text = (scenes[0]["body"] if scenes else "") or "(no captions available)"
        transcript_text = (transcript[0]["body"] if transcript else "") or "(no speech / no audio track)"
        summary = strip_think(video_ex.summarize_video(
            self.client, self.config.models.agent, captions_text, transcript_text,
            context=self._neighbor_context(file_id, path),
        ))
        if summary:
            scid = self.index.store_content(
                file_id, "video_summary", self._version("video_summary"), summary
            )
            schunks = chunk_text(summary, self.config.limits.chunk_tokens,
                                 self.config.limits.chunk_overlap_tokens)
            self.index.store_chunks(file_id, scid, "video_summary",
                                    self.embedder.embed_chunks(schunks))

    def _summarize_deferred_audio(self, item, path: Path) -> None:
        """Agent-model summary over an audio file's stored transcript. Runs in
        the end-of-run sweep with the agent model loaded once, instead of
        swapping it against the vision model per file."""
        from .extractors import audio as audio_ex

        file_id = item["file_id"]
        rows = self.index.get_content(file_id, "whisper")
        transcript_text = (rows[0]["body"] if rows else "") or ""
        if not transcript_text.strip():
            return  # silence / no speech: nothing to summarize
        summary = strip_think(
            self.client.generate(
                self.config.models.agent,
                AUDIO_SUMMARY_PROMPT + transcript_text[:30000],
            ).strip()
        )
        if summary:
            scid = self.index.store_content(
                file_id, "audio_summary", self._version("audio_summary"), summary
            )
            schunks = chunk_text(summary, self.config.limits.chunk_tokens,
                                 self.config.limits.chunk_overlap_tokens)
            self.index.store_chunks(file_id, scid, "audio_summary",
                                    self.embedder.embed_chunks(schunks))

    def _top_up_prefetch(
        self,
        executor: ThreadPoolExecutor,
        prefetched: dict[int, Future],
        priority: list[str],
        current_id: int,
        depth: int,
    ) -> None:
        """Queue CPU prep for the next `depth` items after the current one."""
        upcoming = self.index.peek_pending(
            tier=2, kind_priority=priority,
            newest_first=self.config.deep.newest_first, limit=depth + 1,
        )
        for it in upcoming:
            qid = it["id"]
            kind = it["kind"] or it["file_kind"]
            if qid == current_id or qid in prefetched or kind not in ("image", "video"):
                continue
            prefetched[qid] = executor.submit(
                _prepare_cpu, kind, Path(it["path"]),
                self.config.deep.video_fallback_interval_s,
            )

    @staticmethod
    def _take_prep(prefetched: dict[int, Future], queue_id: int) -> dict | None:
        """Collect this item's prefetched prep, waiting if it is still running
        (the work has already started — waiting beats redoing it inline)."""
        fut = prefetched.pop(queue_id, None)
        if fut is None:
            return None
        try:
            return fut.result()
        except Exception as e:  # noqa: BLE001 — prep is best-effort
            log.debug("prefetch failed (%s); processing inline", e)
            return None

    def _process(self, item, path: Path, prep: dict | None = None) -> str | None:
        """Returns "defer_transcript"/"defer_summary" when the item must move
        to a later stage instead of being marked done, else None."""
        if not path.exists():
            raise FileNotFoundError(path)
        if self._reuse_duplicate(item, path):
            return None
        kind = item["kind"] or item["file_kind"]
        if kind == "image":
            self._process_image(item, path, prep)
        elif kind == "pdf_scan":
            self._process_pdf_scan(item, path)
        elif kind == "audio":
            return self._process_audio(item, path)
        elif kind == "video":
            return self._process_video(item, path, prep)
        else:
            log.info("no tier-2 handler for kind=%s (%s)", kind, path)
        return None

    def _reuse_duplicate(self, item, path: Path) -> bool:
        """A byte-identical file (same hash) was already deep-processed under
        another path: copy its results instead of re-running the models."""
        row = self.index.db.execute(
            "SELECT f2.id, f2.path FROM files f "
            "JOIN files f2 ON f2.hash=f.hash AND f2.id != f.id "
            "JOIN queue q2 ON q2.file_id=f2.id AND q2.tier=2 AND q2.status='done' "
            "WHERE f.id=? AND f.hash IS NOT NULL AND f2.deleted=0 LIMIT 1",
            (item["file_id"],),
        ).fetchone()
        if not row:
            return False
        n = self.index.clone_tier2_content(row["id"], item["file_id"])
        if n:
            log.info("reused deep results of identical %s for %s", row["path"], path)
        return n > 0

    def _process_image(self, item, path: Path, prep: dict | None = None) -> None:
        data, raw, degraded = image_ex.analyze_image(
            self.client, self.config.models.vision, path,
            prepared=prep.get("vlm_path") if prep else None,
        )
        if degraded:
            self.index.store_content(
                item["file_id"], "vlm_image", self._version("vlm_image"), raw, degraded=True
            )
            return
        body = image_ex.vlm_body_text(data)
        cid = self.index.store_content(
            item["file_id"], "vlm_image", self._version("vlm_image"), body, meta=data
        )
        chunks = chunk_text(body, self.config.limits.chunk_tokens,
                            self.config.limits.chunk_overlap_tokens)
        self.index.store_chunks(item["file_id"], cid, "vlm_image",
                                self.embedder.embed_chunks(chunks))

    def _neighbor_context(self, file_id: int, path: Path) -> str:
        """Digest of already-summarized videos in the same folder, used to prime
        the captioner/summarizer when a folder holds many similar clips (same
        game, same recurring people). Empty string when disabled or no siblings.
        """
        n = self.config.deep.neighbor_context
        if n <= 0:
            return ""
        prefix = str(path.parent) + "/"
        esc = prefix.replace("\\", r"\\").replace("%", r"\%").replace("_", r"\_")
        rows = self.index.db.execute(
            "SELECT c.body FROM content c JOIN files f ON f.id=c.file_id "
            "WHERE c.stage='video_summary' AND f.id != ? AND f.deleted=0 "
            "AND f.path LIKE ? ESCAPE '\\' AND f.path NOT LIKE ? ESCAPE '\\' "
            "ORDER BY f.mtime DESC LIMIT ?",
            (file_id, esc + "%", esc + "%/%", n),
        ).fetchall()
        parts = []
        for r in rows:
            if r["body"]:
                parts.append("- " + " ".join(r["body"].split())[:300])
        if len(parts) < n:
            # Siblings captioned this run but not yet summarized (summaries are
            # deferred to the end-of-run sweep): use their scene captions.
            more = self.index.db.execute(
                "SELECT c.body FROM content c JOIN files f ON f.id=c.file_id "
                "WHERE c.stage='video_scenes' AND f.id != ? AND f.deleted=0 "
                "AND f.path LIKE ? ESCAPE '\\' AND f.path NOT LIKE ? ESCAPE '\\' "
                "AND NOT EXISTS (SELECT 1 FROM content c2 "
                "                WHERE c2.file_id=c.file_id AND c2.stage='video_summary') "
                "ORDER BY f.mtime DESC LIMIT ?",
                (file_id, esc + "%", esc + "%/%", n - len(parts)),
            ).fetchall()
            for r in more:
                if r["body"]:
                    parts.append("- (scene captions) " + " ".join(r["body"].split())[:300])
        return "\n".join(parts)

    def _process_pdf_scan(self, item, path: Path) -> None:
        import tempfile

        meta_rows = self.index.get_content(item["file_id"], "pdf_text")
        scanned_pages: list[int] = []
        if meta_rows and meta_rows[0]["meta"]:
            import json as _json

            scanned_pages = _json.loads(meta_rows[0]["meta"]).get("scanned_pages", [])
        if not scanned_pages:
            return
        parts = []
        with tempfile.TemporaryDirectory(prefix="file-index-pdf-") as tmp:
            for pno in scanned_pages[:50]:  # sanity cap per document
                img = pdf_ex.rasterize_page(path, pno, Path(tmp) / f"p{pno}.png")
                data, raw, degraded = image_ex.analyze_image(
                    self.client, self.config.models.vision, img
                )
                text = image_ex.vlm_body_text(data) if data else raw
                parts.append(f"[page {pno + 1}]\n{text}")
        body = "\n\n".join(parts)
        cid = self.index.store_content(
            item["file_id"], "pdf_scan_vlm", self._version("pdf_scan_vlm"), body
        )
        chunks = chunk_text(body, self.config.limits.chunk_tokens,
                            self.config.limits.chunk_overlap_tokens)
        self.index.store_chunks(item["file_id"], cid, "pdf_scan_vlm",
                                self.embedder.embed_chunks(chunks))

    def _process_audio(self, item, path: Path) -> str | None:
        """With summaries deferred (the default), audio does no work on the
        critical path at all: transcription happens on the background CPU
        thread and the summary in the end-of-run sweep, so neither Whisper nor
        the agent model ever competes with the pinned vision model."""
        from .extractors import audio as audio_ex

        if self.config.deep.defer_video_summaries:
            return "defer_transcript"

        result = audio_ex.transcribe(
            path,
            self.config.models.whisper,
            self.config.deep.whisper_device,
            self.config.deep.whisper_compute_type,
        )
        transcript_text = audio_ex.format_transcript(result["segments"])
        cid = self.index.store_content(
            item["file_id"], "whisper", self._version("whisper"), transcript_text,
            meta={"language": result["language"], "duration": result["duration"],
                  "segments": result["segments"]},
        )
        chunks = chunk_segments(result["segments"], self.config.limits.chunk_tokens)
        self.index.store_chunks(item["file_id"], cid, "whisper",
                                self.embedder.embed_chunks(chunks))
        # summary via agent model
        if result["text"].strip():
            try:
                summary = self.client.generate(
                    self.config.models.agent,
                    "Summarize this audio transcript in a short paragraph. /no_think\n\n"
                    + result["text"][:30000],
                ).strip()
                summary = strip_think(summary)
                if summary:
                    scid = self.index.store_content(
                        item["file_id"], "audio_summary", self._version("audio_summary"), summary
                    )
                    schunks = chunk_text(summary, self.config.limits.chunk_tokens,
                                         self.config.limits.chunk_overlap_tokens)
                    self.index.store_chunks(item["file_id"], scid, "audio_summary",
                                            self.embedder.embed_chunks(schunks))
            except Exception as e:  # noqa: BLE001 — transcript already stored
                log.warning("audio summary failed for %s: %s", path, e)

    def _process_video(self, item, path: Path, prep: dict | None = None) -> str | None:
        from .extractors import audio as audio_ex
        from .extractors import video as video_ex

        defer = self.config.deep.defer_video_summaries
        result = video_ex.process_video(
            path,
            self.client,
            vision_model=self.config.models.vision,
            agent_model=self.config.models.agent,
            frames_per_scene=self.config.deep.video_frames_per_scene,
            fallback_interval_s=self.config.deep.video_fallback_interval_s,
            whisper_model=self.config.models.whisper,
            whisper_device=self.config.deep.whisper_device,
            whisper_compute_type=self.config.deep.whisper_compute_type,
            scenes=prep.get("scenes") if prep else None,
            frame_workers=self.config.deep.video_frame_workers,
            context=self._neighbor_context(item["file_id"], path),
            summarize=not defer,
            transcribe=not defer,
            max_scenes=self.config.deep.video_max_scenes,
            dedup_frames=self.config.deep.video_dedup_frames,
        )
        file_id = item["file_id"]

        # per-scene captions: body is the timestamped caption text, chunks carry
        # timestamp ranges so search hits resolve to a moment in the video
        scene_lines = []
        scene_chunks = []
        for s in result["scenes"]:
            if not s["captions"]:
                continue
            cap = " | ".join(s["captions"])
            scene_lines.append(
                f"[{format_ts(s['start'])} - {format_ts(s['end'])}] {cap}"
            )
            scene_chunks.append({"text": cap, "ts_start": s["start"], "ts_end": s["end"]})
        cid = self.index.store_content(
            file_id, "video_scenes", self._version("video_scenes"), "\n".join(scene_lines),
            meta={"scenes": result["scenes"]},
        )
        self.index.store_chunks(file_id, cid, "video_scenes",
                                self.embedder.embed_chunks(scene_chunks))

        if result["transcript"] and result["transcript"]["segments"]:
            self._store_transcript(file_id, result["transcript"])

        summary = strip_think(result["summary"])
        if summary:
            scid = self.index.store_content(
                file_id, "video_summary", self._version("video_summary"), summary
            )
            schunks = chunk_text(summary, self.config.limits.chunk_tokens,
                                 self.config.limits.chunk_overlap_tokens)
            self.index.store_chunks(file_id, scid, "video_summary",
                                    self.embedder.embed_chunks(schunks))
        if defer:
            return "defer_transcript"
        if not summary:
            # Inline summarization failed (it is best-effort inside
            # process_video). Captions and transcript are already stored, so
            # hand the file to the end-of-run sweep rather than marking it
            # done with no summary and never retrying.
            return "defer_summary"
        return None

