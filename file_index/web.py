"""Local web viewer for the index: browse files alongside their captions.

Read-only: the server opens the DB in read-only mode (safe to run while
`scan`/`deep` are working) and only ever serves files that are in the index.
Binds to localhost by default; nothing is exposed beyond the machine.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import sqlite3
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import Config
from .ui import WEB_UI
from .util import fts_escape as _fts_escape

log = logging.getLogger("file_index.web")

THUMB_SIZE = 512
PAGE_LIMIT_MAX = 200

# Extensions browsers won't render/play natively even though the index (and
# thumbnails) handle them fine. Transcoded to a browser-native format on
# first request and cached in `previews/`, mirroring the `thumbs/` cache.
PREVIEW_IMAGE_EXTS = {".heic"}
PREVIEW_VIDEO_EXTS = {".mov"}

# Stages that hold a human-readable caption, in preference order. The first
# four are "real" captions (used for the captioned-only filter/count);
# video_scenes covers videos whose deferred summary hasn't been generated yet.
CAPTION_STAGES = (
    "vlm_image", "video_summary", "audio_summary", "pdf_scan_vlm",
    "video_scenes", "pdf_text", "text",
)


class Store:
    """Read-only, thread-safe (per-thread connection) view of the index DB."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()

    @property
    def db(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    # ---------- queries ----------

    def summary(self) -> dict:
        kinds = {
            r["kind"]: r["n"]
            for r in self.db.execute(
                "SELECT kind, count(*) AS n FROM files WHERE deleted=0 GROUP BY kind"
            )
        }
        captioned = self.db.execute(
            "SELECT count(DISTINCT c.file_id) FROM content c "
            "JOIN files f ON f.id=c.file_id AND f.deleted=0 "
            "WHERE c.stage IN (?,?,?,?)",
            CAPTION_STAGES[:4],
        ).fetchone()[0]
        return {"kinds": kinds, "captioned": captioned, "total": sum(kinds.values())}

    def list_files(
        self,
        q: str = "",
        kind: str = "",
        captioned_only: bool = False,
        offset: int = 0,
        limit: int = 60,
    ) -> dict:
        limit = max(1, min(limit, PAGE_LIMIT_MAX))
        where, params = ["f.deleted=0"], []
        if kind:
            where.append("f.kind=?")
            params.append(kind)
        if captioned_only:
            where.append(
                "EXISTS (SELECT 1 FROM content cc WHERE cc.file_id=f.id "
                "AND cc.stage IN (?,?,?,?))"
            )
            params.extend(CAPTION_STAGES[:4])
        cols = "f.id, f.path, f.size, f.mtime, f.kind, f.tier2_status"

        if q.strip():
            match = _fts_escape(q)
            base = (
                "FROM content_fts JOIN content c ON c.id=content_fts.rowid "
                "JOIN files f ON f.id=c.file_id "
                f"WHERE content_fts MATCH ? AND {' AND '.join(where)}"
            )
            try:
                total = self.db.execute(
                    f"SELECT count(DISTINCT f.id) {base}", [match, *params]
                ).fetchone()[0]
                rows = self.db.execute(
                    f"SELECT {cols}, min(rank) AS r {base} "
                    "GROUP BY f.id ORDER BY r LIMIT ? OFFSET ?",
                    [match, *params, limit, offset],
                ).fetchall()
            except sqlite3.OperationalError:
                total, rows = 0, []
        else:
            base = f"FROM files f WHERE {' AND '.join(where)}"
            total = self.db.execute(f"SELECT count(*) {base}", params).fetchone()[0]
            rows = self.db.execute(
                f"SELECT {cols} {base} ORDER BY f.mtime DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()

        files = [dict(r) for r in rows]
        self._attach_captions(files)
        return {"total": total, "offset": offset, "files": files}

    def _attach_captions(self, files: list[dict]) -> None:
        if not files:
            return
        by_id = {f["id"]: f for f in files}
        marks = ",".join("?" * len(by_id))
        stage_marks = ",".join("?" * len(CAPTION_STAGES))
        rows = self.db.execute(
            f"SELECT file_id, stage, body, meta, degraded FROM content "
            f"WHERE file_id IN ({marks}) AND stage IN ({stage_marks})",
            [*by_id.keys(), *CAPTION_STAGES],
        ).fetchall()
        best: dict[int, sqlite3.Row] = {}
        rank = {s: i for i, s in enumerate(CAPTION_STAGES)}
        for r in rows:
            cur = best.get(r["file_id"])
            if cur is None or rank[r["stage"]] < rank[cur["stage"]]:
                best[r["file_id"]] = r
        for fid, r in best.items():
            f = by_id[fid]
            caption, vlm_type = r["body"] or "", None
            if r["stage"] == "vlm_image" and r["meta"]:
                try:
                    meta = json.loads(r["meta"])
                    caption = meta.get("description") or caption
                    vlm_type = meta.get("type")
                except (json.JSONDecodeError, AttributeError):
                    pass
            f["caption"] = caption[:400]
            f["caption_stage"] = r["stage"]
            f["degraded"] = bool(r["degraded"])
            if vlm_type:
                f["vlm_type"] = vlm_type

    def file_detail(self, file_id: int) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM files WHERE id=? AND deleted=0", (file_id,)
        ).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["content"] = []
        for c in self.db.execute(
            "SELECT stage, body, meta, degraded, extractor_version FROM content "
            "WHERE file_id=? ORDER BY id",
            (file_id,),
        ):
            entry = dict(c)
            if entry["meta"]:
                try:
                    entry["meta"] = json.loads(entry["meta"])
                except json.JSONDecodeError:
                    pass
            out["content"].append(entry)
        err = self.db.execute(
            "SELECT error FROM queue WHERE file_id=? AND status='failed'", (file_id,)
        ).fetchone()
        out["error"] = err["error"] if err else None
        return out

    def status(self) -> dict:
        """The `status` command as data: queue depth, per-kind counts, failures."""
        queue: dict = {"tier1": {}, "tier2": {}}
        for r in self.db.execute(
            "SELECT tier, status, COUNT(*) n FROM queue GROUP BY tier, status"
        ):
            queue[f"tier{r['tier']}"][r["status"]] = r["n"]
        kinds = [
            {"kind": r["kind"] or "?", "n": r["n"], "size": r["s"] or 0}
            for r in self.db.execute(
                "SELECT kind, COUNT(*) n, SUM(size) s FROM files WHERE deleted=0 "
                "GROUP BY kind ORDER BY n DESC"
            )
        ]
        failures = [
            {"tier": r["tier"], "path": r["path"], "error": (r["error"] or "")[:300],
             "retries": r["retries"]}
            for r in self.db.execute(
                "SELECT q.tier, q.error, q.retries, f.path FROM queue q "
                "JOIN files f ON f.id=q.file_id WHERE q.status='failed' "
                "ORDER BY q.updated_at DESC LIMIT 25"
            )
        ]
        totals = self.db.execute(
            "SELECT (SELECT COUNT(*) FROM files WHERE deleted=0) files, "
            "(SELECT COUNT(*) FROM chunks) chunks, "
            "(SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL) embedded"
        ).fetchone()
        return {
            "queue": queue,
            "kinds": kinds,
            "failures": failures,
            "totals": dict(totals),
            "db_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
        }

    def file_path(self, file_id: int) -> tuple[Path, str] | None:
        row = self.db.execute(
            "SELECT path, kind, mime FROM files WHERE id=? AND deleted=0", (file_id,)
        ).fetchone()
        if row is None:
            return None
        return Path(row["path"]), row["kind"]


# ---------- thumbnails ----------


def prune_thumbs(thumb_dir: Path, live_keys: set[str]) -> int:
    """Delete cached thumbnails that no live file claims. Thumbs are named
    `<file_id>-<mtime>.jpg`, so both deleted files and superseded versions of
    a still-live file are collected. Returns the number removed."""
    if not thumb_dir.is_dir():
        return 0
    removed = 0
    for p in thumb_dir.glob("*.jpg"):
        if p.stem not in live_keys:
            try:
                p.unlink()
                removed += 1
            except OSError as e:  # noqa: PERF203 — best-effort cache cleanup
                log.debug("could not remove stale thumb %s: %s", p, e)
    return removed


def prune_previews(preview_dir: Path, live_keys: set[str]) -> int:
    """Delete cached previews (transcoded HEIC/MOV) that no live file claims.
    Same `<file_id>-<mtime>` keying as `prune_thumbs`, across both the .jpg
    and .mp4 outputs a preview can produce. Returns the number removed."""
    if not preview_dir.is_dir():
        return 0
    removed = 0
    for p in (*preview_dir.glob("*.jpg"), *preview_dir.glob("*.mp4")):
        if p.stem not in live_keys:
            try:
                p.unlink()
                removed += 1
            except OSError as e:  # noqa: PERF203 — best-effort cache cleanup
                log.debug("could not remove stale preview %s: %s", p, e)
    return removed


def _make_thumb(src: Path, kind: str, out: Path) -> Path | None:
    """Generate a JPEG thumbnail for an image/video/pdf. None if unsupported."""
    try:
        if kind == "video":
            return _thumb_video(src, out)
        if kind == "image":
            return _thumb_image(src, out)
        if kind == "pdf":
            return _thumb_pdf(src, out)
    except Exception as e:  # noqa: BLE001 — thumbnails are best-effort
        log.debug("thumbnail failed for %s: %s", src, e)
    return None


def _save_jpeg(img, out: Path, max_size: int | None = None, quality: int = 80) -> Path:
    if max_size:
        img.thumbnail((max_size, max_size))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    tmp = out.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    img.save(tmp, "JPEG", quality=quality)
    os.replace(tmp, out)
    return out


def _thumb_image(src: Path, out: Path) -> Path | None:
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    with Image.open(src) as img:
        return _save_jpeg(img, out, THUMB_SIZE)


def _thumb_video(src: Path, out: Path) -> Path | None:
    import tempfile

    from PIL import Image

    from .extractors.video import extract_frame

    with tempfile.TemporaryDirectory(prefix="file-index-thumb-") as tmp:
        frame = extract_frame(src, 1.0, Path(tmp) / "f.jpg")
        if frame is None:
            frame = extract_frame(src, 0.0, Path(tmp) / "f.jpg")
        if frame is None:
            return None
        with Image.open(frame) as img:
            return _save_jpeg(img, out, THUMB_SIZE)


def _thumb_pdf(src: Path, out: Path) -> Path | None:
    import io

    import fitz  # pymupdf
    from PIL import Image

    with fitz.open(src) as doc:
        if doc.page_count == 0:
            return None
        pix = doc[0].get_pixmap(dpi=72)
        with Image.open(io.BytesIO(pix.tobytes("png"))) as img:
            return _save_jpeg(img, out, THUMB_SIZE)


# ---------- previews (full-size, browser-native) ----------


def _make_preview(src: Path, out: Path) -> Path | None:
    """Convert a file the browser can't render/play natively into something
    it can. Best-effort like `_make_thumb`; None on failure."""
    ext = src.suffix.lower()
    try:
        if ext in PREVIEW_IMAGE_EXTS:
            return _preview_image(src, out)
        if ext in PREVIEW_VIDEO_EXTS:
            return _preview_video(src, out)
    except Exception as e:  # noqa: BLE001 — previews are best-effort
        log.debug("preview failed for %s: %s", src, e)
    return None


def _preview_image(src: Path, out: Path) -> Path | None:
    """Full-resolution JPEG for image formats browsers can't display inline
    (HEIC — the default capture format on modern iPhones)."""
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    with Image.open(src) as img:
        return _save_jpeg(img, out, quality=92)


# Backstop only: veryfast x264 runs well above realtime, so this covers even
# multi-hour footage; a transcode that genuinely takes longer is stuck.
PREVIEW_TRANSCODE_TIMEOUT = 4 * 3600

# In-flight ffmpeg preview processes, so shutdown can kill them — an orphaned
# transcode keeps saturating every core long after browse exits.
_transcode_mu = threading.Lock()
_active_transcodes: set = set()


def kill_active_transcodes() -> None:
    with _transcode_mu:
        procs = list(_active_transcodes)
    for p in procs:
        p.kill()


def _preview_video(src: Path, out: Path) -> Path | None:
    """H.264/AAC MP4 for video that browsers won't play in <video> — .mov
    from phones is commonly HEVC, which Chrome/Firefox can't decode."""
    import subprocess

    tmp = out.with_name(f"{out.name}.{os.getpid()}.{threading.get_ident()}.tmp.mp4")
    try:
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-c:a", "aac", "-movflags", "+faststart", str(tmp)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except OSError as e:
        log.warning("preview transcode failed for %s: %s", src, e)
        return None
    with _transcode_mu:
        _active_transcodes.add(proc)
    try:
        _, err = proc.communicate(timeout=PREVIEW_TRANSCODE_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        log.warning("preview transcode timed out for %s", src)
        tmp.unlink(missing_ok=True)
        return None
    finally:
        with _transcode_mu:
            _active_transcodes.discard(proc)
    if proc.returncode != 0:
        log.warning(
            "preview transcode failed for %s: %s",
            src, err.decode(errors="replace").strip() or f"ffmpeg exit {proc.returncode}",
        )
        tmp.unlink(missing_ok=True)
        return None
    os.replace(tmp, out)
    return out


def preview_path(preview_dir: Path, file_id: int, src: Path) -> Path | None:
    """Cache path of the browser-native preview for src, or None if the
    browser can render src as-is (no preview needed)."""
    ext = src.suffix.lower()
    if ext in PREVIEW_IMAGE_EXTS:
        return preview_dir / f"{file_id}-{int(src.stat().st_mtime)}.jpg"
    if ext in PREVIEW_VIDEO_EXTS:
        return preview_dir / f"{file_id}-{int(src.stat().st_mtime)}.mp4"
    return None


class PreviewManager:
    """Builds each preview exactly once, no matter how many threads ask.

    A per-key lock keeps a media request, a duplicate click, and the pre-warm
    thread from launching concurrent ffmpeg runs for the same file;
    `status_and_kick` gives the UI a non-blocking poll target so the first
    view of a big video shows progress instead of a hung <video> element.
    Failed keys are remembered so a corrupt file is not re-transcoded on
    every poll (a server restart clears the memory and retries)."""

    def __init__(self) -> None:
        self._mu = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._failed: set[str] = set()

    def _lock(self, key: str) -> threading.Lock:
        with self._mu:
            return self._locks.setdefault(key, threading.Lock())

    def ensure(self, src: Path, out: Path) -> Path | None:
        """Return the preview for src, building it if missing. Blocks until
        done; concurrent callers for the same key wait instead of duplicating
        the work."""
        if out.exists():
            return out
        with self._lock(out.name):
            if out.exists():
                return out
            res = _make_preview(src, out)
            with self._mu:
                if res is None:
                    self._failed.add(out.name)
                else:
                    self._failed.discard(out.name)
            return res

    def status_and_kick(self, src: Path, out: Path) -> str:
        """Non-blocking: "ready", "pending", or "failed" — starting a
        background build if none is running yet."""
        if out.exists():
            return "ready"
        key = out.name
        with self._mu:
            if key in self._failed:
                return "failed"
            lock = self._locks.setdefault(key, threading.Lock())
        if not lock.locked():
            threading.Thread(
                target=self.ensure, args=(src, out),
                daemon=True, name=f"preview-{key}",
            ).start()
        return "pending"


class PrewarmState:
    """Thread-safe progress of the browse pre-warm thread, for the control
    panel: how far along it is and whether a stop was requested."""

    def __init__(self) -> None:
        self._mu = threading.Lock()
        self._d = {"active": False, "stopping": False,
                   "total": 0, "done": 0, "built": 0, "current": ""}

    def update(self, **kv) -> None:
        with self._mu:
            self._d.update(kv)

    def snapshot(self) -> dict:
        with self._mu:
            return dict(self._d)


def missing_previews(store: Store, preview_dir: Path) -> list[tuple[Path, Path]]:
    """(source, cache path) for every live video whose browser-native preview
    is not built yet. Newest first — the most likely to be browsed."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for r in store.db.execute(
        "SELECT id, path FROM files WHERE deleted=0 AND kind='video' ORDER BY mtime DESC"
    ):
        src = Path(r["path"])
        if not src.is_file():
            continue
        out = preview_path(preview_dir, r["id"], src)
        if out is not None and not out.exists():
            todo.append((src, out))
    return todo


def _prewarm_previews(
    store: Store, preview_dir: Path, previews: PreviewManager,
    stop: threading.Event, state: PrewarmState,
) -> None:
    """Build every missing video preview in the background so first views are
    instant. Deep runs the same work as its last step (see cli.deep); this
    thread mops up whatever is still missing when browse starts."""
    todo = missing_previews(store, preview_dir)
    if not todo:
        return
    log.info("pre-warming %d video preview(s) in the background", len(todo))
    state.update(active=True, total=len(todo), done=0, built=0)
    built = 0
    try:
        for i, (src, out) in enumerate(todo):
            if stop.is_set():
                log.info("preview pre-warm stopped after %d/%d", i, len(todo))
                return
            state.update(current=src.name)
            if previews.ensure(src, out) is not None:
                built += 1
            state.update(done=i + 1, built=built)
        log.info("preview pre-warm finished: %d/%d built", built, len(todo))
    finally:
        state.update(active=False, stopping=False, current="")


# ---------- HTTP ----------


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class Handler(BaseHTTPRequestHandler):
    store: Store
    thumb_dir: Path
    preview_dir: Path
    previews: PreviewManager
    prewarm_state: PrewarmState
    prewarm_stop: threading.Event
    # When set, requests whose Host header is not listed get 403. This is the
    # standard DNS-rebinding defense for localhost servers: a malicious site
    # rebinding its hostname to 127.0.0.1 sends its own domain as Host and can
    # otherwise read the whole index cross-origin. None = no filtering (used
    # for non-loopback binds, which are network-exposed by explicit choice).
    allowed_hosts: set[str] | None = None
    # Control-panel state. Writes are refused outright unless `writable` (set
    # only for loopback binds) and gated on a token minted at server start and
    # embedded in the page — a rebinding attacker cannot read the page to learn
    # it, so blind cross-origin POSTs fail even if a Host check were bypassed.
    config: Config | None = None
    jobs = None
    csrf_token: str = ""
    writable: bool = False
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # route to logging, not stderr
        log.debug("%s " + fmt, self.address_string(), *args)

    # -- helpers --

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, msg: str) -> None:
        self._json({"error": msg}, status=status)

    def _host_ok(self) -> bool:
        if self.allowed_hosts is None:
            return True
        host = (self.headers.get("Host") or "").strip().lower()
        return host in self.allowed_hosts

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 1 << 20:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON body: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    # -- routing --

    def do_GET(self) -> None:  # noqa: N802 — http.server API
        try:
            self._route()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001 — keep the server alive
            log.exception("request failed: %s", self.path)
            try:
                self._error(500, str(e))
            except OSError:
                pass

    def do_POST(self) -> None:  # noqa: N802 — http.server API
        from .control import SettingsError
        from .jobs import JobError

        try:
            if not self._host_ok():
                return self._error(403, "forbidden Host header")
            if not self.writable:
                return self._error(
                    403,
                    "this server is read-only: settings and jobs are disabled when "
                    "bound to a non-loopback address",
                )
            if self.headers.get("X-CSRF-Token", "") != self.csrf_token:
                return self._error(403, "missing or stale CSRF token — reload the page")
            self._route_post()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (SettingsError, JobError, ValueError) as e:
            self._error(400, str(e))
        except Exception as e:  # noqa: BLE001 — keep the server alive
            log.exception("write request failed: %s", self.path)
            try:
                self._error(500, str(e))
            except OSError:
                pass

    def _route_post(self) -> None:
        from . import control
        from .jobs import JobError

        parts = [p for p in urlparse(self.path).path.split("/") if p]
        body = self._read_json()
        cfg = self.config

        if parts == ["api", "roots"]:
            return self._json({"added": control.add_root(cfg, body.get("path", ""))})
        if parts == ["api", "roots", "remove"]:
            return self._json({"removed": control.remove_root(cfg, body.get("path", ""))})
        if parts == ["api", "excludes"]:
            return self._json({"added": control.add_exclude(cfg, body.get("pattern", ""))})
        if parts == ["api", "excludes", "remove"]:
            return self._json(
                {"removed": control.remove_exclude(cfg, body.get("pattern", ""))}
            )
        if len(parts) == 3 and parts[:2] == ["api", "settings"]:
            changed = control.update_section(cfg, parts[2], body.get("values", {}))
            return self._json({"changed": changed})
        if parts == ["api", "prewarm", "stop"]:
            # Graceful: the in-flight transcode finishes (killing it would
            # burn the work and mark the file failed), then the thread exits.
            if self.prewarm_state.snapshot()["active"]:
                self.prewarm_state.update(stopping=True)
            self.prewarm_stop.set()
            return self._json(self.prewarm_state.snapshot())
        if parts == ["api", "jobs"]:
            job = self.jobs.start(body.get("name", ""), body.get("args") or [])
            return self._json(job.summary(), status=201)
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stop":
            if not parts[2].isdigit():
                raise JobError("bad job id")
            job = self.jobs.stop(int(parts[2]), force=bool(body.get("force")))
            return self._json(job.summary())
        self._error(404, "not found")

    def _route(self) -> None:
        if not self._host_ok():
            return self._error(403, "forbidden Host header")
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        parts = [p for p in url.path.split("/") if p]

        if not parts:
            page = WEB_UI.replace("__CSRF_TOKEN__", self.csrf_token).replace(
                "__WRITABLE__", "true" if self.writable else "false"
            )
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parts == ["api", "settings"]:
            from .control import settings_payload

            payload = settings_payload(self.config)
            payload["writable"] = self.writable
            self._json(payload)
        elif parts == ["api", "fs"]:
            from .control import list_directory

            try:
                self._json(list_directory(qs.get("path", [""])[0] or None))
            except Exception as e:  # noqa: BLE001 — surfaced to the picker
                self._error(400, str(e))
        elif parts == ["api", "status"]:
            self._json(self.store.status())
        elif parts == ["api", "jobs"]:
            self._json({"jobs": self.jobs.list(), "writable": self.writable})
        elif len(parts) == 3 and parts[:2] == ["api", "jobs"] and parts[2].isdigit():
            job = self.jobs.get(int(parts[2]))
            if job is None:
                self._error(404, "unknown job")
            else:
                self._json(job.detail())
        elif parts == ["api", "summary"]:
            self._json(self.store.summary())
        elif parts == ["api", "files"]:
            self._json(
                self.store.list_files(
                    q=qs.get("q", [""])[0],
                    kind=qs.get("kind", [""])[0],
                    captioned_only=qs.get("captioned", [""])[0] == "1",
                    offset=int(qs.get("offset", ["0"])[0]),
                    limit=int(qs.get("limit", ["60"])[0]),
                )
            )
        elif len(parts) == 3 and parts[:2] == ["api", "file"] and parts[2].isdigit():
            detail = self.store.file_detail(int(parts[2]))
            if detail is None:
                self._error(404, "unknown file id")
            else:
                self._json(detail)
        elif parts == ["api", "prewarm"]:
            self._json(self.prewarm_state.snapshot())
        elif len(parts) == 3 and parts[:2] == ["api", "preview"] and parts[2].isdigit():
            self._serve_preview_status(int(parts[2]))
        elif len(parts) == 2 and parts[0] == "thumb" and parts[1].isdigit():
            self._serve_thumb(int(parts[1]))
        elif len(parts) == 2 and parts[0] == "media" and parts[1].isdigit():
            self._serve_media(int(parts[1]))
        else:
            self._error(404, "not found")

    # -- file serving --

    def _serve_thumb(self, file_id: int) -> None:
        info = self.store.file_path(file_id)
        if info is None:
            return self._error(404, "unknown file id")
        src, kind = info
        if not src.exists():
            return self._error(404, "file missing on disk")
        out = self.thumb_dir / f"{file_id}-{int(src.stat().st_mtime)}.jpg"
        if not out.exists() and _make_thumb(src, kind, out) is None:
            return self._error(404, "no thumbnail for this type")
        data = out.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    def _serve_preview_status(self, file_id: int) -> None:
        """Poll target for the UI: is this file's preview ready? Kicks off a
        background transcode on first ask, so the media request that follows
        a "ready" answer is served straight from the cache."""
        info = self.store.file_path(file_id)
        if info is None:
            return self._error(404, "unknown file id")
        src, _kind = info
        if not src.is_file():
            return self._error(404, "file missing on disk")
        out = preview_path(self.preview_dir, file_id, src)
        if out is None:  # browser-native, media/ serves the original
            return self._json({"status": "ready"})
        self._json({"status": self.previews.status_and_kick(src, out)})

    def _serve_media(self, file_id: int) -> None:
        info = self.store.file_path(file_id)
        if info is None:
            return self._error(404, "unknown file id")
        src, _kind = info
        if not src.is_file():
            return self._error(404, "file missing on disk")
        ctype = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
        out = preview_path(self.preview_dir, file_id, src)
        if out is not None:
            if self.previews.ensure(src, out) is None:
                return self._error(404, "preview unavailable for this file")
            src = out
            ctype = "image/jpeg" if out.suffix == ".jpg" else "video/mp4"
        size = src.stat().st_size
        start, end = 0, size - 1
        status = 200
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            status = 206
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
                    if start > int(m.group(2)):
                        # malformed (e.g. bytes=500-100): ignore per RFC 7233
                        status, start, end = 200, 0, size - 1
            else:  # suffix range: last N bytes
                start = max(0, size - int(m.group(2)))
            if start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with src.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(1 << 16, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class _QuietServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that doesn't spray tracebacks when a client hangs
    up mid-connection — browsers reset kept-alive sockets constantly (tab
    refresh, aborted <video> loads), and stock socketserver prints each one."""

    def handle_error(self, request, client_address) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, TimeoutError)):
            log.debug("client %s dropped the connection", client_address)
            return
        super().handle_error(request, client_address)


def make_server(
    cfg: Config, host: str = "127.0.0.1", port: int = 8765, prewarm: bool = False
) -> ThreadingHTTPServer:
    import secrets

    from .jobs import JobRunner

    thumb_dir = cfg.data_dir / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = cfg.data_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    # The control panel edits config and launches jobs, so it is enabled only
    # on a loopback bind. Exposing it on a LAN address would hand anyone who
    # can reach the port the ability to index arbitrary directories.
    writable = is_loopback(host)
    handler = type(
        "BoundHandler",
        (Handler,),
        {
            "store": Store(cfg.db_path),
            "thumb_dir": thumb_dir,
            "preview_dir": preview_dir,
            "previews": PreviewManager(),
            "prewarm_state": PrewarmState(),
            "prewarm_stop": threading.Event(),
            "config": cfg,
            "jobs": JobRunner(),
            "csrf_token": secrets.token_urlsafe(32),
            "writable": writable,
        },
    )
    server = _QuietServer((host, port), handler)
    server.daemon_threads = True
    server.job_runner = handler.jobs  # so serve() can stop jobs on shutdown
    server.prewarm_stop = handler.prewarm_stop  # so serve() can halt the pre-warmer
    if prewarm:
        threading.Thread(
            target=_prewarm_previews,
            args=(handler.store, preview_dir, handler.previews,
                  handler.prewarm_stop, handler.prewarm_state),
            daemon=True, name="preview-prewarm",
        ).start()
    if is_loopback(host):
        actual_port = server.server_address[1]  # resolved when port=0
        allowed = set()
        for h in {"127.0.0.1", "localhost", "[::1]", host.lower()}:
            allowed.add(f"{h}:{actual_port}")
            if actual_port == 80:
                allowed.add(h)
        handler.allowed_hosts = allowed
    return server


def serve(
    cfg: Config, host: str = "127.0.0.1", port: int = 8765,
    open_browser: bool = True, prewarm: bool = True,
) -> None:
    server = make_server(cfg, host, port, prewarm=prewarm)
    url = f"http://{host}:{port}/"
    log.info("browse UI listening on %s", url)
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner = getattr(server, "job_runner", None)
        if runner is not None:
            runner.shutdown()  # don't orphan a scan/deep when the UI stops
        server.prewarm_stop.set()
        kill_active_transcodes()  # free the CPU the moment browse exits
        server.server_close()
