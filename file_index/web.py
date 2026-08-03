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


def _preview_video(src: Path, out: Path) -> Path | None:
    """H.264/AAC MP4 for video that browsers won't play in <video> — .mov
    from phones is commonly HEVC, which Chrome/Firefox can't decode."""
    import subprocess

    tmp = out.with_name(f"{out.name}.{os.getpid()}.{threading.get_ident()}.tmp.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-c:a", "aac", "-movflags", "+faststart", str(tmp)],
            capture_output=True, timeout=600, check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        log.warning("preview transcode failed for %s: %s", src, e)
        tmp.unlink(missing_ok=True)
        return None
    os.replace(tmp, out)
    return out


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

    def _serve_media(self, file_id: int) -> None:
        info = self.store.file_path(file_id)
        if info is None:
            return self._error(404, "unknown file id")
        src, _kind = info
        if not src.is_file():
            return self._error(404, "file missing on disk")
        ext = src.suffix.lower()
        ctype = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
        if ext in PREVIEW_IMAGE_EXTS or ext in PREVIEW_VIDEO_EXTS:
            preview_ext = ".jpg" if ext in PREVIEW_IMAGE_EXTS else ".mp4"
            out = self.preview_dir / f"{file_id}-{int(src.stat().st_mtime)}{preview_ext}"
            if not out.exists() and _make_preview(src, out) is None:
                return self._error(404, "preview unavailable for this file")
            src = out
            ctype = "image/jpeg" if preview_ext == ".jpg" else "video/mp4"
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


def make_server(cfg: Config, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
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
            "config": cfg,
            "jobs": JobRunner(),
            "csrf_token": secrets.token_urlsafe(32),
            "writable": writable,
        },
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.job_runner = handler.jobs  # so serve() can stop jobs on shutdown
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
    cfg: Config, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True
) -> None:
    server = make_server(cfg, host, port)
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
        server.server_close()
