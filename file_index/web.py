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

log = logging.getLogger("file_index.web")

THUMB_SIZE = 512
PAGE_LIMIT_MAX = 200

# Stages that hold a human-readable caption, in preference order. The first
# four are "real" captions (used for the captioned-only filter/count);
# video_scenes covers videos whose deferred summary hasn't been generated yet.
CAPTION_STAGES = (
    "vlm_image", "video_summary", "audio_summary", "pdf_scan_vlm",
    "video_scenes", "pdf_text", "text",
)


def _fts_escape(query: str) -> str:
    """Quote each term so user input is never parsed as FTS5 syntax."""
    terms = [t for t in re.split(r"\s+", query.strip()) if t]
    return " ".join('"' + t.replace('"', '""') + '"' for t in terms)


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
            "SELECT count(DISTINCT file_id) FROM content WHERE stage IN (?,?,?,?)",
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

    def file_path(self, file_id: int) -> tuple[Path, str] | None:
        row = self.db.execute(
            "SELECT path, kind, mime FROM files WHERE id=? AND deleted=0", (file_id,)
        ).fetchone()
        if row is None:
            return None
        return Path(row["path"]), row["kind"]


# ---------- thumbnails ----------


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


def _save_jpeg(img, out: Path) -> Path:
    img.thumbnail((THUMB_SIZE, THUMB_SIZE))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    tmp = out.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    img.save(tmp, "JPEG", quality=80)
    os.replace(tmp, out)
    return out


def _thumb_image(src: Path, out: Path) -> Path | None:
    from PIL import Image

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(src) as img:
        return _save_jpeg(img, out)


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
            return _save_jpeg(img, out)


def _thumb_pdf(src: Path, out: Path) -> Path | None:
    import io

    import fitz  # pymupdf
    from PIL import Image

    with fitz.open(src) as doc:
        if doc.page_count == 0:
            return None
        pix = doc[0].get_pixmap(dpi=72)
        with Image.open(io.BytesIO(pix.tobytes("png"))) as img:
            return _save_jpeg(img, out)


# ---------- HTTP ----------


class Handler(BaseHTTPRequestHandler):
    store: Store
    thumb_dir: Path
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

    def _route(self) -> None:
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        parts = [p for p in url.path.split("/") if p]

        if not parts:
            body = WEB_UI.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
        size = src.stat().st_size
        ctype = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        status = 200
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            status = 206
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
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
    thumb_dir = cfg.data_dir / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    handler = type(
        "BoundHandler", (Handler,), {"store": Store(cfg.db_path), "thumb_dir": thumb_dir}
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
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
        server.server_close()


# ---------- UI (single page, no external assets) ----------

WEB_UI = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>file-index browser</title>
<style>
:root {
  --bg: #101014; --panel: #17171d; --card: #1c1c24; --border: #2a2a35;
  --text: #e8e8ee; --dim: #9a9aa8; --accent: #7aa2f7; --badge: #242430;
}
* { box-sizing: border-box; margin: 0; }
body { background: var(--bg); color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { position: sticky; top: 0; z-index: 10; background: var(--panel);
  border-bottom: 1px solid var(--border); padding: 10px 16px;
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
header h1 { font-size: 15px; font-weight: 600; margin-right: 6px; }
#q { flex: 1 1 220px; max-width: 420px; background: var(--card); color: var(--text);
  border: 1px solid var(--border); border-radius: 8px; padding: 7px 12px; outline: none; }
#q:focus { border-color: var(--accent); }
.chip { background: var(--card); border: 1px solid var(--border); color: var(--dim);
  border-radius: 999px; padding: 4px 12px; cursor: pointer; font-size: 13px; }
.chip.on { color: var(--text); border-color: var(--accent); background: #20283e; }
label.cap { color: var(--dim); display: flex; gap: 6px; align-items: center;
  cursor: pointer; font-size: 13px; }
#count { color: var(--dim); font-size: 13px; margin-left: auto; }
#grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 14px; padding: 16px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  overflow: hidden; cursor: pointer; display: flex; flex-direction: column; }
.card:hover { border-color: var(--accent); }
.thumb { aspect-ratio: 16/10; background: #0c0c10; display: flex;
  align-items: center; justify-content: center; overflow: hidden; }
.thumb img { width: 100%; height: 100%; object-fit: cover; }
.thumb .ph { font-size: 34px; opacity: .45; }
.card .body { padding: 10px 12px 12px; display: flex; flex-direction: column; gap: 6px; }
.card .name { font-weight: 600; font-size: 13px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.card .cap-text { color: var(--dim); font-size: 12.5px; display: -webkit-box;
  -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.badges { display: flex; gap: 6px; flex-wrap: wrap; }
.badge { background: var(--badge); color: var(--dim); border-radius: 5px;
  font-size: 11px; padding: 2px 7px; }
#sentinel { height: 60px; }
#empty { color: var(--dim); text-align: center; padding: 60px 0; display: none; }
/* modal */
#overlay { position: fixed; inset: 0; background: rgba(0,0,0,.65); display: none;
  z-index: 20; align-items: center; justify-content: center; padding: 24px; }
#overlay.open { display: flex; }
#modal { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  width: min(1200px, 100%); max-height: 92vh; display: flex; overflow: hidden; }
#m-media { flex: 1.4; background: #000; display: flex; align-items: center;
  justify-content: center; min-width: 0; }
#m-media img, #m-media video { max-width: 100%; max-height: 92vh; object-fit: contain; }
#m-media .ph { font-size: 80px; opacity: .4; }
#m-side { flex: 1; min-width: 320px; max-width: 460px; overflow-y: auto; padding: 18px; }
#m-side h2 { font-size: 15px; word-break: break-all; margin-bottom: 4px; }
#m-side .path { color: var(--dim); font-size: 12px; word-break: break-all;
  cursor: pointer; margin-bottom: 10px; }
#m-side .path:hover { color: var(--accent); }
#m-side section { border-top: 1px solid var(--border); padding: 12px 0; }
#m-side section h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .05em;
  color: var(--accent); margin-bottom: 6px; }
#m-side p, #m-side pre { color: var(--text); font-size: 13px; white-space: pre-wrap;
  word-break: break-word; }
#m-side pre { max-height: 260px; overflow-y: auto; background: var(--card);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; }
#m-close { position: absolute; top: 14px; right: 18px; font-size: 26px; color: #fff;
  background: none; border: none; cursor: pointer; opacity: .7; }
#m-close:hover { opacity: 1; }
@media (max-width: 800px) { #modal { flex-direction: column; overflow-y: auto; }
  #m-side { max-width: none; } }
</style>
</head>
<body>
<header>
  <h1>file-index</h1>
  <input id="q" type="search" placeholder="Search captions, text, transcripts…">
  <div id="chips"></div>
  <label class="cap"><input id="captioned" type="checkbox"> captioned only</label>
  <span id="count"></span>
</header>
<div id="grid"></div>
<div id="empty">No files match.</div>
<div id="sentinel"></div>
<div id="overlay"><button id="m-close">&times;</button><div id="modal">
  <div id="m-media"></div><div id="m-side"></div>
</div></div>
<script>
"use strict";
const state = { q: "", kind: "", captioned: false, offset: 0, total: 0, busy: false, done: false };
const PAGE = 60;
const ICONS = { image: "🖼", video: "🎬", audio: "🎧", pdf: "📄", text: "📄",
  code: "⌨", office: "📄", other: "📦" };
const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

async function loadSummary() {
  const s = await (await fetch("api/summary")).json();
  const chips = $("chips");
  const mk = (label, kind, n) => {
    const c = el("button", "chip" + (state.kind === kind ? " on" : ""),
      n === undefined ? label : `${label} ${n.toLocaleString()}`);
    c.onclick = () => { state.kind = kind; refresh();
      [...chips.children].forEach(x => x.classList.remove("on")); c.classList.add("on"); };
    chips.appendChild(c);
  };
  mk("All", "", s.total);
  for (const k of Object.keys(s.kinds).sort((a, b) => s.kinds[b] - s.kinds[a]))
    mk(k, k, s.kinds[k]);
}

function card(f) {
  const c = el("div", "card");
  const t = el("div", "thumb");
  if (["image", "video", "pdf"].includes(f.kind)) {
    const img = el("img");
    img.loading = "lazy";
    img.src = "thumb/" + f.id;
    img.onerror = () => { t.textContent = ""; t.appendChild(el("span", "ph", ICONS[f.kind] || "📦")); };
    t.appendChild(img);
  } else {
    t.appendChild(el("span", "ph", ICONS[f.kind] || "📦"));
  }
  const body = el("div", "body");
  body.appendChild(el("div", "name", f.path.split("/").pop()));
  const badges = el("div", "badges");
  badges.appendChild(el("span", "badge", f.kind));
  if (f.vlm_type && f.vlm_type !== "other") badges.appendChild(el("span", "badge", f.vlm_type));
  if (f.degraded) badges.appendChild(el("span", "badge", "degraded"));
  body.appendChild(badges);
  body.appendChild(el("div", "cap-text",
    f.caption || (f.tier2_status ? "(no caption)" : "(not yet processed by deep)")));
  c.append(t, body);
  c.onclick = () => openModal(f.id);
  return c;
}

async function loadPage() {
  if (state.busy || state.done) return;
  state.busy = true;
  const p = new URLSearchParams({ q: state.q, kind: state.kind,
    captioned: state.captioned ? "1" : "", offset: state.offset, limit: PAGE });
  const r = await (await fetch("api/files?" + p)).json();
  state.total = r.total;
  for (const f of r.files) $("grid").appendChild(card(f));
  state.offset += r.files.length;
  state.done = state.offset >= r.total || r.files.length === 0;
  $("count").textContent = `${state.offset.toLocaleString()} of ${r.total.toLocaleString()}`;
  $("empty").style.display = r.total === 0 ? "block" : "none";
  state.busy = false;
}

function refresh() {
  state.offset = 0; state.done = false;
  $("grid").textContent = "";
  loadPage();
}

// modal ---------------------------------------------------------------
function section(title, contentEl) {
  const s = el("section");
  s.appendChild(el("h3", "", title));
  s.appendChild(contentEl);
  return s;
}

async function openModal(id) {
  const d = await (await fetch("api/file/" + id)).json();
  const media = $("m-media"), side = $("m-side");
  media.textContent = ""; side.textContent = "";
  if (d.kind === "image") {
    const img = el("img"); img.src = "media/" + d.id; media.appendChild(img);
  } else if (d.kind === "video") {
    const v = el("video"); v.controls = true; v.src = "media/" + d.id;
    media.appendChild(v);
  } else if (d.kind === "audio") {
    const a = el("audio"); a.controls = true; a.src = "media/" + d.id;
    media.appendChild(a);
  } else {
    media.appendChild(el("span", "ph", ICONS[d.kind] || "📦"));
  }
  side.appendChild(el("h2", "", d.path.split("/").pop()));
  const path = el("div", "path", d.path + "  (click to copy)");
  path.onclick = () => navigator.clipboard.writeText(d.path);
  side.appendChild(path);
  const info = el("div", "badges");
  info.appendChild(el("span", "badge", d.kind));
  info.appendChild(el("span", "badge", (d.size / 1048576).toFixed(2) + " MB"));
  info.appendChild(el("span", "badge", new Date(d.mtime * 1000).toLocaleString()));
  side.appendChild(info);
  if (d.error) side.appendChild(section("Processing error", el("p", "", d.error)));
  for (const c of d.content) side.appendChild(renderStage(c));
  $("overlay").classList.add("open");
}

function renderStage(c) {
  const m = typeof c.meta === "object" && c.meta !== null ? c.meta : null;
  if (c.stage === "vlm_image" && m) {
    const box = el("div");
    if (m.description) box.appendChild(el("p", "", m.description));
    if (m.ocr_text) {
      box.appendChild(el("h3", "", "Text in image"));
      box.appendChild(el("pre", "", m.ocr_text));
    }
    if (m.objects && m.objects.length) {
      const b = el("div", "badges");
      for (const o of m.objects) b.appendChild(el("span", "badge", o));
      box.appendChild(b);
    }
    if (m.inferred_context) box.appendChild(el("p", "", "Context: " + m.inferred_context));
    return section("Image analysis" + (c.degraded ? " (degraded)" : ""), box);
  }
  if (c.stage === "exif" && m) {
    const pre = el("pre", "", Object.entries(m)
      .map(([k, v]) => k + ": " + JSON.stringify(v)).join("\n"));
    return section("EXIF", pre);
  }
  const titles = { text: "Extracted text", pdf_text: "PDF text", pdf_scan_vlm: "Scanned PDF",
    whisper: "Transcript", audio_summary: "Audio summary", video_scenes: "Scenes",
    video_transcript: "Video transcript", video_summary: "Video summary" };
  const body = (c.body || "").slice(0, 20000);
  const node = ["video_summary", "audio_summary"].includes(c.stage)
    ? el("p", "", body) : el("pre", "", body);
  return section(titles[c.stage] || c.stage, node);
}

function closeModal() {
  $("overlay").classList.remove("open");
  $("m-media").textContent = "";  // stop any playing video
}
$("overlay").onclick = (e) => { if (e.target.id === "overlay") closeModal(); };
$("m-close").onclick = closeModal;
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

// wiring --------------------------------------------------------------
let debounce;
$("q").oninput = () => {
  clearTimeout(debounce);
  debounce = setTimeout(() => { state.q = $("q").value; refresh(); }, 300);
};
$("captioned").onchange = () => { state.captioned = $("captioned").checked; refresh(); };
new IntersectionObserver((es) => { if (es[0].isIntersecting) loadPage(); })
  .observe($("sentinel"));
loadSummary();
loadPage();
</script>
</body>
</html>
"""
