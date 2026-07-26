"""Filesystem crawler: walks whitelisted roots, hashes files, detects
new/modified/moved/unchanged files, and populates the work queue.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .index import Index, PENDING_DEEP, PENDING_METADATA

log = logging.getLogger("file_index.crawler")

try:
    import blake3  # type: ignore

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# extensions → kind routing
TEXT_EXTS = {
    ".txt", ".md", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".xml", ".html", ".htm", ".css",
}
CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp",
    ".rs", ".go", ".rb", ".sh", ".bash", ".zsh", ".pl", ".lua", ".sql", ".php",
    ".kt", ".swift", ".scala", ".r", ".jl", ".m", ".ps1", ".bat", ".gradle",
    ".dockerfile", ".mk", ".cmake", ".vue", ".svelte",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".heic"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
OFFICE_EXTS = {".docx", ".xlsx", ".pptx"}


def classify(path: Path, mime: str) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in OFFICE_EXTS:
        return "office"
    if ext in IMAGE_EXTS or mime.startswith("image/"):
        return "image"
    if ext in AUDIO_EXTS or mime.startswith("audio/"):
        return "audio"
    if ext in VIDEO_EXTS or mime.startswith("video/"):
        return "video"
    if ext in CODE_EXTS:
        return "code"
    if ext in TEXT_EXTS or mime.startswith("text/"):
        return "text"
    return "other"


# kinds that get tier-2 deep processing
DEEP_KINDS = {"image", "audio", "video"}


def hash_file(path: Path, algo: str = "blake3", chunk_size: int = 1 << 20) -> str:
    if algo == "blake3" and HAS_BLAKE3:
        h = blake3.blake3()
    else:
        h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def exclude_pattern(raw: str) -> str:
    """Normalize an exclude argument to an fnmatch glob. Existing paths are
    matched literally ('[' would otherwise start a character class — think
    torrent dirs like '...[TGx]'); directories cover everything beneath them.
    """
    p = Path(raw).expanduser()
    if p.is_dir():
        return str(p.resolve()).replace("[", "[[]") + "/*"
    if p.exists():
        return str(p.resolve()).replace("[", "[[]")
    return raw


def apply_exclude(config: Config, index: Index, pattern: str) -> tuple[str, int, int]:
    """Add an exclude pattern and soft-remove matching files from the index.

    Index-only: never touches files on disk. A directory argument excludes
    everything under it. Returns (normalized_pattern, files_removed,
    pending_deep_skipped). Reversible: drop the pattern from config.yaml and
    re-run `scan` — extractions on soft-deleted rows are retained.
    """
    pattern = exclude_pattern(pattern)
    removed = pending = 0
    for row in index.db.execute(
        "SELECT f.id, f.path, q.status FROM files f "
        "LEFT JOIN queue q ON q.file_id=f.id AND q.tier=2 WHERE f.deleted=0"
    ).fetchall():
        if fnmatch.fnmatch(row["path"], pattern):
            index.mark_deleted(row["id"])
            removed += 1
            if row["status"] == PENDING_DEEP:
                pending += 1
    if pattern not in config.excludes:
        config.excludes.append(pattern)
        config.save()
    index.audit("exclude", None, None,
                f"pattern={pattern} removed={removed} pending_skipped={pending}")
    index.commit()
    return pattern, removed, pending


@dataclass
class CrawlStats:
    scanned: int = 0
    new: int = 0
    modified: int = 0
    moved: int = 0
    unchanged: int = 0
    removed: int = 0
    errors: int = 0
    error_paths: list[str] = field(default_factory=list)


class Crawler:
    def __init__(self, config: Config, index: Index):
        self.config = config
        self.index = index

    def _excluded(self, path: Path) -> bool:
        s = str(path)
        return any(fnmatch.fnmatch(s, pat) for pat in self.config.excludes)

    def walk(self) -> list[Path]:
        """Yield candidate files under whitelisted roots, honoring excludes."""
        out: list[Path] = []
        for root in self.config.roots:
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                d = Path(dirpath)
                dirnames[:] = [
                    n for n in dirnames if not self._excluded(d / n / "_")
                    and not self._excluded(d / n)
                ]
                for name in filenames:
                    p = d / name
                    if self._excluded(p):
                        continue
                    try:
                        if p.is_symlink() or not p.is_file():
                            continue
                    except OSError:
                        continue
                    out.append(p)
        return out

    def crawl(self, progress_cb=None) -> CrawlStats:
        stats = CrawlStats()
        seen_paths: set[str] = set()
        files = self.walk()
        for i, p in enumerate(files):
            if progress_cb:
                progress_cb(i, len(files), str(p))
            try:
                self._process_one(p, stats, seen_paths)
            except (OSError, PermissionError) as e:
                stats.errors += 1
                stats.error_paths.append(str(p))
                log.warning("crawl error on %s: %s", p, e)
            # checkpoint frequently so a kill loses at most one file
            if i % 100 == 0:
                self.index.commit()
        self._mark_missing(seen_paths, stats)
        self.index.commit()
        return stats

    def _process_one(self, p: Path, stats: CrawlStats, seen_paths: set[str]) -> None:
        stats.scanned += 1
        st = p.stat()
        path_s = str(p.resolve())
        seen_paths.add(path_s)

        existing = self.index.get_file_by_path(path_s)
        if (
            existing
            and not existing["deleted"]
            and existing["size"] == st.st_size
            and abs(existing["mtime"] - st.st_mtime) < 1e-6
        ):
            # cheap skip: size+mtime unchanged → assume same content, no re-hash
            stats.unchanged += 1
            return

        if st.st_size > self.config.limits.max_file_size:
            log.info("skipping oversized file %s (%d bytes)", p, st.st_size)
            return

        hash_ = hash_file(p, self.config.limits.hash_algo)
        mime = mimetypes.guess_type(path_s)[0] or "application/octet-stream"
        kind = classify(p, mime)

        if existing and existing["hash"] == hash_:
            # same path, same content — just refresh stat metadata
            self.index.upsert_file(path_s, hash_, st.st_size, st.st_mtime, mime, kind)
            stats.unchanged += 1
            return

        if not existing:
            by_hash = self.index.get_file_by_hash(hash_)
            if by_hash and not Path(by_hash["path"]).exists():
                # moved: same content, old path gone → update path, keep extractions
                self.index.move_file(by_hash["id"], path_s, st.st_mtime)
                stats.moved += 1
                return

        file_id = self.index.upsert_file(path_s, hash_, st.st_size, st.st_mtime, mime, kind)
        self.index.enqueue(file_id, 1, PENDING_METADATA, kind, st.st_mtime)
        if kind in DEEP_KINDS:
            self.index.enqueue(file_id, 2, PENDING_DEEP, kind, st.st_mtime)
        if existing:
            stats.modified += 1
        else:
            stats.new += 1

    def _mark_missing(self, seen_paths: set[str], stats: CrawlStats) -> None:
        """Mark files that vanished from disk as deleted (soft — index only)."""
        for row in self.index.db.execute("SELECT id, path FROM files WHERE deleted=0"):
            if row["path"] not in seen_paths:
                in_roots = any(
                    row["path"].startswith(str(r.resolve()) + os.sep)
                    for r in self.config.roots
                )
                if in_roots and not Path(row["path"]).exists():
                    self.index.mark_deleted(row["id"])
                    stats.removed += 1
