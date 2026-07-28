"""Single-SQLite-database index: metadata, content, FTS5 keyword search,
sqlite-vec embeddings, the persistent work queue, and the audit log.

The queue lives in the same database so a checkpoint (file done + queue status)
is one atomic transaction — killing the process mid-run loses nothing.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path

# Queue statuses
PENDING_METADATA = "pending_metadata"
PENDING_DEEP = "pending_deep"
# Video whose captions+transcript are done but whose summary is deferred to the
# end-of-run sweep (all summaries run with the agent model loaded once, instead
# of a vision<->agent VRAM swap per video).
PENDING_SUMMARY = "pending_summary"
DONE = "done"
FAILED = "failed"

# Stages produced by tier-2 processing (safe to copy between identical files).
TIER2_STAGES = {
    "vlm_image", "pdf_scan_vlm", "whisper", "audio_summary",
    "video_scenes", "video_transcript", "video_summary",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    hash TEXT,
    size INTEGER,
    mtime REAL,
    mime TEXT,
    kind TEXT,               -- text|code|pdf|office|image|audio|video|other
    tier1_status TEXT,       -- done|failed|NULL
    tier2_status TEXT,       -- done|failed|skipped|NULL
    created_at REAL,
    updated_at REAL,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash);

CREATE TABLE IF NOT EXISTS content (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,           -- e.g. text, pdf_text, exif, vlm_image, whisper, video_scenes, video_summary
    extractor_version TEXT NOT NULL,
    body TEXT,                     -- raw extracted text / transcript / caption text
    meta TEXT,                     -- JSON: structured output (VLM JSON, EXIF dict, scene list...)
    degraded INTEGER DEFAULT 0,    -- 1 if output failed validation and is stored raw
    created_at REAL,
    UNIQUE(file_id, stage)
);

CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
    body, path UNINDEXED, stage UNINDEXED,
    tokenize='porter unicode61'
);
-- content.id is used as the fts rowid so hits map back to content rows.

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    content_id INTEGER NOT NULL REFERENCES content(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    source_stage TEXT NOT NULL,
    text TEXT NOT NULL,
    ts_start REAL,                 -- nullable, for audio/video chunks
    ts_end REAL,
    embedding BLOB                 -- float32 little-endian; also mirrored into vec table when available
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);

CREATE TABLE IF NOT EXISTS queue (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    tier INTEGER NOT NULL,         -- 1 or 2
    status TEXT NOT NULL,          -- pending_metadata|pending_deep|done|failed
    kind TEXT,                     -- routing hint for tier2 priority (image|audio|video|pdf_scan)
    mtime REAL,                    -- for newest-first ordering
    error TEXT,
    retries INTEGER DEFAULT 0,
    updated_at REAL,
    UNIQUE(file_id, tier)
);
CREATE INDEX IF NOT EXISTS idx_queue_status ON queue(status, tier);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY,
    ts REAL,
    op TEXT NOT NULL,
    before_path TEXT,
    after_path TEXT,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def _f32_blob(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _blob_f32(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


@dataclass
class SearchHit:
    file_id: int
    path: str
    stage: str
    snippet: str
    score: float
    ts_start: float | None = None
    ts_end: float | None = None
    source: str = "fts"  # fts | vec


class Index:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        self._vec_dim: int | None = None
        self._has_vec = self._try_load_sqlite_vec()

    def _try_load_sqlite_vec(self) -> bool:
        try:
            import sqlite_vec  # type: ignore

            self.db.enable_load_extension(True)
            sqlite_vec.load(self.db)
            self.db.enable_load_extension(False)
            return True
        except Exception:
            # Fall back to brute-force cosine search over the chunks.embedding blobs.
            return False

    def _ensure_vec_table(self, dim: int) -> None:
        if not self._has_vec or self._vec_dim == dim:
            return
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(embedding float[{dim}])"
        )
        self._vec_dim = dim

    # ---------- files ----------

    def get_file_by_path(self, path: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()

    def get_file_by_hash(self, hash_: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM files WHERE hash=? AND deleted=0", (hash_,)
        ).fetchone()

    def upsert_file(
        self, path: str, hash_: str | None, size: int, mtime: float, mime: str, kind: str
    ) -> int:
        now = time.time()
        row = self.get_file_by_path(path)
        if row:
            self.db.execute(
                "UPDATE files SET hash=?, size=?, mtime=?, mime=?, kind=?, updated_at=?, deleted=0 WHERE id=?",
                (hash_, size, mtime, mime, kind, now, row["id"]),
            )
            return row["id"]
        cur = self.db.execute(
            "INSERT INTO files(path, hash, size, mtime, mime, kind, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (path, hash_, size, mtime, mime, kind, now, now),
        )
        return cur.lastrowid

    def move_file(self, file_id: int, new_path: str, mtime: float) -> None:
        """Same content hash observed at a new path: update path, keep extractions."""
        self.db.execute(
            "UPDATE files SET path=?, mtime=?, updated_at=?, deleted=0 WHERE id=?",
            (new_path, mtime, time.time(), file_id),
        )
        # keep FTS paths in sync
        for row in self.db.execute(
            "SELECT id, stage FROM content WHERE file_id=?", (file_id,)
        ):
            self.db.execute(
                "UPDATE content_fts SET path=? WHERE rowid=?", (new_path, row["id"])
            )

    def mark_deleted(self, file_id: int) -> None:
        self.db.execute(
            "UPDATE files SET deleted=1, updated_at=? WHERE id=?", (time.time(), file_id)
        )

    def set_tier_status(self, file_id: int, tier: int, status: str) -> None:
        col = "tier1_status" if tier == 1 else "tier2_status"
        self.db.execute(
            f"UPDATE files SET {col}=?, updated_at=? WHERE id=?",
            (status, time.time(), file_id),
        )

    # ---------- content ----------

    def store_content(
        self,
        file_id: int,
        stage: str,
        extractor_version: str,
        body: str | None,
        meta: dict | None = None,
        degraded: bool = False,
    ) -> int:
        """Insert or replace one extraction stage's output for a file.

        Versioned by (stage, extractor_version): re-running a stage with a newer
        extractor replaces only that stage's rows, FTS entries, and chunks.
        """
        path_row = self.db.execute(
            "SELECT path FROM files WHERE id=?", (file_id,)
        ).fetchone()
        path = path_row["path"] if path_row else ""
        old = self.db.execute(
            "SELECT id FROM content WHERE file_id=? AND stage=?", (file_id, stage)
        ).fetchone()
        if old:
            self._delete_content_row(old["id"])
        cur = self.db.execute(
            "INSERT INTO content(file_id, stage, extractor_version, body, meta, degraded, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                file_id,
                stage,
                extractor_version,
                body,
                json.dumps(meta) if meta is not None else None,
                1 if degraded else 0,
                time.time(),
            ),
        )
        content_id = cur.lastrowid
        if body:
            self.db.execute(
                "INSERT INTO content_fts(rowid, body, path, stage) VALUES(?,?,?,?)",
                (content_id, body, path, stage),
            )
        return content_id

    def _delete_content_row(self, content_id: int) -> None:
        self.db.execute("DELETE FROM content_fts WHERE rowid=?", (content_id,))
        if self._has_vec and self._vec_dim:
            for c in self.db.execute(
                "SELECT id FROM chunks WHERE content_id=?", (content_id,)
            ):
                self.db.execute("DELETE FROM chunk_vec WHERE rowid=?", (c["id"],))
        self.db.execute("DELETE FROM chunks WHERE content_id=?", (content_id,))
        self.db.execute("DELETE FROM content WHERE id=?", (content_id,))

    def clone_tier2_content(self, src_file_id: int, dst_file_id: int) -> int:
        """Copy tier-2 extraction results to a byte-identical file (same hash),
        including chunks and embeddings. Returns the number of stages copied.
        """
        n = 0
        for row in self.db.execute(
            "SELECT * FROM content WHERE file_id=?", (src_file_id,)
        ).fetchall():
            if row["stage"] not in TIER2_STAGES:
                continue
            meta = json.loads(row["meta"]) if row["meta"] else None
            cid = self.store_content(
                dst_file_id, row["stage"], row["extractor_version"], row["body"],
                meta=meta, degraded=bool(row["degraded"]),
            )
            chunks = [
                {
                    "text": ch["text"],
                    "ts_start": ch["ts_start"],
                    "ts_end": ch["ts_end"],
                    "embedding": _blob_f32(ch["embedding"]) if ch["embedding"] else None,
                }
                for ch in self.db.execute(
                    "SELECT * FROM chunks WHERE content_id=? ORDER BY chunk_index",
                    (row["id"],),
                )
            ]
            self.store_chunks(dst_file_id, cid, row["stage"], chunks)
            n += 1
        return n

    def get_content(self, file_id: int, stage: str | None = None) -> list[sqlite3.Row]:
        if stage:
            return self.db.execute(
                "SELECT * FROM content WHERE file_id=? AND stage=?", (file_id, stage)
            ).fetchall()
        return self.db.execute(
            "SELECT * FROM content WHERE file_id=?", (file_id,)
        ).fetchall()

    # ---------- chunks / vectors ----------

    def store_chunks(
        self,
        file_id: int,
        content_id: int,
        source_stage: str,
        chunks: list[dict],
    ) -> None:
        """chunks: [{text, embedding: list[float]|None, ts_start, ts_end}, ...]"""
        for i, ch in enumerate(chunks):
            emb = ch.get("embedding")
            blob = _f32_blob(emb) if emb else None
            cur = self.db.execute(
                "INSERT INTO chunks(file_id, content_id, chunk_index, source_stage, text, ts_start, ts_end, embedding) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    file_id,
                    content_id,
                    i,
                    source_stage,
                    ch["text"],
                    ch.get("ts_start"),
                    ch.get("ts_end"),
                    blob,
                ),
            )
            if emb and self._has_vec:
                self._ensure_vec_table(len(emb))
                self.db.execute(
                    "INSERT INTO chunk_vec(rowid, embedding) VALUES(?,?)",
                    (cur.lastrowid, _f32_blob(emb)),
                )

    # ---------- queue ----------

    def enqueue(self, file_id: int, tier: int, status: str, kind: str, mtime: float) -> None:
        self.db.execute(
            "INSERT INTO queue(file_id, tier, status, kind, mtime, updated_at) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(file_id, tier) DO UPDATE SET status=excluded.status, kind=excluded.kind, "
            "mtime=excluded.mtime, error=NULL, retries=0, updated_at=excluded.updated_at",
            (file_id, tier, status, kind, mtime, time.time()),
        )

    def next_pending(
        self,
        tier: int,
        kind_priority: list[str] | None = None,
        newest_first: bool = True,
        status: str | None = None,
    ) -> sqlite3.Row | None:
        rows = self.peek_pending(tier, kind_priority, newest_first, limit=1, status=status)
        return rows[0] if rows else None

    def peek_pending(
        self,
        tier: int,
        kind_priority: list[str] | None = None,
        newest_first: bool = True,
        limit: int = 1,
        status: str | None = None,
    ) -> list[sqlite3.Row]:
        """The next `limit` pending items in processing order, without claiming
        them. Row 0 is what `next_pending` would return; the rest let the deep
        worker prefetch CPU work for upcoming files."""
        if status is None:
            status = PENDING_METADATA if tier == 1 else PENDING_DEEP
        order = []
        if kind_priority:
            cases = " ".join(
                f"WHEN '{k}' THEN {i}" for i, k in enumerate(kind_priority)
            )
            order.append(f"CASE q.kind {cases} ELSE 99 END")
        order.append("q.mtime " + ("DESC" if newest_first else "ASC"))
        return self.db.execute(
            f"SELECT q.*, f.path, f.kind AS file_kind, f.mime FROM queue q "
            f"JOIN files f ON f.id=q.file_id "
            f"WHERE q.status=? AND q.tier=? AND f.deleted=0 "
            f"ORDER BY {', '.join(order)} LIMIT ?",
            (status, tier, limit),
        ).fetchall()

    def mark_done(self, queue_id: int) -> None:
        self.db.execute(
            "UPDATE queue SET status=?, error=NULL, updated_at=? WHERE id=?",
            (DONE, time.time(), queue_id),
        )

    def set_queue_status(self, queue_id: int, status: str) -> None:
        self.db.execute(
            "UPDATE queue SET status=?, updated_at=? WHERE id=?",
            (status, time.time(), queue_id),
        )

    def mark_failed(
        self, queue_id: int, error: str, max_retries: int = 3,
        retry_status: str | None = None,
    ) -> None:
        """`retry_status` overrides the status a retryable failure returns to
        (e.g. a failed deferred summary goes back to pending_summary, not a
        full re-run of the captions)."""
        row = self.db.execute("SELECT retries, tier FROM queue WHERE id=?", (queue_id,)).fetchone()
        retries = (row["retries"] if row else 0) + 1
        pending = retry_status or (
            PENDING_METADATA if (row and row["tier"] == 1) else PENDING_DEEP
        )
        status = FAILED if retries >= max_retries else pending
        self.db.execute(
            "UPDATE queue SET status=?, error=?, retries=?, updated_at=? WHERE id=?",
            (status, error[:2000], retries, time.time(), queue_id),
        )

    def queue_stats(self) -> dict:
        stats: dict = {"tier1": {}, "tier2": {}}
        for row in self.db.execute(
            "SELECT tier, status, COUNT(*) n FROM queue GROUP BY tier, status"
        ):
            stats[f"tier{row['tier']}"][row["status"]] = row["n"]
        return stats

    def failures(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT q.tier, q.error, q.retries, f.path FROM queue q JOIN files f ON f.id=q.file_id "
            "WHERE q.status=? ORDER BY q.updated_at DESC LIMIT ?",
            (FAILED, limit),
        ).fetchall()

    # ---------- search ----------

    def fts_search(self, query: str, limit: int = 20) -> list[SearchHit]:
        hits = []
        try:
            rows = self.db.execute(
                "SELECT c.file_id, c.stage, f.path, "
                "snippet(content_fts, 0, '[', ']', '…', 12) AS snip, rank "
                "FROM content_fts JOIN content c ON c.id = content_fts.rowid "
                "JOIN files f ON f.id = c.file_id "
                "WHERE content_fts MATCH ? AND f.deleted=0 ORDER BY rank LIMIT ?",
                (_fts_escape(query), limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        for r in rows:
            hits.append(
                SearchHit(
                    file_id=r["file_id"],
                    path=r["path"],
                    stage=r["stage"],
                    snippet=r["snip"],
                    score=-float(r["rank"]),  # fts5 rank is negative-better
                    source="fts",
                )
            )
        return hits

    def vector_search(self, embedding: list[float], limit: int = 20) -> list[SearchHit]:
        hits: list[SearchHit] = []
        if self._has_vec:
            self._ensure_vec_table(len(embedding))
            try:
                rows = self.db.execute(
                    "SELECT v.rowid, v.distance FROM chunk_vec v "
                    "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
                    (_f32_blob(embedding), limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            for r in rows:
                ch = self.db.execute(
                    "SELECT c.*, f.path FROM chunks c JOIN files f ON f.id=c.file_id "
                    "WHERE c.id=? AND f.deleted=0",
                    (r["rowid"],),
                ).fetchone()
                if ch:
                    hits.append(
                        SearchHit(
                            file_id=ch["file_id"],
                            path=ch["path"],
                            stage=ch["source_stage"],
                            snippet=ch["text"][:300],
                            score=1.0 / (1.0 + float(r["distance"])),
                            ts_start=ch["ts_start"],
                            ts_end=ch["ts_end"],
                            source="vec",
                        )
                    )
            return hits
        # Brute-force fallback (no sqlite-vec extension available)
        import numpy as np

        q = np.array(embedding, dtype=np.float32)
        qn = q / (np.linalg.norm(q) + 1e-9)
        scored = []
        for ch in self.db.execute(
            "SELECT c.id, c.embedding FROM chunks c JOIN files f ON f.id=c.file_id "
            "WHERE c.embedding IS NOT NULL AND f.deleted=0"
        ):
            v = np.frombuffer(ch["embedding"], dtype=np.float32)
            sim = float(np.dot(qn, v / (np.linalg.norm(v) + 1e-9)))
            scored.append((sim, ch["id"]))
        scored.sort(reverse=True)
        for sim, cid in scored[:limit]:
            ch = self.db.execute(
                "SELECT c.*, f.path FROM chunks c JOIN files f ON f.id=c.file_id WHERE c.id=?",
                (cid,),
            ).fetchone()
            hits.append(
                SearchHit(
                    file_id=ch["file_id"],
                    path=ch["path"],
                    stage=ch["source_stage"],
                    snippet=ch["text"][:300],
                    score=sim,
                    ts_start=ch["ts_start"],
                    ts_end=ch["ts_end"],
                    source="vec",
                )
            )
        return hits

    # ---------- audit ----------

    def audit(self, op: str, before: str | None, after: str | None, detail: str = "") -> None:
        self.db.execute(
            "INSERT INTO audit_log(ts, op, before_path, after_path, detail) VALUES(?,?,?,?,?)",
            (time.time(), op, before, after, detail),
        )
        self.db.commit()

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()


def _fts_escape(query: str) -> str:
    """Quote each term so user queries can't break FTS5 syntax."""
    terms = [t.replace('"', '""') for t in query.split()]
    return " ".join(f'"{t}"' for t in terms if t)


def merge_hits(
    fts: list[SearchHit], vec: list[SearchHit], limit: int = 20, k: int = 60
) -> list[SearchHit]:
    """Reciprocal-rank-fusion merge of keyword and vector result lists."""
    scores: dict[tuple[int, str, float | None], float] = {}
    best: dict[tuple[int, str, float | None], SearchHit] = {}
    for hits in (fts, vec):
        for rank, h in enumerate(hits):
            key = (h.file_id, h.stage, h.ts_start)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in best or (h.ts_start is not None and best[key].ts_start is None):
                best[key] = h
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for key, s in ranked[:limit]:
        h = best[key]
        h.score = s
        out.append(h)
    return out
