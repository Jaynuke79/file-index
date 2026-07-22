"""Hybrid search: FTS5 keyword + vector similarity, merged with RRF."""

from __future__ import annotations

from .config import Config
from .embed import Embedder
from .index import Index, SearchHit, merge_hits


def hybrid_search(
    config: Config, index: Index, query: str, limit: int = 20
) -> list[SearchHit]:
    fts_hits = index.fts_search(query, limit=limit * 2)
    vec_hits: list[SearchHit] = []
    embedder = Embedder(config)
    if embedder.client.ping():
        emb = embedder.embed_query(query)
        if emb:
            vec_hits = index.vector_search(emb, limit=limit * 2)
    return merge_hits(fts_hits, vec_hits, limit=limit)


def format_ts(seconds: float | None) -> str:
    if seconds is None:
        return ""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
