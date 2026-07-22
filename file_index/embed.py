"""Text chunking and embedding.

Token counts are approximated as words * 4/3 (≈0.75 words per token), which is
close enough for sizing ~500-token chunks without pulling in a tokenizer.
"""

from __future__ import annotations

import logging

from .config import Config
from .ollama_client import OllamaClient, OllamaError

log = logging.getLogger("file_index.embed")


def chunk_text(
    text: str,
    chunk_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[dict]:
    """Split text into ~chunk_tokens word-boundary chunks with overlap.

    Returns [{text, ts_start: None, ts_end: None}, ...].
    """
    words = text.split()
    if not words:
        return []
    words_per_chunk = max(1, int(chunk_tokens * 0.75))
    overlap_words = min(int(overlap_tokens * 0.75), words_per_chunk - 1)
    step = words_per_chunk - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start : start + words_per_chunk]
        chunks.append({"text": " ".join(piece), "ts_start": None, "ts_end": None})
        if start + words_per_chunk >= len(words):
            break
    return chunks


def chunk_segments(
    segments: list[dict],
    chunk_tokens: int = 500,
) -> list[dict]:
    """Group timestamped segments ({text, start, end}) into ~chunk_tokens chunks,
    preserving the covered timestamp range on each chunk."""
    words_per_chunk = max(1, int(chunk_tokens * 0.75))
    chunks: list[dict] = []
    cur_texts: list[str] = []
    cur_words = 0
    cur_start: float | None = None
    cur_end: float | None = None
    for seg in segments:
        t = seg["text"].strip()
        if not t:
            continue
        n = len(t.split())
        if cur_words and cur_words + n > words_per_chunk:
            chunks.append(
                {"text": " ".join(cur_texts), "ts_start": cur_start, "ts_end": cur_end}
            )
            cur_texts, cur_words, cur_start = [], 0, None
        cur_texts.append(t)
        cur_words += n
        if cur_start is None:
            cur_start = seg.get("start")
        cur_end = seg.get("end")
    if cur_texts:
        chunks.append(
            {"text": " ".join(cur_texts), "ts_start": cur_start, "ts_end": cur_end}
        )
    return chunks


class Embedder:
    def __init__(self, config: Config, client: OllamaClient | None = None):
        self.config = config
        self.client = client or OllamaClient(config.models.ollama_url)
        self.model = config.models.embed

    def embed_chunks(self, chunks: list[dict], batch_size: int = 16) -> list[dict]:
        """Fill each chunk's `embedding`. On failure, leaves embedding=None
        (keyword search still works; vectors can be backfilled later)."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            try:
                vecs = self.client.embed(self.model, [c["text"] for c in batch])
                for c, v in zip(batch, vecs):
                    c["embedding"] = v
            except (OllamaError, Exception) as e:  # noqa: BLE001 — keep indexing alive
                log.warning("embedding batch failed: %s", e)
                for c in batch:
                    c.setdefault("embedding", None)
        return chunks

    def embed_query(self, query: str) -> list[float] | None:
        try:
            return self.client.embed(self.model, [query])[0]
        except Exception as e:  # noqa: BLE001
            log.warning("query embedding failed: %s", e)
            return None
