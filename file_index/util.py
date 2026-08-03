"""Small helpers shared across modules.

These previously existed as near-identical copies in index/web (FTS escaping),
agent/queue (think-block stripping), and audio/search (timestamp formatting),
and had begun to drift.
"""

from __future__ import annotations

import re


def fts_escape(query: str) -> str:
    """Quote each term so user input is never parsed as FTS5 syntax."""
    terms = [t for t in re.split(r"\s+", query.strip()) if t]
    return " ".join('"' + t.replace('"', '""') + '"' for t in terms)


def strip_think(text: str) -> str:
    """Remove qwen3 <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def format_ts(seconds: float | None) -> str:
    """Seconds as [HH:]MM:SS. Empty string for None."""
    if seconds is None:
        return ""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
