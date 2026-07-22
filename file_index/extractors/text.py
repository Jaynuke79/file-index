"""Text/code/config/markdown extraction with encoding detection."""

from __future__ import annotations

from pathlib import Path

VERSION = "text-1.0"


def extract(path: Path, size_cap: int = 1_000_000) -> str:
    raw = path.read_bytes()[: size_cap * 2]  # read a bit extra; cap after decode
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        from charset_normalizer import from_bytes

        best = from_bytes(raw).best()
        if best is None:
            raise ValueError(f"could not detect encoding of {path}")
        text = str(best)
    return text[:size_cap]
