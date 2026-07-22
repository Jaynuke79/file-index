"""Audio transcription with faster-whisper (CUDA), tier 2.

The Whisper model is loaded lazily and cached at module level so repeated
files in one `deep` run reuse it. It lives on the GPU alongside nothing else —
Ollama models are loaded/unloaded by Ollama itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("file_index.extractors.audio")

VERSION = "whisper-large-v3-1.0"

_model = None
_model_key: tuple | None = None


def _get_model(name: str, device: str, compute_type: str):
    global _model, _model_key
    key = (name, device, compute_type)
    if _model is None or _model_key != key:
        from faster_whisper import WhisperModel

        log.info("loading faster-whisper %s on %s (%s)", name, device, compute_type)
        try:
            _model = WhisperModel(name, device=device, compute_type=compute_type)
        except Exception as e:  # noqa: BLE001 — fall back to CPU if CUDA is broken
            if device != "cpu":
                log.warning("whisper on %s failed (%s); falling back to cpu int8", device, e)
                _model = WhisperModel(name, device="cpu", compute_type="int8")
            else:
                raise
        _model_key = key
    return _model


def transcribe(
    path: Path,
    model_name: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
) -> dict:
    """Returns {language, duration, segments: [{start, end, text}], text}."""
    model = _get_model(model_name, device, compute_type)
    segments_iter, info = model.transcribe(str(path), vad_filter=True)
    segments = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments_iter
    ]
    return {
        "language": info.language,
        "duration": round(info.duration, 2),
        "segments": segments,
        "text": " ".join(s["text"] for s in segments),
    }


def format_transcript(segments: list[dict]) -> str:
    """Human-readable timestamped transcript for storage/FTS."""
    lines = []
    for s in segments:
        lines.append(f"[{_ts(s['start'])} - {_ts(s['end'])}] {s['text']}")
    return "\n".join(lines)


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
