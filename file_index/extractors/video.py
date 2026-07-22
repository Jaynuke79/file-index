"""Video pipeline (tier 2, quality-first):

1. Content-aware scene detection (PySceneDetect), fixed-interval fallback.
2. 1-2 frames per scene via ffmpeg → VLM caption per frame, timestamps kept.
3. Audio track → Whisper timestamped transcript.
4. Agent-model summarization over scene captions + transcript.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from ..ollama_client import OllamaClient
from . import audio as audio_ex

log = logging.getLogger("file_index.extractors.video")

VERSION = "video-1.0"

CAPTION_PROMPT = (
    "Describe this video frame in 1-3 sentences: what is happening, who/what is "
    "visible, any readable text. Be concrete and specific."
)

SUMMARY_PROMPT = """You are summarizing a video from its scene captions and audio transcript.

Scene captions (timestamped):
{captions}

Audio transcript (timestamped):
{transcript}

Write a coherent summary of the video: what it is, what happens over time, key
topics or events, and anything notable. 1-3 paragraphs. /no_think"""


def detect_scenes(path: Path, fallback_interval_s: float = 10.0) -> list[tuple[float, float]]:
    """Returns [(start_s, end_s)] per scene. Falls back to fixed intervals."""
    try:
        from scenedetect import ContentDetector, detect

        scenes = detect(str(path), ContentDetector())
        if scenes:
            return [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
    except Exception as e:  # noqa: BLE001 — fallback path below
        log.warning("scene detection failed for %s: %s", path, e)
    dur = probe_duration(path)
    if dur <= 0:
        return []
    out = []
    t = 0.0
    while t < dur:
        out.append((t, min(t + fallback_interval_s, dur)))
        t += fallback_interval_s
    return out


def probe_duration(path: Path) -> float:
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        return float(r.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0


def extract_frame(path: Path, timestamp: float, out_path: Path) -> Path | None:
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-ss", f"{timestamp:.2f}",
                "-i", str(path), "-frames:v", "1", "-q:v", "3", str(out_path),
            ],
            capture_output=True, timeout=120, check=True,
        )
        return out_path if out_path.exists() else None
    except (subprocess.SubprocessError, OSError) as e:
        log.warning("frame extraction failed at %.1fs of %s: %s", timestamp, path, e)
        return None


def extract_audio_track(path: Path, out_path: Path) -> Path | None:
    """Extract mono 16 kHz wav for Whisper. None if the video has no audio."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", str(path),
                "-vn", "-ac", "1", "-ar", "16000", str(out_path),
            ],
            capture_output=True, timeout=600, check=True,
        )
        return out_path if out_path.exists() and out_path.stat().st_size > 44 else None
    except (subprocess.SubprocessError, OSError):
        return None


def process_video(
    path: Path,
    client: OllamaClient,
    vision_model: str,
    agent_model: str,
    frames_per_scene: int = 2,
    fallback_interval_s: float = 10.0,
    whisper_model: str = "large-v3",
    whisper_device: str = "cuda",
    whisper_compute_type: str = "float16",
) -> dict:
    """Full pipeline. Returns:
    {scenes: [{start, end, captions: [str]}], transcript: {...}|None, summary: str}
    """
    scenes = detect_scenes(path, fallback_interval_s)
    if not scenes:
        raise ValueError(f"could not determine scenes/duration for {path}")
    log.info("%s: %d scenes", path.name, len(scenes))

    scene_results = []
    with tempfile.TemporaryDirectory(prefix="file-index-video-") as tmp:
        tmpdir = Path(tmp)
        for i, (start, end) in enumerate(scenes):
            span = end - start
            n = max(1, min(frames_per_scene, 2))
            offsets = [start + span * 0.5] if n == 1 or span < 2 else [
                start + span * 0.25, start + span * 0.75
            ]
            captions = []
            for j, ts in enumerate(offsets):
                frame = extract_frame(path, ts, tmpdir / f"s{i}_f{j}.jpg")
                if not frame:
                    continue
                try:
                    cap = client.generate(vision_model, CAPTION_PROMPT, images=[frame])
                    captions.append(cap.strip())
                except Exception as e:  # noqa: BLE001 — keep other scenes going
                    log.warning("caption failed scene %d of %s: %s", i, path, e)
            scene_results.append(
                {"start": round(start, 2), "end": round(end, 2), "captions": captions}
            )

        # Whisper on the audio track (after VLM so Ollama can swap models freely;
        # Whisper is a separate CUDA process anyway).
        transcript = None
        wav = extract_audio_track(path, tmpdir / "audio.wav")
        if wav:
            try:
                transcript = audio_ex.transcribe(
                    wav, whisper_model, whisper_device, whisper_compute_type
                )
            except Exception as e:  # noqa: BLE001
                log.warning("whisper failed for %s: %s", path, e)

    captions_text = "\n".join(
        f"[{audio_ex._ts(s['start'])} - {audio_ex._ts(s['end'])}] " + " | ".join(s["captions"])
        for s in scene_results
        if s["captions"]
    ) or "(no captions available)"
    transcript_text = (
        audio_ex.format_transcript(transcript["segments"])
        if transcript and transcript["segments"]
        else "(no speech / no audio track)"
    )
    try:
        summary = client.generate(
            agent_model,
            SUMMARY_PROMPT.format(captions=captions_text, transcript=transcript_text),
        ).strip()
    except Exception as e:  # noqa: BLE001
        log.warning("video summary failed for %s: %s", path, e)
        summary = ""

    return {"scenes": scene_results, "transcript": transcript, "summary": summary}
