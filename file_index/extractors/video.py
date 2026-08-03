"""Video pipeline (tier 2, quality-first):

1. Content-aware scene detection (PySceneDetect), fixed-interval fallback.
2. 1-2 frames per scene via ffmpeg → VLM caption per frame, timestamps kept.
3. Audio track → Whisper timestamped transcript.
4. Agent-model summarization over scene captions + transcript.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

# OpenCV (PySceneDetect's decode backend) lets ffmpeg log straight to stderr,
# which floods the console on damaged streams. Must be set before cv2 loads.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # AV_LOG_QUIET

from ..ollama_client import OllamaClient
from . import audio as audio_ex

log = logging.getLogger("file_index.extractors.video")

VERSION = "video-1.0"


class VLMUnavailable(RuntimeError):
    """Every caption call for a video failed — the vision model is down, so
    the file must be retried rather than stored with empty captions."""

CAPTION_PROMPT = (
    "Describe this video frame in 1-3 sentences: what is happening, who/what is "
    "visible, any readable text. Be concrete and specific."
)

CAPTION_CONTEXT_SUFFIX = (
    "\nBackground: other videos from the same folder were about the following "
    "(this frame may or may not match — describe only what you actually see):\n{context}"
)

SUMMARY_PROMPT = """You are summarizing a video from its scene captions and audio transcript.
{context}
Scene captions (timestamped):
{captions}

Audio transcript (timestamped):
{transcript}

Write a coherent summary of the video: what it is, what happens over time, key
topics or events, and anything notable. If the background context above helps
identify the game, activity, or recurring people, use it. 1-3 paragraphs. /no_think"""

SUMMARY_CONTEXT_BLOCK = """
Background — summaries of other videos from the same folder (may describe the
same game, activity, or people):
{context}
"""


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


def sample_scenes(
    scenes: list[tuple[float, float]], max_scenes: int
) -> list[tuple[float, float]]:
    """Evenly sample at most max_scenes across the timeline (0 = keep all)."""
    if max_scenes <= 0 or len(scenes) <= max_scenes:
        return scenes
    step = len(scenes) / max_scenes
    return [scenes[int(i * step)] for i in range(max_scenes)]


def frame_dhash(path: Path) -> int | None:
    """64-bit difference hash of an image; None if it cannot be decoded."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            px = img.convert("L").resize((9, 8)).tobytes()
        bits = 0
        for row in range(8):
            for col in range(8):
                bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
        return bits
    except Exception:  # noqa: BLE001 — dedup is best-effort
        return None


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# Frames within this hamming distance of an already-captioned frame are
# considered near-duplicates and skipped (gameplay/screen recordings repeat
# almost-identical frames across scene cuts).
DHASH_NEAR_DUPLICATE = 6


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
    scenes: list[tuple[float, float]] | None = None,
    frame_workers: int = 4,
    context: str = "",
    summarize: bool = True,
    transcribe: bool = True,
    max_scenes: int = 0,
    dedup_frames: bool = False,
) -> dict:
    """Full pipeline. Returns:
    {scenes: [{start, end, captions: [str]}], transcript: {...}|None, summary: str}

    `scenes` can be supplied by a caller that already ran `detect_scenes` (the
    deep worker prefetches it for the next video while the GPU is busy).
    ffmpeg frame/audio extraction runs on `frame_workers` threads so the CPU
    stays ahead of the GPU instead of alternating with it per scene.
    `context` is optional background from already-indexed sibling videos
    (e.g. "these are Smite matches"), fed to the captioner and summarizer.
    With `summarize=False` the agent-model summary is skipped (summary="");
    the deep worker defers it to an end-of-run sweep via `summarize_video` so
    the vision model is not swapped out of VRAM per video. With
    `transcribe=False` Whisper is skipped too (transcript=None); the worker
    runs `transcribe_video` on a background thread so the GPU can move on.
    `max_scenes` caps VLM work on scene-heavy videos; `dedup_frames` skips
    frames that are near-duplicates of already-captioned ones.
    """
    from concurrent.futures import ThreadPoolExecutor

    if scenes is None:
        scenes = detect_scenes(path, fallback_interval_s)
    if not scenes:
        raise ValueError(f"no readable video stream in {path} — file is likely corrupt or truncated")
    total_scenes = len(scenes)
    scenes = sample_scenes(scenes, max_scenes)
    if len(scenes) < total_scenes:
        log.info("%s: %d scenes (sampled down from %d)", path.name, len(scenes), total_scenes)
    else:
        log.info("%s: %d scenes", path.name, len(scenes))

    scene_results = []
    skipped_dups = 0
    caption_attempts = caption_failures = 0
    seen_hashes: list[int] = []
    with tempfile.TemporaryDirectory(prefix="file-index-video-") as tmp:
        tmpdir = Path(tmp)
        with ThreadPoolExecutor(max_workers=max(1, frame_workers)) as pool:
            wav_future = (
                pool.submit(extract_audio_track, path, tmpdir / "audio.wav")
                if transcribe else None
            )
            frame_futures = []
            for i, (start, end) in enumerate(scenes):
                span = end - start
                n = max(1, min(frames_per_scene, 2))
                offsets = [start + span * 0.5] if n == 1 or span < 2 else [
                    start + span * 0.25, start + span * 0.75
                ]
                frame_futures.append([
                    pool.submit(extract_frame, path, ts, tmpdir / f"s{i}_f{j}.jpg")
                    for j, ts in enumerate(offsets)
                ])
            caption_prompt = CAPTION_PROMPT + (
                CAPTION_CONTEXT_SUFFIX.format(context=context) if context else ""
            )
            for i, (start, end) in enumerate(scenes):
                captions = []
                for fut in frame_futures[i]:
                    frame = fut.result()
                    if not frame:
                        continue
                    if dedup_frames:
                        h = frame_dhash(frame)
                        if h is not None:
                            if any(_hamming(h, s) <= DHASH_NEAR_DUPLICATE for s in seen_hashes):
                                skipped_dups += 1
                                continue
                            seen_hashes.append(h)
                    caption_attempts += 1
                    try:
                        cap = client.generate(vision_model, caption_prompt, images=[frame])
                        captions.append(cap.strip())
                    except Exception as e:  # noqa: BLE001 — keep other scenes going
                        caption_failures += 1
                        log.warning("caption failed scene %d of %s: %s", i, path, e)
                scene_results.append(
                    {"start": round(start, 2), "end": round(end, 2), "captions": captions}
                )
            wav = wav_future.result() if wav_future else None
        if skipped_dups:
            log.info("%s: skipped %d near-duplicate frames", path.name, skipped_dups)
        # Every VLM call failing means the model/server is broken, not the
        # video: raise so the queue retries this file later instead of
        # storing empty captions and marking it done forever.
        if caption_attempts and caption_failures == caption_attempts:
            raise VLMUnavailable(
                f"all {caption_attempts} caption calls failed for {path} — "
                "vision model unavailable?"
            )

        # Whisper on the audio track. Inline only when transcribe=True (the
        # deep worker instead runs transcribe_video on a background thread so
        # a slow — possibly CPU-fallback — transcription never idles the GPU).
        transcript = None
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
    summary = ""
    if summarize:
        try:
            summary = summarize_video(
                client, agent_model, captions_text, transcript_text, context
            )
        except Exception as e:  # noqa: BLE001
            log.warning("video summary failed for %s: %s", path, e)

    return {
        "scenes": scene_results,
        "transcript": transcript,
        "summary": summary,
        "captions_text": captions_text,
        "transcript_text": transcript_text,
    }


def transcribe_video(
    path: Path,
    whisper_model: str = "large-v3",
    whisper_device: str = "cuda",
    whisper_compute_type: str = "float16",
) -> dict | None:
    """Extract the audio track and Whisper-transcribe it. None when the video
    has no audio. Runs standalone (own temp dir) so the deep worker can call it
    on a background thread while the GPU captions the next video."""
    with tempfile.TemporaryDirectory(prefix="file-index-transcribe-") as tmp:
        wav = extract_audio_track(path, Path(tmp) / "audio.wav")
        if not wav:
            return None
        return audio_ex.transcribe(
            wav, whisper_model, whisper_device, whisper_compute_type
        )


def summarize_video(
    client: OllamaClient,
    agent_model: str,
    captions_text: str,
    transcript_text: str,
    context: str = "",
) -> str:
    """Agent-model summary over caption/transcript text. Raises on model
    failure so a deferred summary can be retried from the queue."""
    return client.generate(
        agent_model,
        SUMMARY_PROMPT.format(
            context=SUMMARY_CONTEXT_BLOCK.format(context=context) if context else "",
            captions=captions_text,
            transcript=transcript_text,
        ),
    ).strip()
