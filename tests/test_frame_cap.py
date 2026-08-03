"""Frames handed to the VLM must respect the same size cap as still images.

Regression for: video frames were extracted at native resolution and sent
straight to the VLM, so 4K sources failed every caption call with
`cudaMalloc failed: out of memory` and sub-4K sources still paid for
oversized inputs.
"""

import shutil
import subprocess

import pytest

from file_index.extractors import video as video_ex
from file_index.extractors.image import VLM_MAX_DIM

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def _make_clip(path, width, height, seconds=1):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=size={width}x{height}:rate=10:duration={seconds}",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True, timeout=120,
    )
    return path


def _dims(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True, timeout=60,
    ).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def test_4k_frame_is_downscaled_to_the_cap(tmp_path):
    clip = _make_clip(tmp_path / "uhd.mp4", 3840, 2160)
    frame = video_ex.extract_frame(clip, 0.0, tmp_path / "f.jpg")
    assert frame is not None
    w, h = _dims(frame)
    assert max(w, h) <= VLM_MAX_DIM, f"frame {w}x{h} exceeds the VLM cap"
    assert w / h == pytest.approx(3840 / 2160, rel=0.01)  # aspect preserved


def test_1080p_frame_is_downscaled_to_the_cap(tmp_path):
    clip = _make_clip(tmp_path / "fhd.mp4", 1920, 1080)
    frame = video_ex.extract_frame(clip, 0.0, tmp_path / "f.jpg")
    assert max(_dims(frame)) <= VLM_MAX_DIM


def test_small_frame_is_not_upscaled(tmp_path):
    """min(cap, iw) must never enlarge an already-small frame — upscaling would
    cost encoder time for no added detail."""
    clip = _make_clip(tmp_path / "small.mp4", 640, 360)
    frame = video_ex.extract_frame(clip, 0.0, tmp_path / "f.jpg")
    assert _dims(frame) == (640, 360)


def test_thumbnail_path_still_works(tmp_path):
    """web._thumb_video reuses extract_frame; capping must not break it."""
    from file_index.web import _thumb_video

    clip = _make_clip(tmp_path / "uhd.mp4", 3840, 2160)
    out = _thumb_video(clip, tmp_path / "t.jpg")
    assert out is not None and out.exists()
    assert max(_dims(out)) <= 512  # THUMB_SIZE
