"""A video whose frames cannot be decoded must say so, not look fully processed.

Regression for: ffmpeg failing on every frame produced no VLM calls at all, so
the VLMUnavailable guard never fired; the file stored an empty video_scenes
body and was marked done, indistinguishable from a genuinely captioned video.
One real .avi logged 1508 frame-extraction failures this way.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from file_index.extractors import video as video_ex
from file_index.index import PENDING_DEEP
from file_index.queue import Tier2Worker

SCENES = [(float(i * 5), float(i * 5 + 5)) for i in range(30)]


@pytest.fixture
def vid(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    path = root / "broken.avi"
    path.write_bytes(b"not really a video")
    fid = index.upsert_file(str(path), "h1", 18, 1000.0, "video/x-msvideo", "video")
    index.enqueue(fid, 2, PENDING_DEEP, "video", 1000.0)
    index.commit()
    monkeypatch.setattr(video_ex, "detect_scenes", lambda p, i: list(SCENES))
    monkeypatch.setattr(video_ex, "extract_audio_track", lambda p, out: None)
    monkeypatch.setattr(video_ex, "frame_dhash", lambda p: None)
    return cfg, index, path, fid


def test_undecodable_video_is_flagged_and_gives_up_early(vid, monkeypatch):
    cfg, index, path, fid = vid
    calls = []
    monkeypatch.setattr(
        video_ex, "extract_frame",
        lambda p, ts, out: calls.append(ts) or None,  # every extraction fails
    )
    client = MagicMock()

    result = video_ex.process_video(
        path, client, vision_model="v", agent_model="a",
        frames_per_scene=1, frame_workers=4, summarize=False, transcribe=False,
    )

    assert result["frames_undecodable"] is True
    client.generate.assert_not_called()          # no VLM work attempted
    assert all(s["captions"] == [] for s in result["scenes"])
    # bailed out instead of one extraction per scene (30 scenes available)
    assert len(calls) <= 4, f"attempted {len(calls)} extractions"
    assert len(result["scenes"]) < len(SCENES)


def test_partial_decode_is_not_flagged_and_processes_everything(vid, monkeypatch):
    """One bad frame among good ones must not trip the give-up path."""
    cfg, index, path, fid = vid
    n = {"i": 0}

    def flaky(p, ts, out):
        n["i"] += 1
        if n["i"] == 1:
            return None            # first frame fails, the rest decode
        out.write_bytes(b"jpeg")
        return out

    monkeypatch.setattr(video_ex, "extract_frame", flaky)
    client = MagicMock()
    client.generate.return_value = "a caption"

    result = video_ex.process_video(
        path, client, vision_model="v", agent_model="a",
        frames_per_scene=1, frame_workers=4, summarize=False, transcribe=False,
    )

    assert result["frames_undecodable"] is False
    assert len(result["scenes"]) == len(SCENES)   # no early exit
    assert sum(len(s["captions"]) for s in result["scenes"]) == len(SCENES) - 1


def test_worker_marks_scenes_degraded(vid, monkeypatch):
    cfg, index, path, fid = vid
    monkeypatch.setattr(video_ex, "extract_frame", lambda p, ts, out: None)
    worker = Tier2Worker(cfg, index, client=MagicMock())
    worker.embedder = MagicMock()
    worker.embedder.embed_chunks.side_effect = lambda c: c

    item = index.next_pending(tier=2)
    worker._process(item, path, None)

    row = index.get_content(fid, "video_scenes")[0]
    assert row["degraded"] == 1, "empty captions must be recorded as degraded"


def test_audio_still_processed_when_frames_are_undecodable(vid, monkeypatch):
    """The audio stream is demuxed separately — a dead video stream must not
    cost the transcript."""
    cfg, index, path, fid = vid
    monkeypatch.setattr(video_ex, "extract_frame", lambda p, ts, out: None)
    monkeypatch.setattr(
        video_ex, "extract_audio_track",
        lambda p, out: (out.write_bytes(b"wav"), out)[1],
    )
    monkeypatch.setattr(
        video_ex.audio_ex, "transcribe",
        lambda *a, **k: {"language": "en", "duration": 3.0,
                         "segments": [{"start": 0.0, "end": 3.0, "text": "hello"}],
                         "text": "hello"},
    )

    result = video_ex.process_video(
        path, MagicMock(), vision_model="v", agent_model="a",
        frames_per_scene=1, summarize=False, transcribe=True,
    )

    assert result["frames_undecodable"] is True
    assert result["transcript"]["segments"][0]["text"] == "hello"
