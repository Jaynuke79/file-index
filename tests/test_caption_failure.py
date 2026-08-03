"""A dead vision model must not silently produce empty, permanently-done videos.

Regression for: every frame caption call failing (Ollama crashed / model OOM)
returned scenes with empty caption lists, the worker stored an empty
video_scenes body and advanced the item to done, and nothing ever retried it.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from file_index.extractors import video as video_ex
from file_index.index import PENDING_DEEP, PENDING_SUMMARY
from file_index.queue import Tier2Worker


@pytest.fixture
def video_item(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    path = root / "clip.mp4"
    path.write_bytes(b"fake video")
    fid = index.upsert_file(str(path), "h1", 10, 1000.0, "video/mp4", "video")
    index.enqueue(fid, 2, PENDING_DEEP, "video", 1000.0)
    index.commit()

    monkeypatch.setattr(video_ex, "detect_scenes", lambda p, i: [(0.0, 5.0), (5.0, 10.0)])
    monkeypatch.setattr(
        video_ex, "extract_frame", lambda p, ts, out: (out.write_bytes(b"x"), out)[1]
    )
    monkeypatch.setattr(video_ex, "extract_audio_track", lambda p, out: None)
    monkeypatch.setattr(video_ex, "frame_dhash", lambda p: None)
    return cfg, index, path, fid


def test_all_captions_failing_raises(video_item, monkeypatch):
    cfg, index, path, fid = video_item
    client = MagicMock()
    client.generate.side_effect = RuntimeError("connection refused")

    with pytest.raises(video_ex.VLMUnavailable):
        video_ex.process_video(
            path, client, vision_model="v", agent_model="a",
            summarize=False, transcribe=False,
        )


def test_worker_marks_dead_vlm_video_retryable(video_item, monkeypatch):
    cfg, index, path, fid = video_item
    worker = Tier2Worker(cfg, index, client=MagicMock())
    worker.client.generate.side_effect = RuntimeError("connection refused")
    worker.client.ping.return_value = True
    worker.embedder = MagicMock()
    worker.embedder.embed_chunks.side_effect = lambda c: c

    item = index.next_pending(tier=2)
    with pytest.raises(video_ex.VLMUnavailable):
        worker._process(item, path, None)

    # nothing stored, so a retry redoes the work rather than seeing it "done"
    assert index.get_content(fid, "video_scenes") == []


def test_partial_caption_failure_still_succeeds(video_item):
    cfg, index, path, fid = video_item
    client = MagicMock()
    client.generate.side_effect = [RuntimeError("blip"), "a cat walks past"]

    result = video_ex.process_video(
        path, client, vision_model="v", agent_model="a",
        frames_per_scene=1, summarize=False, transcribe=False,
    )
    captions = [c for s in result["scenes"] for c in s["captions"]]
    assert captions == ["a cat walks past"]


def test_unreadable_frames_are_not_treated_as_model_failure(video_item, monkeypatch):
    """No frames decoded => no VLM calls attempted => not a model outage."""
    cfg, index, path, fid = video_item
    monkeypatch.setattr(video_ex, "extract_frame", lambda p, ts, out: None)
    client = MagicMock()

    result = video_ex.process_video(
        path, client, vision_model="v", agent_model="a",
        summarize=False, transcribe=False,
    )
    assert all(s["captions"] == [] for s in result["scenes"])
    client.generate.assert_not_called()


def test_inline_summary_failure_defers_instead_of_completing(video_item, monkeypatch):
    """With defer_video_summaries off, a failed summary must move the item to
    the end-of-run sweep, not mark it done with captions but no summary."""
    cfg, index, path, fid = video_item
    cfg.deep.defer_video_summaries = False
    monkeypatch.setattr(
        video_ex, "summarize_video",
        MagicMock(side_effect=RuntimeError("agent model unavailable")),
    )
    worker = Tier2Worker(cfg, index, client=MagicMock())
    worker.client.generate.return_value = "a caption"
    worker.embedder = MagicMock()
    worker.embedder.embed_chunks.side_effect = lambda c: c

    item = index.next_pending(tier=2)
    outcome = worker._process(item, path, None)

    assert outcome == "defer_summary"
    assert index.get_content(fid, "video_scenes")[0]["body"]  # captions kept
    index.set_queue_status(item["id"], PENDING_SUMMARY)
    index.commit()
    assert index.next_pending(tier=2, status=PENDING_SUMMARY) is not None
