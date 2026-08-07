"""Audio uses the same off-critical-path machinery as video: CPU background
transcription and a deferred, agent-model-once summary sweep.

Before this, _process_audio transcribed inline on the configured device
(cuda by default, next to the pinned vision model) and then called the agent
model for a summary — a vision<->agent VRAM swap per audio file.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from file_index.index import PENDING_DEEP, PENDING_SUMMARY, PENDING_TRANSCRIPT
from file_index.queue import Tier2Worker

TRANSCRIPT = {
    "language": "en",
    "duration": 12.0,
    "segments": [{"start": 0.0, "end": 12.0, "text": "quarterly numbers look good"}],
    "text": "quarterly numbers look good",
}


@pytest.fixture
def audio_item(tmp_env):
    cfg, index, root = tmp_env
    path = root / "memo.mp3"
    path.write_bytes(b"fake audio")
    fid = index.upsert_file(str(path), "h1", 10, 1000.0, "audio/mpeg", "audio")
    index.enqueue(fid, 2, PENDING_DEEP, "audio", 1000.0)
    index.commit()
    worker = Tier2Worker(cfg, index, client=MagicMock())
    worker.embedder = MagicMock()
    worker.embedder.embed_chunks.side_effect = lambda c: c
    return cfg, index, path, fid, worker


def test_audio_defers_instead_of_transcribing_inline(audio_item):
    cfg, index, path, fid, worker = audio_item
    item = index.next_pending(tier=2)

    outcome = worker._process(item, path, None)

    assert outcome == "defer_transcript"
    # nothing ran on the critical path: no Whisper, no agent model
    worker.client.generate.assert_not_called()
    assert index.get_content(fid, "whisper") == []


def test_background_transcription_uses_cpu_and_stores_whisper_stage(audio_item, monkeypatch):
    cfg, index, path, fid, worker = audio_item
    from file_index.extractors import audio as audio_ex

    calls = []

    def fake_transcribe(p, model, device, compute, num_workers=1):
        calls.append((Path(p).name, device, compute))
        return TRANSCRIPT

    monkeypatch.setattr(audio_ex, "transcribe", fake_transcribe)

    item = index.next_pending(tier=2)
    index.set_queue_status(item["id"], PENDING_TRANSCRIPT)
    index.commit()

    in_flight: dict = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker._pump_transcripts(pool, in_flight)
        assert len(in_flight) == 1
        worker._drain_transcripts(in_flight, wait=True)

    assert calls == [("memo.mp3", "cpu", "int8")]  # never the configured cuda
    rows = index.get_content(fid, "whisper")
    assert rows and "quarterly numbers" in rows[0]["body"]
    assert index.get_content(fid, "video_transcript") == []  # audio stage, not video
    # advanced to the summary sweep
    assert index.db.execute(
        "SELECT status FROM queue WHERE file_id=?", (fid,)
    ).fetchone()["status"] == PENDING_SUMMARY


def test_summary_sweep_writes_audio_summary(audio_item):
    cfg, index, path, fid, worker = audio_item
    worker._store_transcript(fid, TRANSCRIPT, "audio")
    item = index.next_pending(tier=2)
    index.set_queue_status(item["id"], PENDING_SUMMARY)
    index.commit()
    worker.client.generate.return_value = "A short finance memo."

    sweep_item = index.next_pending(tier=2, status=PENDING_SUMMARY)
    worker._summarize_deferred(sweep_item, path)

    rows = index.get_content(fid, "audio_summary")
    assert rows and rows[0]["body"] == "A short finance memo."
    prompt = worker.client.generate.call_args[0][1]
    assert "quarterly numbers look good" in prompt


def test_silent_audio_produces_no_summary(audio_item):
    cfg, index, path, fid, worker = audio_item
    item = index.next_pending(tier=2)
    index.set_queue_status(item["id"], PENDING_SUMMARY)
    index.commit()

    sweep_item = index.next_pending(tier=2, status=PENDING_SUMMARY)
    worker._summarize_deferred(sweep_item, path)

    worker.client.generate.assert_not_called()
    assert index.get_content(fid, "audio_summary") == []


def test_inline_mode_still_available(audio_item, monkeypatch):
    cfg, index, path, fid, worker = audio_item
    cfg.deep.defer_video_summaries = False
    from file_index.extractors import audio as audio_ex

    monkeypatch.setattr(audio_ex, "transcribe", lambda *a, **k: TRANSCRIPT)
    worker.client.generate.return_value = "inline summary"

    item = index.next_pending(tier=2)
    outcome = worker._process(item, path, None)

    assert outcome is None  # completes in place
    assert index.get_content(fid, "whisper")
    assert index.get_content(fid, "audio_summary")[0]["body"] == "inline summary"


def test_interrupted_run_resumes_pending_audio(audio_item, monkeypatch):
    """A killed run leaves the audio item in pending_transcript; the next run
    picks it up without redoing anything."""
    cfg, index, path, fid, worker = audio_item
    from file_index.extractors import audio as audio_ex

    monkeypatch.setattr(audio_ex, "transcribe", lambda *a, **k: TRANSCRIPT)
    item = index.next_pending(tier=2)
    index.set_queue_status(item["id"], PENDING_TRANSCRIPT)
    index.commit()

    fresh = Tier2Worker(cfg, index, client=MagicMock())
    fresh.embedder = MagicMock()
    fresh.embedder.embed_chunks.side_effect = lambda c: c
    in_flight: dict = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        fresh._pump_transcripts(pool, in_flight)
        fresh._drain_transcripts(in_flight, wait=True)

    assert index.get_content(fid, "whisper")
    assert fresh.pending_count() == 1  # now waiting on the summary sweep
