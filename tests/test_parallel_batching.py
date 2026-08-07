"""Concurrency knobs: parallel VLM captions within a video and a parallel
summary sweep. Both must overlap model calls without reordering results or
moving DB writes off the main thread.
"""

import threading
import time
from pathlib import Path

import file_index.extractors.video as video_ex
from file_index.index import PENDING_SUMMARY
from file_index.queue import Tier2Worker


class ConcurrencyClient:
    """Fake Ollama that records how many generate() calls overlap."""

    def __init__(self, reply="generated text", delay=0.05):
        self.reply = reply
        self.delay = delay
        self.keep_alive = None
        self.active = 0
        self.max_active = 0
        self.calls = []
        self._lock = threading.Lock()

    def require(self):
        pass

    def generate(self, model, prompt, images=None, format_json=False):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append(images[0].name if images else None)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1
        if images:
            return f"caption of {images[0].name}"
        return self.reply

    def embed(self, model, texts):
        return [[0.1] * 8 for _ in texts]


def test_captions_run_concurrently_and_stay_in_scene_order(tmp_path, monkeypatch):
    def fake_extract(path, ts, out_path):
        out_path.write_bytes(b"img")
        return out_path

    monkeypatch.setattr(video_ex, "extract_frame", fake_extract)
    client = ConcurrencyClient()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    result = video_ex.process_video(
        video, client, vision_model="v", agent_model="a",
        frames_per_scene=1, scenes=[(float(i), float(i) + 1) for i in range(6)],
        frame_workers=2, caption_workers=3,
        summarize=False, transcribe=False, dedup_frames=False,
    )

    assert client.max_active >= 2  # captions actually overlapped
    captions = [s["captions"][0] for s in result["scenes"]]
    assert captions == [f"caption of s{i}_f0.jpg" for i in range(6)]  # order kept


def test_serial_captions_still_work(tmp_path, monkeypatch):
    monkeypatch.setattr(
        video_ex, "extract_frame",
        lambda path, ts, out: (out.write_bytes(b"img"), out)[1],
    )
    client = ConcurrencyClient(delay=0)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    result = video_ex.process_video(
        video, client, vision_model="v", agent_model="a",
        frames_per_scene=1, scenes=[(0.0, 1.0), (1.0, 2.0)],
        caption_workers=1, summarize=False, transcribe=False, dedup_frames=False,
    )
    assert [s["captions"][0] for s in result["scenes"]] == [
        "caption of s0_f0.jpg", "caption of s1_f0.jpg",
    ]


def _pending_summary_video(index, root, name, mtime):
    p = root / name
    p.write_bytes(b"fake video")
    fid = index.upsert_file(str(p), f"h-{name}", 10, mtime, "video/mp4", "video")
    index.enqueue(fid, 2, PENDING_SUMMARY, "video", mtime)
    index.store_content(fid, "video_scenes", "video-1.0", f"[00:00 - 00:05] scene of {name}")
    return fid


def test_summary_sweep_runs_workers_in_parallel(tmp_env):
    cfg, index, root = tmp_env
    cfg.deep.summary_workers = 3
    fids = [
        _pending_summary_video(index, root, f"v{i}.mp4", 100.0 + i) for i in range(6)
    ]
    index.commit()

    worker = Tier2Worker(cfg, index, client=ConcurrencyClient(reply="a summary"))
    result = worker.run()

    assert result == {"done": 6, "failed": 0}
    assert worker.client.max_active >= 2  # summaries overlapped
    for fid in fids:
        assert index.get_content(fid, "video_summary")[0]["body"] == "a summary"
        row = index.db.execute(
            "SELECT status FROM queue WHERE file_id=?", (fid,)
        ).fetchone()
        assert row["status"] == "done"


def test_summary_sweep_serial_when_one_worker(tmp_env):
    cfg, index, root = tmp_env
    cfg.deep.summary_workers = 1
    _pending_summary_video(index, root, "v.mp4", 100.0)
    index.commit()

    worker = Tier2Worker(cfg, index, client=ConcurrencyClient(reply="a summary"))
    result = worker.run()
    assert result == {"done": 1, "failed": 0}
    assert worker.client.max_active == 1
