"""Deep-pass progress is measured in pipeline steps (video/audio = 3, rest = 1)
so the bar's total stays fixed while files defer between stages, instead of the
denominator drifting upward as each deferred file got counted both as "done"
and as still pending.
"""

from unittest.mock import MagicMock

import file_index.extractors.video as video_ex
from file_index.index import PENDING_DEEP, PENDING_SUMMARY, PENDING_TRANSCRIPT
from file_index.queue import Tier2Worker


class FakeOllama:
    def __init__(self):
        self.keep_alive = None

    def require(self):
        pass

    def generate(self, model, prompt, images=None, format_json=False):
        return "generated text"

    def embed(self, model, texts):
        return [[0.1] * 8 for _ in texts]


def _enqueue(index, root, name, kind, mime, status=PENDING_DEEP, mtime=100.0):
    p = root / name
    p.write_bytes(b"fake bytes")
    fid = index.upsert_file(str(p), f"h-{name}", 10, mtime, mime, kind)
    index.enqueue(fid, 2, PENDING_DEEP, kind, mtime)
    if status != PENDING_DEEP:
        qid = index.db.execute(
            "SELECT id FROM queue WHERE file_id=? AND tier=2", (fid,)
        ).fetchone()["id"]
        index.set_queue_status(qid, status)
    return fid


def test_pending_units_weights_stages(tmp_env):
    cfg, index, root = tmp_env
    _enqueue(index, root, "a.mp4", "video", "video/mp4")                       # 3
    _enqueue(index, root, "b.mp3", "audio", "audio/mpeg", PENDING_TRANSCRIPT)  # 2
    _enqueue(index, root, "c.jpg", "image", "image/jpeg")                      # 1
    _enqueue(index, root, "d.mp4", "video", "video/mp4", PENDING_SUMMARY)      # 1
    index.commit()

    worker = Tier2Worker(cfg, index, client=MagicMock())
    assert worker.pending_count() == 4
    assert worker.pending_units() == 7


def test_progress_total_never_grows_across_deferrals(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    for i in range(3):
        _enqueue(index, root, f"v{i}.mp4", "video", "video/mp4", mtime=100.0 + i)
    index.commit()

    def fake_process_video(path, client, **kwargs):
        return {
            "scenes": [{"start": 0.0, "end": 5.0, "captions": [f"scene of {path.name}"]}],
            "transcript": None, "summary": "",
        }

    monkeypatch.setattr(video_ex, "process_video", fake_process_video)
    monkeypatch.setattr(
        video_ex, "transcribe_video",
        lambda *a, **k: {"language": "en", "duration": 5.0,
                         "segments": [{"start": 0.0, "end": 2.0, "text": "speech"}]},
    )

    worker = Tier2Worker(cfg, index, client=FakeOllama())
    assert worker.pending_units() == 9  # 3 videos x 3 steps

    seen = []
    result = worker.run(progress_cb=lambda p, done, remaining, eta: seen.append((done, remaining)))

    assert seen[0] == (0, 9)
    # the displayed total (done + remaining) may shrink when a stage is
    # skipped, but must never grow — that was the bug
    assert all(d + r <= 9 for d, r in seen)
    # done counts completed steps, monotonically
    dones = [d for d, _ in seen]
    assert dones == sorted(dones)
    # result counts whole files, and everything drained
    assert result == {"done": 3, "failed": 0}
    assert worker.pending_units() == 0
