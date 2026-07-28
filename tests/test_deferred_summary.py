"""Deferred video summaries: captions/transcripts store immediately, the
summary runs in an end-of-run sweep with the agent model loaded once."""

from pathlib import Path

import file_index.extractors.video as video_ex
from file_index.config import load_config
from file_index.queue import Tier2Worker


class FakeOllama:
    def __init__(self, fail_agent=False):
        self.calls = []
        self.fail_agent = fail_agent

    def require(self):
        pass

    def generate(self, model, prompt, images=None, format_json=False):
        self.calls.append({"model": model, "prompt": prompt})
        if self.fail_agent and model == "qwen3:30b-a3b":
            raise RuntimeError("agent model unavailable")
        return "generated summary text"

    def embed(self, model, texts):
        return [[0.1] * 8 for _ in texts]


def enqueue_video(index, root, name, mtime=100.0):
    p = root / name
    p.write_bytes(b"fake video bytes")
    fid = index.upsert_file(str(p), f"h-{name}", 16, mtime, "video/mp4", "video")
    index.enqueue(fid, 2, "pending_deep", "video", mtime)
    return fid, p


def fake_process_video(path, client, **kwargs):
    assert kwargs.get("summarize") is False  # worker must defer the summary
    return {
        "scenes": [{"start": 0.0, "end": 5.0, "captions": ["a knight fighting"]}],
        "transcript": None,
        "summary": "",
        "captions_text": "[00:00 - 00:05] a knight fighting",
        "transcript_text": "(no speech / no audio track)",
    }


def test_deferred_summary_full_run(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    fid, _ = enqueue_video(index, root, "clip.mp4")
    index.commit()
    monkeypatch.setattr(video_ex, "process_video", fake_process_video)

    client = FakeOllama()
    worker = Tier2Worker(cfg, index, client=client)
    result = worker.run()
    assert result == {"done": 1, "failed": 0}

    # scenes stored by the caption phase, summary stored by the sweep
    assert index.get_content(fid, "video_scenes")
    summary = index.get_content(fid, "video_summary")
    assert summary and summary[0]["body"] == "generated summary text"
    row = index.db.execute("SELECT status FROM queue WHERE file_id=?", (fid,)).fetchone()
    assert row["status"] == "done"
    tier2 = index.db.execute("SELECT tier2_status FROM files WHERE id=?", (fid,)).fetchone()
    assert tier2["tier2_status"] == "done"
    # the sweep prompt was built from the stored scene captions
    agent_calls = [c for c in client.calls if c["model"] == "qwen3:30b-a3b"]
    assert len(agent_calls) == 1
    assert "a knight fighting" in agent_calls[0]["prompt"]


def test_resume_pending_summary_from_previous_run(tmp_env):
    """A run killed between captions and sweep resumes summary-only."""
    cfg, index, root = tmp_env
    fid, _ = enqueue_video(index, root, "clip.mp4")
    index.store_content(fid, "video_scenes", "video-1.0", "[00:00 - 00:05] two people talking")
    index.store_content(fid, "video_transcript", "whisper-large-v3-1.0", "[00:01 - 00:04] hello there")
    index.db.execute("UPDATE queue SET status='pending_summary' WHERE file_id=?", (fid,))
    index.commit()

    client = FakeOllama()
    worker = Tier2Worker(cfg, index, client=client)
    worker.run()

    summary = index.get_content(fid, "video_summary")
    assert summary and summary[0]["body"] == "generated summary text"
    prompt = [c for c in client.calls if c["model"] == "qwen3:30b-a3b"][0]["prompt"]
    assert "two people talking" in prompt and "hello there" in prompt
    # vision model was never touched
    assert all(c["model"] == "qwen3:30b-a3b" for c in client.calls)


def test_summary_failure_retries_without_redoing_captions(tmp_env):
    cfg, index, root = tmp_env
    fid, _ = enqueue_video(index, root, "clip.mp4")
    index.store_content(fid, "video_scenes", "video-1.0", "[00:00 - 00:05] a dog")
    index.db.execute("UPDATE queue SET status='pending_summary' WHERE file_id=?", (fid,))
    index.commit()

    worker = Tier2Worker(cfg, index, client=FakeOllama(fail_agent=True))
    result = worker.run()
    assert result["failed"] == 1

    row = index.db.execute(
        "SELECT status, retries, error FROM queue WHERE file_id=?", (fid,)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["retries"] == cfg.limits.max_retries
    assert "agent model unavailable" in row["error"]
    # the captions survived every retry — only the summary step reran
    assert index.get_content(fid, "video_scenes")
    assert not index.get_content(fid, "video_summary")


def test_inline_mode_when_defer_disabled(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    cfg.deep.defer_video_summaries = False
    fid, _ = enqueue_video(index, root, "clip.mp4")
    index.commit()

    def inline_process_video(path, client, **kwargs):
        assert kwargs.get("summarize") is True
        return {
            "scenes": [{"start": 0.0, "end": 5.0, "captions": ["x"]}],
            "transcript": None, "summary": "inline summary",
            "captions_text": "x", "transcript_text": "(no speech / no audio track)",
        }

    monkeypatch.setattr(video_ex, "process_video", inline_process_video)
    worker = Tier2Worker(cfg, index, client=FakeOllama())
    assert worker.run() == {"done": 1, "failed": 0}
    summary = index.get_content(fid, "video_summary")
    assert summary and summary[0]["body"] == "inline summary"
    assert index.db.execute(
        "SELECT count(*) FROM queue WHERE status='pending_summary'"
    ).fetchone()[0] == 0


def test_neighbor_context_falls_back_to_scene_captions(tmp_env):
    cfg, index, root = tmp_env
    sib = index.upsert_file(str(root / "a.mp4"), "h-a", 1, 10.0, "video/mp4", "video")
    index.store_content(sib, "video_scenes", "video-1.0", "[00:00 - 00:10] Smite arena gameplay")
    target = index.upsert_file(str(root / "b.mp4"), "h-b", 1, 20.0, "video/mp4", "video")
    index.commit()

    worker = Tier2Worker(cfg, index, client=FakeOllama())
    ctx = worker._neighbor_context(target, root / "b.mp4")
    assert "(scene captions)" in ctx and "Smite arena gameplay" in ctx


def test_defer_flag_config_round_trip(tmp_env):
    cfg, index, root = tmp_env
    cfg.deep.defer_video_summaries = False
    cfg.save(cfg.config_path)
    assert load_config(cfg.config_path).deep.defer_video_summaries is False
