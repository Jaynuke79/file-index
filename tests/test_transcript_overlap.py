"""Background transcription overlap, scene capping, frame dedup, keep_alive."""

import json
from pathlib import Path

import file_index.extractors.video as video_ex
from file_index.config import Config, load_config
from file_index.ollama_client import OllamaClient
from file_index.queue import Tier2Worker


class FakeOllama:
    def __init__(self):
        self.calls = []
        self.keep_alive = None

    def require(self):
        pass

    def generate(self, model, prompt, images=None, format_json=False):
        self.calls.append({"model": model, "prompt": prompt})
        return "generated text"

    def embed(self, model, texts):
        return [[0.1] * 8 for _ in texts]


def enqueue_video(index, root, name, mtime=100.0):
    p = root / name
    p.write_bytes(b"fake video bytes")
    fid = index.upsert_file(str(p), f"h-{name}", 16, mtime, "video/mp4", "video")
    index.enqueue(fid, 2, "pending_deep", "video", mtime)
    return fid, p


def fake_process_video(path, client, **kwargs):
    assert kwargs.get("transcribe") is False  # whisper must not run inline
    assert kwargs.get("summarize") is False
    return {
        "scenes": [{"start": 0.0, "end": 5.0, "captions": [f"scene of {path.name}"]}],
        "transcript": None, "summary": "",
        "captions_text": f"scene of {path.name}",
        "transcript_text": "(no speech / no audio track)",
    }


def test_transcription_runs_in_background_and_stores(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    fids = [enqueue_video(index, root, f"v{i}.mp4", mtime=100.0 + i)[0] for i in range(3)]
    index.commit()
    monkeypatch.setattr(video_ex, "process_video", fake_process_video)
    devices = []

    def fake_transcribe(path, model, device, compute, num_workers=1):
        devices.append(device)
        return {"language": "en", "duration": 5.0,
                "segments": [{"start": 0.0, "end": 2.0,
                              "text": f"speech in {Path(path).name}"}]}

    monkeypatch.setattr(video_ex, "transcribe_video", fake_transcribe)

    worker = Tier2Worker(cfg, index, client=FakeOllama())
    result = worker.run()
    assert result == {"done": 3, "failed": 0}
    # background whisper must never take VRAM from the pinned vision model
    assert devices and all(d == "cpu" for d in devices)
    for fid in fids:
        t = index.get_content(fid, "video_transcript")
        assert t and "speech in" in t[0]["body"]
        assert index.get_content(fid, "video_summary")
        row = index.db.execute("SELECT status FROM queue WHERE file_id=?", (fid,)).fetchone()
        assert row["status"] == "done"
    # nothing left in any intermediate state
    assert worker.pending_count() == 0


def test_resume_from_pending_transcript(tmp_env, monkeypatch):
    """Interrupted mid-transcription: captions survive, transcript is redone."""
    cfg, index, root = tmp_env
    fid, _ = enqueue_video(index, root, "clip.mp4")
    index.store_content(fid, "video_scenes", "video-1.0", "[00:00 - 00:05] a boat")
    index.db.execute("UPDATE queue SET status='pending_transcript' WHERE file_id=?", (fid,))
    index.commit()
    monkeypatch.setattr(
        video_ex, "transcribe_video",
        lambda *a, **k: {"language": "en", "duration": 3.0,
                         "segments": [{"start": 0.0, "end": 1.0, "text": "ahoy"}]},
    )

    worker = Tier2Worker(cfg, index, client=FakeOllama())
    worker.run()
    assert "ahoy" in index.get_content(fid, "video_transcript")[0]["body"]
    assert index.get_content(fid, "video_summary")
    assert index.db.execute(
        "SELECT status FROM queue WHERE file_id=?", (fid,)
    ).fetchone()["status"] == "done"


def test_whisper_failure_is_not_fatal(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    fid, _ = enqueue_video(index, root, "clip.mp4")
    index.store_content(fid, "video_scenes", "video-1.0", "[00:00 - 00:05] a dog")
    index.db.execute("UPDATE queue SET status='pending_transcript' WHERE file_id=?", (fid,))
    index.commit()
    monkeypatch.setattr(
        video_ex, "transcribe_video",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("whisper exploded")),
    )

    worker = Tier2Worker(cfg, index, client=FakeOllama())
    result = worker.run()
    assert result["failed"] == 0
    # no transcript, but the summary still happened from the captions
    assert not index.get_content(fid, "video_transcript")
    assert index.get_content(fid, "video_summary")


def test_sample_scenes_even_spread():
    scenes = [(float(i), float(i + 1)) for i in range(100)]
    sampled = video_ex.sample_scenes(scenes, 10)
    assert len(sampled) == 10
    assert sampled[0] == scenes[0]
    starts = [s[0] for s in sampled]
    assert starts == sorted(starts)
    assert starts[-1] >= 90.0  # reaches the end of the timeline
    assert video_ex.sample_scenes(scenes, 0) == scenes
    assert video_ex.sample_scenes(scenes, 200) == scenes


def test_dedup_skips_near_duplicate_frames(tmp_path, monkeypatch):
    from PIL import Image

    def gradient(seed):
        img = Image.new("L", (64, 64))
        img.putdata([(x * 3 + y + seed * 40) % 256 for y in range(64) for x in range(64)])
        return img

    frames = {"dup": gradient(0), "other": gradient(3)}
    order = ["dup", "dup", "other"]  # scene frames: two identical, one distinct

    def fake_extract_frame(path, ts, out):
        frames[order.pop(0)].save(out, "PNG")
        return out

    monkeypatch.setattr(video_ex, "extract_frame", fake_extract_frame)
    client = FakeOllama()
    result = video_ex.process_video(
        tmp_path / "c.mp4", client, vision_model="vlm", agent_model="agent",
        frames_per_scene=1, scenes=[(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)],
        transcribe=False, summarize=False, dedup_frames=True,
    )
    vlm_calls = [c for c in client.calls if c["model"] == "vlm"]
    assert len(vlm_calls) == 2  # second duplicate frame skipped
    assert [len(s["captions"]) for s in result["scenes"]] == [1, 0, 1]


def test_keep_alive_sent_when_set(monkeypatch):
    sent = {}

    class FakeResp:
        ok = True

        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "ok", "embeddings": [[0.0]]}

    def fake_post(url, json=None, timeout=None, **kw):
        sent[url.rsplit("/", 1)[-1]] = json
        return FakeResp()

    import file_index.ollama_client as oc
    monkeypatch.setattr(oc.requests, "post", fake_post)
    client = OllamaClient()
    client.generate("m", "hi")
    assert "keep_alive" not in sent["generate"]  # unset by default
    client.keep_alive = -1
    client.generate("m", "hi")
    client.embed("m", ["x"])
    assert sent["generate"]["keep_alive"] == -1
    assert sent["embed"]["keep_alive"] == -1


def test_worker_pins_models_during_run(tmp_env):
    cfg, index, root = tmp_env
    client = FakeOllama()
    Tier2Worker(cfg, index, client=client).run()  # empty queue
    assert client.keep_alive == -1


def test_new_config_defaults_and_round_trip(tmp_env):
    cfg, index, root = tmp_env
    fresh = Config()
    assert fresh.deep.prefetch_files == 4
    assert fresh.deep.video_max_scenes == 40
    assert fresh.deep.video_dedup_frames is True
    cfg.deep.video_max_scenes = 12
    cfg.deep.video_dedup_frames = False
    cfg.save(cfg.config_path)
    loaded = load_config(cfg.config_path)
    assert loaded.deep.video_max_scenes == 12
    assert loaded.deep.video_dedup_frames is False
