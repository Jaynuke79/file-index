"""Tests for deep-pass parallelization (prefetch, parallel frame extraction)
and neighbor context for folders of similar clips."""

import json
from pathlib import Path

from file_index.config import load_config
from file_index.extractors import video as video_ex
from file_index.queue import Tier2Worker
import file_index.queue as queue_mod


class FakeOllama:
    """Stands in for OllamaClient: canned VLM JSON, captions, and embeddings."""

    def __init__(self):
        self.calls = []

    def require(self):
        pass

    def ping(self):
        return True

    def generate(self, model, prompt, images=None, format_json=False):
        self.calls.append({"model": model, "prompt": prompt, "images": images})
        if format_json:
            return json.dumps({
                "description": "a thing", "ocr_text": "", "type": "photo",
                "objects": [], "people_count": 0, "inferred_context": "",
            })
        return "generated text"

    def embed(self, model, texts):
        return [[0.1] * 8 for _ in texts]


def test_peek_pending_matches_next_pending_order(tmp_env):
    cfg, index, root = tmp_env
    for name, kind, mtime in [
        ("v.mp4", "video", 10.0), ("a.jpg", "image", 5.0), ("b.jpg", "image", 20.0),
    ]:
        fid = index.upsert_file(str(root / name), f"h-{name}", 1, mtime, "x/y", kind)
        index.enqueue(fid, 2, "pending_deep", kind, mtime)
    index.commit()

    peeked = [
        r["path"] for r in index.peek_pending(
            tier=2, kind_priority=["image", "video"], newest_first=True, limit=10
        )
    ]
    popped = []
    while True:
        item = index.next_pending(tier=2, kind_priority=["image", "video"], newest_first=True)
        if item is None:
            break
        popped.append(item["path"])
        index.mark_done(item["id"])
    assert peeked == popped
    assert [p.rsplit("/", 1)[-1] for p in popped] == ["b.jpg", "a.jpg", "v.mp4"]


def test_process_video_parallel_frames_and_context(tmp_path, monkeypatch):
    extracted = []

    def fake_extract_frame(path, ts, out):
        extracted.append(ts)
        out.write_bytes(b"jpg")
        return out

    monkeypatch.setattr(video_ex, "extract_frame", fake_extract_frame)
    monkeypatch.setattr(video_ex, "extract_audio_track", lambda p, o: None)
    monkeypatch.setattr(
        video_ex, "detect_scenes",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("detect_scenes must be skipped")),
    )

    client = FakeOllama()
    scenes = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    result = video_ex.process_video(
        tmp_path / "clip.mp4", client, vision_model="vlm", agent_model="agent",
        frames_per_scene=1, scenes=scenes, frame_workers=3,
        context="- Smite match, ranked duel",
    )
    # one frame per scene, captions in scene order, all captioned
    assert len(extracted) == 3
    assert [s["captions"] for s in result["scenes"]] == [["generated text"]] * 3
    assert [round(s["start"]) for s in result["scenes"]] == [0, 10, 20]
    # context reached both the captioner and the summarizer
    caption_calls = [c for c in client.calls if c["model"] == "vlm"]
    summary_calls = [c for c in client.calls if c["model"] == "agent"]
    assert all("Smite match" in c["prompt"] for c in caption_calls)
    assert len(summary_calls) == 1 and "Smite match" in summary_calls[0]["prompt"]
    assert result["summary"] == "generated text"


def test_process_video_no_context_keeps_prompts_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(video_ex, "extract_frame", lambda p, ts, o: (o.write_bytes(b"j"), o)[1])
    monkeypatch.setattr(video_ex, "extract_audio_track", lambda p, o: None)
    client = FakeOllama()
    video_ex.process_video(
        tmp_path / "c.mp4", client, vision_model="vlm", agent_model="agent",
        frames_per_scene=1, scenes=[(0.0, 5.0)],
    )
    assert all("Background" not in c["prompt"] for c in client.calls)


def test_tier2_prefetch_prepares_upcoming_images(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    from PIL import Image

    for i, mtime in enumerate([30.0, 20.0, 10.0]):
        p = root / f"img{i}.png"
        Image.new("RGB", (32, 32), (i * 40, 0, 0)).save(p)
        fid = index.upsert_file(str(p), f"h{i}", p.stat().st_size, mtime, "image/png", "image")
        index.enqueue(fid, 2, "pending_deep", "image", mtime)
    index.commit()

    seen = []
    real_analyze = queue_mod.image_ex.analyze_image

    def spy_analyze(client, model, path, prepared=None):
        seen.append({"path": Path(path), "prepared": prepared})
        return real_analyze(client, model, path, prepared=prepared)

    monkeypatch.setattr(queue_mod.image_ex, "analyze_image", spy_analyze)

    cfg.deep.prefetch_files = 2
    worker = Tier2Worker(cfg, index, client=FakeOllama())
    result = worker.run()
    assert result == {"done": 3, "failed": 0}
    # every file after the first got a prefetched, VLM-safe image
    assert seen[0]["prepared"] is None
    for call in seen[1:]:
        assert call["prepared"] is not None and call["prepared"].exists()
    # captions landed in the index
    n = index.db.execute(
        "SELECT count(*) FROM content WHERE stage='vlm_image'"
    ).fetchone()[0]
    assert n == 3


def test_tier2_prefetch_disabled_still_works(tmp_env):
    cfg, index, root = tmp_env
    from PIL import Image

    p = root / "one.png"
    Image.new("RGB", (16, 16)).save(p)
    fid = index.upsert_file(str(p), "h", p.stat().st_size, 1.0, "image/png", "image")
    index.enqueue(fid, 2, "pending_deep", "image", 1.0)
    index.commit()

    cfg.deep.prefetch_files = 0
    worker = Tier2Worker(cfg, index, client=FakeOllama())
    assert worker.run() == {"done": 1, "failed": 0}


def test_neighbor_context_same_folder_only(tmp_env):
    cfg, index, root = tmp_env
    smite = root / "smite"
    (smite / "sub").mkdir(parents=True)

    def add_video(path, mtime, summary=None):
        fid = index.upsert_file(str(path), f"h-{path.name}-{mtime}", 1, mtime, "video/mp4", "video")
        if summary:
            index.store_content(fid, "video_summary", "video-1.0", summary)
        return fid

    add_video(smite / "a.mp4", 10.0, "Smite match one, Ares vs Thor")
    add_video(smite / "b.mp4", 20.0, "Smite match two, arena mode")
    add_video(smite / "sub" / "c.mp4", 30.0, "Unrelated screen recording")
    target = add_video(smite / "new.mp4", 40.0)
    index.commit()

    worker = Tier2Worker(cfg, index, client=FakeOllama())
    ctx = worker._neighbor_context(target, smite / "new.mp4")
    assert "Smite match one" in ctx and "Smite match two" in ctx
    assert "Unrelated" not in ctx  # subfolder is not the same folder
    # newest sibling first
    assert ctx.index("match two") < ctx.index("match one")

    cfg.deep.neighbor_context = 0
    assert worker._neighbor_context(target, smite / "new.mp4") == ""


def test_new_deep_config_round_trip(tmp_env):
    cfg, index, root = tmp_env
    cfg.deep.prefetch_files = 5
    cfg.deep.video_frame_workers = 7
    cfg.deep.neighbor_context = 9
    path = cfg.config_path
    cfg.save(path)
    loaded = load_config(path)
    assert loaded.deep.prefetch_files == 5
    assert loaded.deep.video_frame_workers == 7
    assert loaded.deep.neighbor_context == 9
