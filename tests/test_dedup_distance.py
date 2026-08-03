"""The frame-dedup threshold is tunable from config, not a hardcoded constant."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from file_index.config import Config, load_config
from file_index.extractors import video as video_ex
from file_index.index import PENDING_DEEP
from file_index.queue import Tier2Worker

SCENES = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0)]


@pytest.fixture
def vid(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    path = root / "clip.mp4"
    path.write_bytes(b"video")
    fid = index.upsert_file(str(path), "h1", 5, 1000.0, "video/mp4", "video")
    index.enqueue(fid, 2, PENDING_DEEP, "video", 1000.0)
    index.commit()
    monkeypatch.setattr(video_ex, "detect_scenes", lambda p, i: list(SCENES))
    monkeypatch.setattr(video_ex, "extract_audio_track", lambda p, out: None)
    monkeypatch.setattr(
        video_ex, "extract_frame",
        lambda p, ts, out: (out.write_bytes(b"jpeg"), out)[1],
    )
    return cfg, index, path, fid


def _captions_with(distance, hashes, **kw):
    """Run the pipeline with a scripted hash per extracted frame."""
    seq = iter(hashes)
    client = MagicMock()
    client.generate.return_value = "a caption"
    return client, seq, distance


def test_distance_controls_how_much_is_skipped(vid, monkeypatch):
    cfg, index, path, fid = vid
    # Pairwise hamming distances are all >= 4, so the threshold alone decides:
    # at distance 2 every frame survives; at 6 each is folded into an earlier
    # one until something sits 8 bits away.
    hashes = [0b00000000, 0b00001111, 0b11110000, 0b11111111]
    client = MagicMock()
    client.generate.return_value = "cap"

    seq = iter(hashes)
    monkeypatch.setattr(video_ex, "frame_dhash", lambda p: next(seq))
    strict = video_ex.process_video(
        path, client, vision_model="v", agent_model="a", frames_per_scene=1,
        summarize=False, transcribe=False, dedup_frames=True, dedup_distance=2,
    )
    n_strict = sum(len(s["captions"]) for s in strict["scenes"])

    seq = iter(hashes)
    monkeypatch.setattr(video_ex, "frame_dhash", lambda p: next(seq))
    loose = video_ex.process_video(
        path, client, vision_model="v", agent_model="a", frames_per_scene=1,
        summarize=False, transcribe=False, dedup_frames=True, dedup_distance=6,
    )
    n_loose = sum(len(s["captions"]) for s in loose["scenes"])

    assert n_strict == 4, "distance 2 keeps all four distinct-enough frames"
    assert n_loose == 2, "distance 6 folds the two 4-bit neighbours away"
    assert n_loose < n_strict


def test_default_distance_matches_the_historical_constant(vid, monkeypatch):
    """Behaviour must be unchanged for anyone who does not set the new key."""
    assert Config().deep.video_dedup_distance == video_ex.DHASH_NEAR_DUPLICATE


def test_worker_passes_the_configured_distance(vid, monkeypatch):
    cfg, index, path, fid = vid
    cfg.deep.video_dedup_distance = 9
    seen = {}
    monkeypatch.setattr(
        video_ex, "process_video",
        lambda *a, **kw: seen.update(kw) or {
            "scenes": [], "transcript": None, "summary": "",
            "captions_text": "", "transcript_text": "", "frames_undecodable": False},
    )
    worker = Tier2Worker(cfg, index, client=MagicMock())
    worker.embedder = MagicMock()
    worker.embedder.embed_chunks.side_effect = lambda c: c

    worker._process(index.next_pending(tier=2), path, None)

    assert seen["dedup_distance"] == 9


def test_config_round_trip(tmp_path):
    cfg = Config()
    cfg.roots = [tmp_path]
    cfg.data_dir = tmp_path / "d"
    cfg.deep.video_dedup_distance = 10
    p = cfg.save(tmp_path / "config.yaml")
    assert load_config(p).deep.video_dedup_distance == 10
