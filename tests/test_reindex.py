"""extractor_version is now actionable: reindex re-queues exactly the files
whose stage version (extractor constant or configured model) has changed."""

import pytest
from typer.testing import CliRunner

from file_index import cli
from file_index.index import PENDING_DEEP, PENDING_METADATA
from file_index.reindex import find_stale, is_stale, requeue, stage_version


def _add(index, path, kind, stage, version, mime="application/octet-stream"):
    path.write_text("x")
    fid = index.upsert_file(str(path), f"h{path.name}", 1, 1000.0, mime, kind)
    index.store_content(fid, stage, version, "body text")
    index.commit()
    return fid


# ---------- version computation ----------


def test_model_dependent_stages_carry_the_model(tmp_env):
    cfg, index, root = tmp_env
    assert stage_version("vlm_image", cfg) == f"image-1.2+{cfg.models.vision}"
    assert stage_version("video_summary", cfg) == f"video-1.0+{cfg.models.agent}"
    assert stage_version("whisper", cfg) == f"whisper-large-v3-1.0+{cfg.models.whisper}"
    # stages with no model are just the extractor constant
    assert stage_version("text", cfg) == "text-1.0"
    assert stage_version("pdf_text", cfg) == "pdf-1.0"


def test_is_stale_rules():
    assert not is_stale("image-1.2+m", "image-1.2+m")
    assert is_stale("image-1.2+old-model", "image-1.2+new-model")
    assert is_stale("image-1.1+m", "image-1.2+m")
    # legacy rows (no model recorded) are judged on the extractor alone
    assert not is_stale("image-1.2", "image-1.2+any-model")
    assert is_stale("image-1.1", "image-1.2+any-model")


# ---------- stale detection ----------


def test_model_swap_marks_only_that_stage_stale(tmp_env):
    cfg, index, root = tmp_env
    img = _add(index, root / "a.png", "image", "vlm_image",
               stage_version("vlm_image", cfg))
    txt = _add(index, root / "b.txt", "text", "text", stage_version("text", cfg))

    assert find_stale(index, cfg) == {}

    cfg.models.vision = "some-better-vlm:70b"
    stale = find_stale(index, cfg)

    assert list(stale) == ["vlm_image"]
    assert [f[0] for f in stale["vlm_image"]] == [img]
    assert txt not in [f[0] for files in stale.values() for f in files]


def test_stage_filter_and_force(tmp_env):
    cfg, index, root = tmp_env
    img = _add(index, root / "a.png", "image", "vlm_image", "image-0.9+old")
    _add(index, root / "b.txt", "text", "text", "text-0.1")

    only = find_stale(index, cfg, stages=["vlm_image"])
    assert list(only) == ["vlm_image"]

    forced = find_stale(index, cfg, stages=["text"], force=True)
    assert list(forced) == ["text"]  # up-to-date rows included under --force

    with pytest.raises(ValueError):
        find_stale(index, cfg, stages=["not_a_stage"])


def test_deleted_files_are_never_requeued(tmp_env):
    cfg, index, root = tmp_env
    fid = _add(index, root / "a.png", "image", "vlm_image", "image-0.9+old")
    index.mark_deleted(fid)
    index.commit()
    assert find_stale(index, cfg) == {}


# ---------- requeue routing ----------


def test_requeue_routes_to_the_right_tier_and_kind(tmp_env):
    cfg, index, root = tmp_env
    img = _add(index, root / "a.png", "image", "vlm_image", "image-0.9+old")
    txt = _add(index, root / "b.txt", "text", "text", "text-0.1")
    pdf = _add(index, root / "c.pdf", "pdf", "pdf_scan_vlm", "image-0.9+old")

    counts = requeue(index, find_stale(index, cfg))

    assert counts == {"tier1": 1, "tier2": 2}
    rows = {
        (r["file_id"], r["tier"]): r
        for r in index.db.execute("SELECT * FROM queue")
    }
    assert rows[(txt, 1)]["status"] == PENDING_METADATA
    assert rows[(img, 2)]["status"] == PENDING_DEEP
    assert rows[(img, 2)]["kind"] == "image"
    assert rows[(pdf, 2)]["kind"] == "pdf_scan"  # scanned-PDF handler, not "pdf"
    assert (txt, 2) not in rows  # untouched stages are not queued


def test_file_stale_in_two_stages_of_one_tier_queues_once(tmp_env):
    cfg, index, root = tmp_env
    path = root / "clip.mp4"
    path.write_text("x")
    fid = index.upsert_file(str(path), "h1", 1, 1000.0, "video/mp4", "video")
    index.store_content(fid, "video_scenes", "video-0.1+old", "scenes")
    index.store_content(fid, "video_summary", "video-0.1+old", "summary")
    index.commit()

    counts = requeue(index, find_stale(index, cfg))

    assert counts == {"tier1": 0, "tier2": 1}
    n = index.db.execute(
        "SELECT count(*) n FROM queue WHERE file_id=?", (fid,)
    ).fetchone()["n"]
    assert n == 1


# ---------- CLI ----------


def test_reindex_cli_dry_run_then_apply(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    _add(index, root / "a.png", "image", "vlm_image", "image-0.9+old")
    monkeypatch.setattr(cli, "_load", lambda: (cfg, index))

    result = CliRunner().invoke(cli.app, ["reindex", "--dry-run"])
    assert result.exit_code == 0
    assert "vlm_image" in result.output and "dry-run" in result.output
    assert index.db.execute("SELECT count(*) n FROM queue").fetchone()["n"] == 0

    result = CliRunner().invoke(cli.app, ["reindex", "--yes"])
    assert result.exit_code == 0
    assert index.db.execute("SELECT count(*) n FROM queue").fetchone()["n"] == 1


def test_reindex_cli_reports_up_to_date(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    _add(index, root / "a.txt", "text", "text", stage_version("text", cfg))
    monkeypatch.setattr(cli, "_load", lambda: (cfg, index))
    result = CliRunner().invoke(cli.app, ["reindex", "--yes"])
    assert "up to date" in result.output


def test_reindex_cli_rejects_unknown_stage(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    monkeypatch.setattr(cli, "_load", lambda: (cfg, index))
    result = CliRunner().invoke(cli.app, ["reindex", "-s", "bogus", "--yes"])
    assert result.exit_code == 1
    # rich colorizes the message, so compare against the plain text
    import re

    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "unknown stage(s): bogus" in plain


def test_worker_stamps_model_aware_versions(tmp_env):
    """A stage written today records the model, so a later swap is detectable."""
    from unittest.mock import MagicMock

    from file_index.queue import Tier2Worker

    cfg, index, root = tmp_env
    worker = Tier2Worker(cfg, index, client=MagicMock())
    assert worker._version("vlm_image") == f"image-1.2+{cfg.models.vision}"
    assert worker._version("video_transcript") == (
        f"whisper-large-v3-1.0+{cfg.models.whisper}"
    )
