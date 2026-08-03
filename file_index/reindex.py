"""Stage versioning and selective re-extraction.

Each `content` row records the `extractor_version` that produced it. This
module turns that record into something actionable: it computes the version a
stage *would* produce today — including the configured model for stages whose
output depends on one — and finds the files whose stored version no longer
matches, so only those are re-queued.
"""

from __future__ import annotations

from .config import Config
from .extractors import audio as audio_ex
from .extractors import image as image_ex
from .extractors import office as office_ex
from .extractors import pdf as pdf_ex
from .extractors import text as text_ex
from .extractors import video as video_ex
from .index import PENDING_DEEP, PENDING_METADATA, Index

# stage -> (tier, queue kind). The queue kind routes tier-2 work to the right
# handler; tier-1 rows are re-queued under the file's own kind.
STAGE_ROUTES: dict[str, tuple[int, str | None]] = {
    "text": (1, None),
    "pdf_text": (1, None),
    "office": (1, None),
    "exif": (1, None),
    "vlm_image": (2, "image"),
    "pdf_scan_vlm": (2, "pdf_scan"),
    "whisper": (2, "audio"),
    "audio_summary": (2, "audio"),
    "video_scenes": (2, "video"),
    "video_transcript": (2, "video"),
    "video_summary": (2, "video"),
}

# Which configured model a stage's output depends on, if any.
_STAGE_MODEL = {
    "vlm_image": "vision",
    "pdf_scan_vlm": "vision",
    "video_scenes": "vision",
    "whisper": "whisper",
    "video_transcript": "whisper",
    "audio_summary": "agent",
    "video_summary": "agent",
}

_STAGE_BASE = {
    "text": text_ex.VERSION,
    "pdf_text": pdf_ex.VERSION,
    "office": office_ex.VERSION,
    "exif": image_ex.VERSION,
    "vlm_image": image_ex.VERSION,
    "pdf_scan_vlm": image_ex.VERSION,
    "whisper": audio_ex.VERSION,
    "audio_summary": audio_ex.VERSION,
    "video_scenes": video_ex.VERSION,
    "video_transcript": audio_ex.VERSION,
    "video_summary": video_ex.VERSION,
}


def stage_version(stage: str, config: Config) -> str:
    """The version string this stage produces right now.

    Model-dependent stages carry the model name (`image-1.2+qwen2.5vl:32b`) so
    swapping models in config.yaml is visible to `reindex`; other stages are
    just the extractor constant.
    """
    base = _STAGE_BASE.get(stage, stage)
    model_role = _STAGE_MODEL.get(stage)
    if not model_role:
        return base
    return f"{base}+{getattr(config.models, model_role)}"


def is_stale(stored: str, current: str) -> bool:
    """Does `stored` need re-extraction to become `current`?

    Rows written before versions carried a model name have no `+model` part;
    those are compared on the extractor version alone rather than assuming the
    model changed, so upgrading does not re-run everything once.
    """
    if stored == current:
        return False
    if "+" not in stored and "+" in current:
        return stored != current.split("+", 1)[0]
    return True


def find_stale(
    index: Index, config: Config, stages: list[str] | None = None, force: bool = False
) -> dict[str, list[tuple[int, str, float, str]]]:
    """Group re-extractable files by stage.

    Returns {stage: [(file_id, path, mtime, file_kind), ...]} for every live
    file whose stored version for that stage is out of date (or every file
    with the stage, when `force`).
    """
    wanted = set(stages) if stages else set(STAGE_ROUTES)
    unknown = wanted - set(STAGE_ROUTES)
    if unknown:
        raise ValueError(f"unknown stage(s): {', '.join(sorted(unknown))}")
    out: dict[str, list[tuple[int, str, float, str]]] = {}
    marks = ",".join("?" * len(wanted))
    rows = index.db.execute(
        f"SELECT c.stage, c.extractor_version, f.id, f.path, f.mtime, f.kind "
        f"FROM content c JOIN files f ON f.id=c.file_id "
        f"WHERE f.deleted=0 AND c.stage IN ({marks})",
        sorted(wanted),
    ).fetchall()
    for r in rows:
        current = stage_version(r["stage"], config)
        if force or is_stale(r["extractor_version"], current):
            out.setdefault(r["stage"], []).append(
                (r["id"], r["path"], r["mtime"], r["kind"])
            )
    return out


def requeue(index: Index, stale: dict[str, list[tuple[int, str, float, str]]]) -> dict:
    """Re-enqueue the given files at the tier that owns each stale stage.

    Enqueueing is per (file, tier), so a file stale in several stages of the
    same tier is queued once; a stage from the other tier is untouched.
    """
    queued: dict[int, set[int]] = {1: set(), 2: set()}
    for stage, files in stale.items():
        tier, kind = STAGE_ROUTES[stage]
        for file_id, _path, mtime, file_kind in files:
            if file_id in queued[tier]:
                continue
            index.enqueue(
                file_id, tier,
                PENDING_METADATA if tier == 1 else PENDING_DEEP,
                kind or file_kind, mtime or 0.0,
            )
            queued[tier].add(file_id)
    index.commit()
    return {"tier1": len(queued[1]), "tier2": len(queued[2])}
