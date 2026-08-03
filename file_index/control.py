"""Settings edits and filesystem browsing for the web control panel.

Every mutation goes through `Config.save()` — the same writer `init` and
`exclude` use — so a config edited in the UI stays byte-compatible with the CLI
and hand editing.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import Config

# deep/limits fields the UI may set, with their coercion. Anything absent here
# is not editable from the browser, so a stray key cannot reach the config.
DEEP_FIELDS: dict[str, type] = {
    "newest_first": bool,
    "video_frames_per_scene": int,
    "video_fallback_interval_s": float,
    "whisper_device": str,
    "whisper_compute_type": str,
    "prefetch_files": int,
    "video_frame_workers": int,
    "video_max_scenes": int,
    "video_dedup_frames": bool,
    "video_dedup_distance": int,
    "neighbor_context": int,
    "defer_video_summaries": bool,
}
MODEL_FIELDS = ("agent", "vision", "embed", "whisper", "ollama_url")
LIMIT_FIELDS: dict[str, type] = {
    "text_size_cap": int,
    "chunk_tokens": int,
    "chunk_overlap_tokens": int,
    "max_retries": int,
    "max_file_size": int,
}

NON_NEGATIVE = {
    "video_frames_per_scene", "prefetch_files", "video_frame_workers",
    "video_max_scenes", "video_dedup_distance", "neighbor_context",
    "text_size_cap", "chunk_tokens", "chunk_overlap_tokens", "max_retries",
    "max_file_size", "video_fallback_interval_s",
}


class SettingsError(ValueError):
    """A rejected edit, with a message meant for the user."""


def settings_payload(cfg: Config) -> dict:
    """Everything the settings view renders."""
    d = cfg.to_dict()
    return {
        "config_path": str(cfg.config_path),
        "data_dir": str(cfg.data_dir),
        "roots": d["roots"],
        "excludes": d["excludes"],
        "models": d["models"],
        "limits": d["limits"],
        "deep": d["deep"],
        "editable": {
            "deep": sorted(DEEP_FIELDS),
            "models": list(MODEL_FIELDS),
            "limits": sorted(LIMIT_FIELDS),
        },
    }


def _coerce(name: str, typ: type, raw):
    if typ is bool:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    try:
        val = typ(raw)
    except (TypeError, ValueError) as e:
        raise SettingsError(f"{name}: expected {typ.__name__}, got {raw!r}") from e
    if name in NON_NEGATIVE and val < 0:
        raise SettingsError(f"{name} cannot be negative")
    return val


def add_root(cfg: Config, raw: str) -> str:
    """Whitelist another directory. Returns the resolved path."""
    if not raw or not raw.strip():
        raise SettingsError("no directory given")
    p = Path(raw).expanduser()
    try:
        p = p.resolve()
    except OSError as e:
        raise SettingsError(f"{raw}: {e}") from e
    if not p.is_dir():
        raise SettingsError(f"{p} is not a directory")
    if not os.access(p, os.R_OK | os.X_OK):
        raise SettingsError(f"{p} is not readable")
    existing = [Path(r).expanduser().resolve() for r in map(str, cfg.roots)]
    if p in existing:
        raise SettingsError(f"{p} is already indexed")
    # Adding a parent of an existing root would double-index the child.
    for e in existing:
        if p == e or p in e.parents:
            raise SettingsError(f"{p} already contains indexed root {e}")
        if e in p.parents:
            raise SettingsError(f"{p} is already covered by indexed root {e}")
    cfg.roots = [*cfg.roots, p]
    cfg.save()
    return str(p)


def remove_root(cfg: Config, raw: str) -> str:
    """Stop indexing a directory. Already-indexed files are left in the index —
    `exclude` or `purge` remove those, and this stays reversible."""
    target = Path(raw).expanduser()
    kept, removed = [], None
    for r in cfg.roots:
        rp = Path(str(r)).expanduser()
        if rp == target or str(rp) == str(target):
            removed = str(rp)
        else:
            kept.append(r)
    if removed is None:
        raise SettingsError(f"{raw} is not an indexed root")
    if not kept:
        raise SettingsError(
            "at least one root is required — add another before removing this one"
        )
    cfg.roots = kept
    cfg.save()
    return removed


def add_exclude(cfg: Config, pattern: str) -> str:
    from .crawler import exclude_pattern

    if not pattern or not pattern.strip():
        raise SettingsError("no pattern given")
    norm = exclude_pattern(pattern.strip())
    if norm in cfg.excludes:
        raise SettingsError(f"{norm} is already excluded")
    cfg.excludes = [*cfg.excludes, norm]
    cfg.save()
    return norm


def remove_exclude(cfg: Config, pattern: str) -> str:
    if pattern not in cfg.excludes:
        raise SettingsError(f"{pattern} is not in the exclude list")
    cfg.excludes = [p for p in cfg.excludes if p != pattern]
    cfg.save()
    return pattern


def update_section(cfg: Config, section: str, patch: dict) -> dict:
    """Apply a validated patch to models/limits/deep. Returns what changed."""
    if not isinstance(patch, dict):
        raise SettingsError("expected an object of field -> value")
    if section == "models":
        target, allowed = cfg.models, {k: str for k in MODEL_FIELDS}
    elif section == "limits":
        target, allowed = cfg.limits, LIMIT_FIELDS
    elif section == "deep":
        target, allowed = cfg.deep, DEEP_FIELDS
    else:
        raise SettingsError(f"unknown settings section {section!r}")

    changes: dict = {}
    for key, raw in patch.items():
        if key not in allowed:
            raise SettingsError(f"{section}.{key} is not editable")
        val = _coerce(key, allowed[key], raw)
        if getattr(target, key) != val:
            changes[key] = val
    if not changes:
        return {}
    for key, val in changes.items():
        setattr(target, key, val)
    cfg.save()
    return changes


def list_directory(raw: str | None) -> dict:
    """Directory listing for the root picker: subdirectories only, never file
    contents. Used to choose a folder to index, so it is deliberately not
    limited to the existing roots — but it returns names, not data."""
    p = Path(raw).expanduser() if raw else Path.home()
    try:
        p = p.resolve()
    except OSError as e:
        raise SettingsError(f"{raw}: {e}") from e
    if not p.is_dir():
        raise SettingsError(f"{p} is not a directory")
    entries = []
    try:
        with os.scandir(p) as it:
            for e in it:
                if e.name.startswith("."):
                    continue
                try:
                    if e.is_dir(follow_symlinks=False):
                        entries.append(e.name)
                except OSError:
                    continue
    except PermissionError as e:
        raise SettingsError(f"{p} is not readable") from e
    entries.sort(key=str.lower)
    return {
        "path": str(p),
        "parent": str(p.parent) if p.parent != p else None,
        "dirs": entries[:1000],
        "truncated": len(entries) > 1000,
    }
