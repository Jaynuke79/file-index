"""Configuration loading and validation.

All tunables — model names, roots, excludes, caps — live in config.yaml so they
can be swapped without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_DIR = Path(
    os.environ.get("FILE_INDEX_CONFIG_DIR", "~/.config/file-index")
).expanduser()
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"
DEFAULT_DATA_DIR = Path(
    os.environ.get("FILE_INDEX_DATA_DIR", "~/.local/share/file-index")
).expanduser()

DEFAULT_EXCLUDES = [
    "**/node_modules/**",
    "**/.git/**",
    "**/venv/**",
    "**/.venv/**",
    "**/__pycache__/**",
    "**/.cache/**",
    "**/cache/**",
    "**/.tox/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/dist/**",
    "**/build/**",
    "**/*.pyc",
]


@dataclass
class ModelsConfig:
    agent: str = "qwen3:30b-a3b"
    vision: str = "qwen2.5vl:32b"
    embed: str = "nomic-embed-text"
    whisper: str = "large-v3"  # faster-whisper model name, not Ollama
    ollama_url: str = "http://localhost:11434"


@dataclass
class LimitsConfig:
    text_size_cap: int = 1_000_000  # bytes of extracted text stored per file
    chunk_tokens: int = 500
    chunk_overlap_tokens: int = 50
    max_retries: int = 3
    hash_algo: str = "blake3"  # or "sha256"
    max_file_size: int = 2_000_000_000  # skip hashing files above this (2 GB)


@dataclass
class DeepConfig:
    # Priority order for tier-2 work; earlier = processed first.
    priority: list[str] = field(default_factory=lambda: ["image", "audio", "video"])
    newest_first: bool = True
    video_frames_per_scene: int = 2
    video_fallback_interval_s: float = 10.0
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    # CPU work (image transcode/downscale, video scene detection) for upcoming
    # queue items runs in background threads while the GPU processes the
    # current file. 0 disables prefetching.
    prefetch_files: int = 4
    # Parallel ffmpeg frame/audio extractions per video while the VLM captions.
    video_frame_workers: int = 4
    # Videos with more detected scenes than this are sampled down to this many,
    # evenly spread across the timeline. 0 = caption every scene.
    video_max_scenes: int = 40
    # Skip VLM captioning of frames that are near-duplicates (perceptual hash)
    # of frames already captioned in the same video — common in gameplay and
    # screen recordings.
    video_dedup_frames: bool = True
    # Hamming distance between frame perceptual hashes below which a frame is
    # considered a near-duplicate of one already captioned. Higher = more
    # aggressive skipping (cheaper, coarser); 0 skips only identical frames.
    video_dedup_distance: int = 6
    # When deep-processing a video, feed the captioner/summarizer the summaries
    # of up to this many already-indexed videos from the same folder, so a
    # folder of similar clips (same game, same people) is understood as such.
    # 0 disables.
    neighbor_context: int = 3
    # Defer summaries to an end-of-run sweep: all captioning runs with the
    # vision model resident, then all summaries run with the agent model
    # loaded once — instead of a ~15 s vision<->agent VRAM swap per file.
    # Also keeps Whisper on the background CPU thread. Applies to audio as
    # well as video (the name predates audio joining the same pipeline);
    # disabling it restores fully inline per-file processing.
    defer_video_summaries: bool = True
    # Concurrent VLM caption requests per video. Values > 1 only pay off when
    # the Ollama server allows parallel requests (OLLAMA_NUM_PARALLEL >= this);
    # otherwise the extra requests just queue server-side, which is harmless.
    video_caption_workers: int = 1
    # Background Whisper transcriptions running at once. Each active job uses
    # a few CPU cores; the model weights are shared between them.
    transcript_workers: int = 2
    # Concurrent agent-model summary generations in the end-of-run sweep.
    # Same OLLAMA_NUM_PARALLEL caveat as video_caption_workers.
    summary_workers: int = 1


@dataclass
class Config:
    roots: list[Path] = field(default_factory=list)
    excludes: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    models: ModelsConfig = field(default_factory=ModelsConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    deep: DeepConfig = field(default_factory=DeepConfig)
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    # Where save() writes. load_config() sets it to the file it read, so a
    # Config constructed in tests can never clobber the user's real config.
    config_path: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.db"

    @property
    def audit_log_path(self) -> Path:
        return self.data_dir / "audit.log"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "file-index.log"

    def is_within_roots(self, path: Path) -> bool:
        """Hard safety check: is `path` inside a whitelisted root?"""
        try:
            resolved = path.resolve()
        except OSError:
            return False
        for root in self.roots:
            try:
                resolved.relative_to(root.resolve())
                return True
            except ValueError:
                continue
        return False

    def to_dict(self) -> dict:
        return {
            "roots": [str(r) for r in self.roots],
            "excludes": self.excludes,
            "data_dir": str(self.data_dir),
            "models": {
                "agent": self.models.agent,
                "vision": self.models.vision,
                "embed": self.models.embed,
                "whisper": self.models.whisper,
                "ollama_url": self.models.ollama_url,
            },
            "limits": {
                "text_size_cap": self.limits.text_size_cap,
                "chunk_tokens": self.limits.chunk_tokens,
                "chunk_overlap_tokens": self.limits.chunk_overlap_tokens,
                "max_retries": self.limits.max_retries,
                "hash_algo": self.limits.hash_algo,
                "max_file_size": self.limits.max_file_size,
            },
            "deep": {
                "priority": self.deep.priority,
                "newest_first": self.deep.newest_first,
                "video_frames_per_scene": self.deep.video_frames_per_scene,
                "video_fallback_interval_s": self.deep.video_fallback_interval_s,
                "whisper_device": self.deep.whisper_device,
                "whisper_compute_type": self.deep.whisper_compute_type,
                "prefetch_files": self.deep.prefetch_files,
                "video_frame_workers": self.deep.video_frame_workers,
                "video_max_scenes": self.deep.video_max_scenes,
                "video_dedup_frames": self.deep.video_dedup_frames,
                "video_dedup_distance": self.deep.video_dedup_distance,
                "neighbor_context": self.deep.neighbor_context,
                "defer_video_summaries": self.deep.defer_video_summaries,
            },
        }

    def save(self, path: Path | None = None) -> Path:
        path = path or self.config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        return path


class ConfigError(Exception):
    pass


def load_config(path: Path | None = None) -> Config:
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(
            f"No config found at {path}. Run `file-index init` first."
        )
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()
    cfg.config_path = path
    cfg.roots = [Path(r).expanduser() for r in raw.get("roots", [])]
    if not cfg.roots:
        raise ConfigError(f"Config at {path} has no whitelisted roots.")
    for r in cfg.roots:
        if not r.is_dir():
            raise ConfigError(f"Whitelisted root does not exist or is not a directory: {r}")
    cfg.excludes = raw.get("excludes", list(DEFAULT_EXCLUDES))
    cfg.data_dir = Path(raw.get("data_dir", str(DEFAULT_DATA_DIR))).expanduser()

    m = raw.get("models", {})
    cfg.models = ModelsConfig(
        agent=m.get("agent", cfg.models.agent),
        vision=m.get("vision", cfg.models.vision),
        embed=m.get("embed", cfg.models.embed),
        whisper=m.get("whisper", cfg.models.whisper),
        ollama_url=m.get("ollama_url", cfg.models.ollama_url),
    )
    li = raw.get("limits", {})
    cfg.limits = LimitsConfig(
        text_size_cap=int(li.get("text_size_cap", cfg.limits.text_size_cap)),
        chunk_tokens=int(li.get("chunk_tokens", cfg.limits.chunk_tokens)),
        chunk_overlap_tokens=int(li.get("chunk_overlap_tokens", cfg.limits.chunk_overlap_tokens)),
        max_retries=int(li.get("max_retries", cfg.limits.max_retries)),
        hash_algo=li.get("hash_algo", cfg.limits.hash_algo),
        max_file_size=int(li.get("max_file_size", cfg.limits.max_file_size)),
    )
    d = raw.get("deep", {})
    cfg.deep = DeepConfig(
        priority=d.get("priority", cfg.deep.priority),
        newest_first=bool(d.get("newest_first", cfg.deep.newest_first)),
        video_frames_per_scene=int(d.get("video_frames_per_scene", cfg.deep.video_frames_per_scene)),
        video_fallback_interval_s=float(d.get("video_fallback_interval_s", cfg.deep.video_fallback_interval_s)),
        whisper_device=d.get("whisper_device", cfg.deep.whisper_device),
        whisper_compute_type=d.get("whisper_compute_type", cfg.deep.whisper_compute_type),
        prefetch_files=int(d.get("prefetch_files", cfg.deep.prefetch_files)),
        video_frame_workers=int(d.get("video_frame_workers", cfg.deep.video_frame_workers)),
        video_max_scenes=int(d.get("video_max_scenes", cfg.deep.video_max_scenes)),
        video_dedup_frames=bool(d.get("video_dedup_frames", cfg.deep.video_dedup_frames)),
        video_dedup_distance=int(d.get("video_dedup_distance", cfg.deep.video_dedup_distance)),
        neighbor_context=int(d.get("neighbor_context", cfg.deep.neighbor_context)),
        defer_video_summaries=bool(d.get("defer_video_summaries", cfg.deep.defer_video_summaries)),
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg
