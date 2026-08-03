extractor_version is stored but nothing ever uses it — the documented re-extraction capability has no mechanism

**Labels:** `type:feature-enhancement`, `priority:p2`  

---

## Problem
The README promises "Extractions are versioned per stage (`content.extractor_version`)
so any stage can be re-run later with a better model without touching the others"
(README.md:131). The version is written on every `store_content` call
(`file_index/index.py:57`), but no code path ever *reads* or compares it: nothing
re-enqueues files whose stored version differs from the current extractor VERSION
constant, and there is no CLI to trigger it. Bumping `image.VERSION` or swapping the
vision model does nothing for already-processed files. Related:
`audio.VERSION = "whisper-large-v3-1.0"` bakes a model name into the constant while
the actual model is config-driven, so a model change isn't even representable in the
version.

## Proposed Solution
Add a `reindex` CLI command (or a `scan --refresh-stage <stage>` flag) that compares
`content.extractor_version` against current constants (and, for model-dependent
stages, the configured model name) and re-enqueues mismatched files at the right tier
with the right queue `kind`. Derive model-dependent versions from config (e.g.
`f"vlm-{model}-1.2"`). Alternatively, drop the claim from the README until built.

## Acceptance Criteria
- [ ] After bumping a stage's version (or changing its model), a documented command
      re-processes exactly the files whose stored version mismatches, leaving other
      stages untouched.
- [ ] README matches the implemented behavior.
