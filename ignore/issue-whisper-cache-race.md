Whisper model cache is not thread-safe and thrashes between concurrent CPU/GPU users

**Labels:** `type:bug`, `priority:p1`  

---

## Problem
`audio._get_model()` caches one model in module globals (`_model`, `_model_key`) with
a check-then-set and no lock (`file_index/extractors/audio.py:21-37`). During a
`deep` run two threads use it concurrently with *different* keys: the background
transcriber thread always requests `("large-v3", "cpu", "int8")`
(`file_index/queue.py:285-289`), while the main thread's inline audio processing
requests the configured `("large-v3", "cuda", "float16")`
(`file_index/queue.py:543-551`).

- **Race:** interleaved calls can leave `_model` from one thread and `_model_key`
  from the other. From then on, callers silently get the wrong device — e.g. the
  "CPU-only" background transcriber running a CUDA model, which is exactly the
  VRAM-grab the design (README, commit ffccb62) exists to prevent: the pinned vision
  model gets pushed into partial CPU offload, slowing every caption ~30x.
- **Thrash:** even without the race, a queue that interleaves audio files and video
  transcripts reloads large-v3 on every key flip (tens of seconds each), and both
  models can be resident at once (CPU RAM + VRAM).

## Proposed Solution
Guard `_get_model` with a `threading.Lock`, and cache per-key (dict keyed by
`(name, device, compute_type)`) instead of a single slot so CPU and CUDA instances
coexist without reloading. Optionally cap to one instance per device. Note
`WhisperModel.transcribe` itself is safe to call from one thread per instance —
per-key caching also ensures the two threads never share an instance.

## Acceptance Criteria
- [ ] Concurrent `transcribe()` calls with different device keys never return a model
      whose device differs from the requested key (test with threads + a fake
      WhisperModel recording constructor args).
- [ ] An interleaved audio/video queue loads each (device, compute_type) model at
      most once per run.
