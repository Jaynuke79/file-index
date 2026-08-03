"""The Whisper model cache is per-key and safe under concurrent access.

Regression for the torn-cache race: the old single-slot cache let the
background CPU transcriber and the inline CUDA path overwrite each other's
model/key, silently handing one thread a model on the wrong device — and
reloaded large-v3 on every device flip.
"""

import sys
import threading
import types

import pytest

from file_index.extractors import audio


class FakeWhisperModel:
    instances: list["FakeWhisperModel"] = []
    lock = threading.Lock()

    def __init__(self, name, device=None, compute_type=None):
        import time

        time.sleep(0.01)  # widen the check-then-set window
        self.key = (name, device, compute_type)
        with FakeWhisperModel.lock:
            FakeWhisperModel.instances.append(self)


@pytest.fixture
def fake_whisper(monkeypatch):
    FakeWhisperModel.instances = []
    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    )
    monkeypatch.setattr(audio, "_models", {})
    return FakeWhisperModel


def test_concurrent_mixed_keys_load_once_each_and_match(fake_whisper):
    results: dict[int, list] = {}
    keys = [
        ("large-v3", "cpu", "int8"),
        ("large-v3", "cuda", "float16"),
    ]

    def worker(tid):
        got = []
        for i in range(10):
            k = keys[(tid + i) % 2]
            got.append((k, audio._get_model(*k)))
        results[tid] = got

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # exactly one instance per key, ever
    assert len(fake_whisper.instances) == 2
    # every caller got the model matching the key it asked for
    for got in results.values():
        for key, model in got:
            assert model.key == key


def test_cuda_failure_falls_back_to_cpu_and_is_cached(monkeypatch):
    calls = []

    class Flaky:
        def __init__(self, name, device=None, compute_type=None):
            calls.append((name, device, compute_type))
            if device == "cuda":
                raise RuntimeError("no CUDA")
            self.key = (name, device, compute_type)

    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Flaky)
    )
    monkeypatch.setattr(audio, "_models", {})

    m1 = audio._get_model("large-v3", "cuda", "float16")
    m2 = audio._get_model("large-v3", "cuda", "float16")
    assert m1 is m2  # fallback cached under the requested key: no CUDA retry
    assert calls == [
        ("large-v3", "cuda", "float16"),
        ("large-v3", "cpu", "int8"),
    ]
