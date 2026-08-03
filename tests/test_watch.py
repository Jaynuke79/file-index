"""Watcher robustness: one bad path must never take down the daemon loop."""

import time

from file_index.watch import DEBOUNCE_SECONDS, Watcher


def _ready(watcher, path_s):
    """Register a path whose debounce window has already expired."""
    watcher.pending[path_s] = time.time() - DEBOUNCE_SECONDS - 1


def test_flush_survives_unexpected_exception(tmp_env, monkeypatch):
    cfg, index, root = tmp_env
    bad = root / "bad.txt"
    good = root / "good.txt"
    bad.write_text("boom")
    good.write_text("fine")

    watcher = Watcher(cfg, index)

    real = watcher.crawler._process_one

    def explode_on_bad(p, stats, seen):
        if p == bad:
            raise ValueError("not an OSError")  # e.g. sqlite lock, extractor bug
        return real(p, stats, seen)

    monkeypatch.setattr(watcher.crawler, "_process_one", explode_on_bad)
    _ready(watcher, str(bad))
    _ready(watcher, str(good))

    watcher._flush()  # must not raise

    assert watcher.pending == {}  # both consumed, loop continues
    assert index.get_file_by_path(str(good.resolve())) is not None


def test_flush_respects_debounce_window(tmp_env):
    cfg, index, root = tmp_env
    f = root / "fresh.txt"
    f.write_text("just changed")
    watcher = Watcher(cfg, index)
    watcher.pending[str(f)] = time.time()  # too recent

    watcher._flush()

    assert str(f) in watcher.pending  # still waiting
    assert index.get_file_by_path(str(f.resolve())) is None


def test_flush_marks_vanished_file_deleted(tmp_env):
    cfg, index, root = tmp_env
    f = root / "gone.txt"
    f.write_text("here for now")
    watcher = Watcher(cfg, index)
    _ready(watcher, str(f))
    watcher._flush()
    fid = index.get_file_by_path(str(f.resolve()))["id"]

    f.unlink()
    _ready(watcher, str(f))
    events = []
    watcher._flush(on_event=lambda kind, p: events.append((kind, p)))

    assert index.db.execute(
        "SELECT deleted FROM files WHERE id=?", (fid,)
    ).fetchone()["deleted"] == 1
    assert events and events[0][0] == "removed"


def test_excluded_paths_are_never_queued(tmp_env):
    cfg, index, root = tmp_env
    cache = root / "__pycache__"
    cache.mkdir()
    junk = cache / "mod.pyc"
    junk.write_text("bytecode")
    watcher = Watcher(cfg, index)

    class _Ev:
        is_directory = False
        src_path = str(junk)

    from file_index.watch import _Handler

    h = _Handler(watcher.pending, watcher.lock, watcher.crawler)
    h.on_created(_Ev())
    assert watcher.pending == {}  # filtered before it ever reaches the queue

    # a non-excluded sibling still gets through
    ok = root / "keep.txt"
    ok.write_text("x")
    _Ev.src_path = str(ok)
    h.on_created(_Ev())
    assert str(ok) in watcher.pending
