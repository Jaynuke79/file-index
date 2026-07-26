import os

from file_index.crawler import Crawler


def crawl(cfg, index):
    return Crawler(cfg, index).crawl()


def test_new_files_enqueued(tmp_env):
    cfg, index, root = tmp_env
    (root / "a.txt").write_text("hello world")
    (root / "b.py").write_text("print('hi')")
    stats = crawl(cfg, index)
    assert stats.new == 2
    assert index.next_pending(tier=1) is not None


def test_unchanged_files_skipped(tmp_env):
    cfg, index, root = tmp_env
    f = root / "a.txt"
    f.write_text("hello world")
    crawl(cfg, index)
    stats = crawl(cfg, index)
    assert stats.new == 0
    assert stats.unchanged == 1
    assert stats.modified == 0


def test_modified_file_requeued(tmp_env):
    cfg, index, root = tmp_env
    f = root / "a.txt"
    f.write_text("hello world")
    crawl(cfg, index)
    # drain queue so we can see it re-enqueued
    item = index.next_pending(tier=1)
    index.mark_done(item["id"])
    index.commit()

    f.write_text("completely different content")
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 10))
    stats = crawl(cfg, index)
    assert stats.modified == 1
    assert index.next_pending(tier=1) is not None


def test_move_detected_without_reextraction(tmp_env):
    cfg, index, root = tmp_env
    f = root / "a.txt"
    f.write_text("some unique content for moving")
    crawl(cfg, index)
    item = index.next_pending(tier=1)
    index.mark_done(item["id"])
    index.store_content(item["file_id"], "text", "text-1.0", "some unique content for moving")
    index.commit()

    new = root / "sub" / "renamed.txt"
    new.parent.mkdir()
    f.rename(new)
    stats = crawl(cfg, index)
    assert stats.moved == 1
    assert stats.new == 0
    row = index.get_file_by_path(str(new.resolve()))
    assert row is not None
    # extraction survived the move, and no new tier-1 work was queued
    assert index.get_content(row["id"], "text")
    assert index.next_pending(tier=1) is None


def test_excludes_respected(tmp_env):
    cfg, index, root = tmp_env
    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.js").write_text("ignored")
    (root / "keep.txt").write_text("kept")
    stats = crawl(cfg, index)
    assert stats.new == 1
    assert index.get_file_by_path(str((root / "node_modules" / "x.js").resolve())) is None


def test_deleted_file_marked(tmp_env):
    cfg, index, root = tmp_env
    f = root / "a.txt"
    f.write_text("hello")
    crawl(cfg, index)
    f.unlink()
    stats = crawl(cfg, index)
    assert stats.removed == 1
    row = index.db.execute("SELECT deleted FROM files").fetchone()
    assert row["deleted"] == 1


def test_symlinks_ignored(tmp_env):
    cfg, index, root = tmp_env
    (root / "real.txt").write_text("real")
    (root / "link.txt").symlink_to(root / "real.txt")
    (root / "broken.txt").symlink_to(root / "nope.txt")
    stats = crawl(cfg, index)
    assert stats.new == 1
    assert stats.errors == 0


def test_exclude_removes_and_stays_gone(tmp_env):
    from file_index.crawler import apply_exclude

    cfg, index, root = tmp_env
    keep = root / "keep.txt"
    keep.write_text("keep me")
    sub = root / "big_videos"
    sub.mkdir()
    (sub / "huge.txt").write_text("slow to process")
    Crawler(cfg, index).crawl()

    pattern, removed, _ = apply_exclude(cfg, index, str(sub))
    assert removed == 1
    assert pattern in cfg.excludes
    row = index.get_file_by_path(str((sub / "huge.txt").resolve()))
    assert row["deleted"] == 1
    # queue no longer serves it
    pending = [
        index.next_pending(tier=1) for _ in range(5)
    ]
    assert all(p is None or "big_videos" not in p["path"] for p in pending)
    # re-crawl honors the new exclude: file stays out
    Crawler(cfg, index).crawl()
    row = index.get_file_by_path(str((sub / "huge.txt").resolve()))
    assert row["deleted"] == 1
    assert index.get_file_by_path(str(keep.resolve()))["deleted"] == 0


def test_exclude_escapes_glob_brackets(tmp_env):
    from file_index.crawler import apply_exclude

    cfg, index, root = tmp_env
    sub = root / "Show.S01.COMPLETE[TGx]"
    sub.mkdir()
    (sub / "e01.txt").write_text("episode one")
    Crawler(cfg, index).crawl()

    _, removed, _ = apply_exclude(cfg, index, str(sub))
    assert removed == 1
    Crawler(cfg, index).crawl()
    assert index.get_file_by_path(str((sub / "e01.txt").resolve()))["deleted"] == 1
