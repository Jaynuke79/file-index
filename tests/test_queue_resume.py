"""Queue resume: simulate a kill mid-run and verify restart picks up exactly
where processing stopped, without redoing completed files."""

from file_index.crawler import Crawler
from file_index.queue import Tier1Worker


def test_tier1_resume_after_kill(tmp_env):
    cfg, index, root = tmp_env
    for i in range(10):
        (root / f"f{i}.txt").write_text(f"unique content number {i}")
    Crawler(cfg, index).crawl()

    # first run: process exactly 4 files, then "die" (stop pulling)
    worker = Tier1Worker(cfg, index, embedder=None)
    processed_first: list[str] = []
    for _ in range(4):
        item = index.next_pending(tier=1)
        worker._process(item, __import__("pathlib").Path(item["path"]))
        index.mark_done(item["id"])
        index.set_tier_status(item["file_id"], 1, "done")
        processed_first.append(item["path"])
        index.commit()  # checkpoint — same as the real loop

    # "restart": new worker over the same db state
    worker2 = Tier1Worker(cfg, index, embedder=None)
    seen: list[str] = []
    worker2.run(progress_cb=lambda p, d, f: seen.append(p))
    # exactly the remaining 6 were processed, none repeated
    assert len(set(seen)) == 6
    assert not (set(seen) & set(processed_first))
    stats = index.queue_stats()
    assert stats["tier1"].get("done") == 10
    assert index.next_pending(tier=1) is None


def test_failed_items_retry_then_give_up(tmp_env):
    cfg, index, root = tmp_env
    f = root / "gone.txt"
    f.write_text("will disappear")
    Crawler(cfg, index).crawl()
    f.unlink()  # file vanishes before processing

    worker = Tier1Worker(cfg, index, embedder=None)
    result = worker.run()
    assert result["failed"] == 1
    assert index.next_pending(tier=1) is None  # not stuck pending


def test_tier2_priority_and_ordering(tmp_env):
    cfg, index, root = tmp_env
    files = {
        "vid.mp4": ("video", 100.0),
        "old.jpg": ("image", 50.0),
        "new.jpg": ("image", 200.0),
    }
    for name, (kind, mtime) in files.items():
        fid = index.upsert_file(str(root / name), f"h-{name}", 1, mtime, "x/y", kind)
        index.enqueue(fid, 2, "pending_deep", kind, mtime)
    index.commit()

    order = []
    while True:
        item = index.next_pending(tier=2, kind_priority=["image", "audio", "video"], newest_first=True)
        if item is None:
            break
        order.append(item["path"].rsplit("/", 1)[-1])
        index.mark_done(item["id"])
    # images before video; among images, newest first
    assert order == ["new.jpg", "old.jpg", "vid.mp4"]
