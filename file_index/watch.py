"""Watchdog daemon: monitors whitelisted roots, debounces bursts of changes,
and enqueues tier-1 (and tier-2 where applicable) work for changed files.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import Config
from .crawler import Crawler
from .index import Index

log = logging.getLogger("file_index.watch")

DEBOUNCE_SECONDS = 2.0


class _Handler(FileSystemEventHandler):
    def __init__(self, pending: dict[str, float], lock: threading.Lock, crawler: Crawler):
        self.pending = pending
        self.lock = lock
        self.crawler = crawler

    def _note(self, path_s: str) -> None:
        p = Path(path_s)
        if self.crawler._excluded(p):
            return
        with self.lock:
            self.pending[path_s] = time.time()

    def on_created(self, event):
        if not event.is_directory:
            self._note(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._note(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._note(event.dest_path)
            self._note(event.src_path)  # old path: will be detected as missing

    def on_deleted(self, event):
        if not event.is_directory:
            self._note(event.src_path)


class Watcher:
    def __init__(self, config: Config, index: Index):
        self.config = config
        self.index = index
        self.crawler = Crawler(config, index)
        self.pending: dict[str, float] = {}
        self.lock = threading.Lock()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self, on_event=None) -> None:
        observer = Observer()
        handler = _Handler(self.pending, self.lock, self.crawler)
        for root in self.config.roots:
            observer.schedule(handler, str(root), recursive=True)
        observer.start()
        log.info("watching %d roots", len(self.config.roots))
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
                self._flush(on_event)
        finally:
            observer.stop()
            observer.join()

    def _flush(self, on_event=None) -> None:
        """Process paths whose last event is older than the debounce window."""
        now = time.time()
        ready: list[str] = []
        with self.lock:
            for path_s, ts in list(self.pending.items()):
                if now - ts >= DEBOUNCE_SECONDS:
                    ready.append(path_s)
                    del self.pending[path_s]
        for path_s in ready:
            p = Path(path_s)
            try:
                if p.is_file():
                    from .crawler import CrawlStats

                    stats = CrawlStats()
                    self.crawler._process_one(p, stats, set())
                    self.index.commit()
                    if on_event and (stats.new or stats.modified or stats.moved):
                        on_event("indexed", path_s)
                else:
                    row = self.index.get_file_by_path(str(p.resolve()))
                    # resolve() on a deleted path still yields the absolute path;
                    # fall back to the raw string if it was already resolved
                    if row is None:
                        row = self.index.get_file_by_path(path_s)
                    if row and not Path(row["path"]).exists():
                        self.index.mark_deleted(row["id"])
                        self.index.commit()
                        if on_event:
                            on_event("removed", path_s)
            except Exception as e:  # noqa: BLE001 — one bad path must never
                # kill the daemon (sqlite lock contention with a concurrent
                # scan/deep run, extractor errors via on_event, …)
                log.warning("watch: error handling %s: %s", path_s, e)
