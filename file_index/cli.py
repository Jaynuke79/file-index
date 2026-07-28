"""file-index CLI: init, scan, deep, search, ask, organize, browse, status, watch."""

from __future__ import annotations

import logging
import shutil
import signal
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)
from rich.table import Table

from . import config as config_mod
from .config import Config, ConfigError, load_config
from .index import Index
from .ollama_client import OllamaClient

app = typer.Typer(help="Fully local filesystem content indexer.", no_args_is_help=True)
console = Console()


def _setup_logging(cfg: Config | None) -> None:
    handlers: list[logging.Handler] = []
    if cfg:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(cfg.log_path)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        handlers.append(fh)
    logging.basicConfig(level=logging.INFO, handlers=handlers or [logging.NullHandler()])


def _load() -> tuple[Config, Index]:
    try:
        cfg = load_config()
    except ConfigError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    _setup_logging(cfg)
    return cfg, Index(cfg.db_path)


@app.command()
def init() -> None:
    """Interactively create config, verify Ollama and models."""
    console.print("[bold]file-index setup[/bold]\n")
    cfg = Config()

    # roots
    roots: list[Path] = []
    console.print("Enter directories to index (whitelist). Empty line to finish.")
    while True:
        raw = typer.prompt("root directory", default="", show_default=False)
        if not raw.strip():
            if roots:
                break
            console.print("[yellow]at least one root is required[/yellow]")
            continue
        p = Path(raw).expanduser()
        if not p.is_dir():
            console.print(f"[yellow]{p} is not a directory[/yellow]")
            continue
        roots.append(p.resolve())
        console.print(f"  added [green]{p.resolve()}[/green]")
    cfg.roots = roots

    if typer.confirm("Add extra exclude patterns beyond the defaults?", default=False):
        console.print(f"defaults: {', '.join(config_mod.DEFAULT_EXCLUDES[:6])} …")
        while True:
            pat = typer.prompt("exclude glob (empty to finish)", default="", show_default=False)
            if not pat.strip():
                break
            cfg.excludes.append(pat.strip())

    # verify environment
    client = OllamaClient(cfg.models.ollama_url)
    if not client.ping():
        console.print(
            f"[red]Ollama is not reachable at {cfg.models.ollama_url}.[/red]\n"
            "Install from https://ollama.com and run `ollama serve`, then re-run "
            "`file-index init`. Config will still be written; tier-1 keyword "
            "indexing works without Ollama."
        )
    else:
        available = client.list_models()
        for role, name in (
            ("agent", cfg.models.agent),
            ("vision", cfg.models.vision),
            ("embeddings", cfg.models.embed),
        ):
            if name in available or f"{name}:latest" in available:
                console.print(f"[green]✓[/green] {role} model {name}")
            else:
                if typer.confirm(f"{role} model '{name}' is missing. Pull it now?", default=True):
                    with Progress(
                        SpinnerColumn(), TextColumn("{task.description}"), console=console
                    ) as prog:
                        task = prog.add_task(f"pulling {name}")
                        client.pull(
                            name,
                            progress_cb=lambda m: prog.update(
                                task, description=f"pulling {name}: {m.get('status', '')}"
                            ),
                        )
                    console.print(f"[green]✓[/green] pulled {name}")
                else:
                    console.print(f"[yellow]skipped {name} — some features will fail[/yellow]")
    if not shutil.which("ffmpeg"):
        console.print("[yellow]ffmpeg not found — video/audio extraction will fail. "
                      "Install with: sudo apt install ffmpeg[/yellow]")

    path = cfg.save()
    console.print(f"\nConfig written to [bold]{path}[/bold]. Next: [bold]file-index scan[/bold]")


@app.command()
def scan() -> None:
    """Tier 1: crawl roots, extract text/metadata, build keyword index."""
    cfg, index = _load()
    from .crawler import Crawler
    from .embed import Embedder
    from .queue import Tier1Worker

    crawler = Crawler(cfg, index)
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("crawling", total=None)

        def crawl_cb(i, total, path):
            prog.update(task, total=total, completed=i,
                        description=f"crawling {Path(path).name[:40]}")

        stats = crawler.crawl(progress_cb=crawl_cb)
    console.print(
        f"crawl: {stats.scanned} scanned, [green]{stats.new} new[/green], "
        f"{stats.modified} modified, {stats.moved} moved, {stats.unchanged} unchanged, "
        f"{stats.removed} removed, [red]{stats.errors} errors[/red]"
    )

    worker = Tier1Worker(cfg, index, Embedder(cfg))
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("extracting")

        def t1_cb(path, done, failed):
            prog.update(task, description=f"extracting [{done} done] {Path(path).name[:40]}")

        result = worker.run(progress_cb=t1_cb)
    index.commit()
    console.print(
        f"tier 1: [green]{result['done']} extracted[/green], [red]{result['failed']} failed[/red]. "
        f"Keyword search is ready: [bold]file-index search \"query\"[/bold]"
    )


@app.command()
def deep() -> None:
    """Tier 2: VLM image analysis, scanned PDFs, Whisper, video pipeline."""
    cfg, index = _load()
    from .queue import Tier2Worker

    worker = Tier2Worker(cfg, index)
    total = worker.pending_count()
    if not total:
        console.print("tier-2 queue is empty — nothing to do")
        return
    console.print(f"{total} files queued for deep processing (ctrl-c safe: resumes where it left off)")

    stop = {"flag": False}

    def on_stop_signal(sig, frame):
        if stop["flag"]:
            raise SystemExit(130)  # second signal: stop now, skip current file
        stop["flag"] = True
        console.print(
            "\n[yellow]finishing current file, then stopping… (signal again to stop now)[/yellow]"
        )

    signal.signal(signal.SIGINT, on_stop_signal)
    signal.signal(signal.SIGTERM, on_stop_signal)

    result = None
    try:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
            TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("deep", total=total)

            def cb(path, done, remaining, eta):
                eta_s = f" ETA {int(eta // 60)}m{int(eta % 60):02d}s" if eta else ""
                prog.update(task, completed=done, total=done + remaining,
                            description=f"{Path(path).name[:36]}{eta_s}")

            try:
                result = worker.run(progress_cb=cb, stop_check=lambda: stop["flag"])
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]{e}[/red]")
                raise typer.Exit(1)
    finally:
        # Don't leave a ~20 GB model squatting in VRAM after we stop.
        n = OllamaClient(cfg.models.ollama_url).unload_all()
        if n:
            console.print(f"[dim]freed GPU memory ({n} model(s) unloaded)[/dim]")
    if result is not None:
        console.print(
            f"tier 2: [green]{result['done']} processed[/green], [red]{result['failed']} failed[/red], "
            f"{worker.pending_count()} remaining"
        )


@app.command()
def search(
    query: str,
    limit: int = typer.Option(15, "--limit", "-n"),
) -> None:
    """Hybrid keyword + semantic search over indexed content."""
    cfg, index = _load()
    from .search import format_ts, hybrid_search

    hits = hybrid_search(cfg, index, query, limit=limit)
    if not hits:
        console.print("no results")
        return
    for h in hits:
        ts = ""
        if h.ts_start is not None:
            ts = f" [cyan]@ {format_ts(h.ts_start)}–{format_ts(h.ts_end)}[/cyan]"
        console.print(f"[bold]{h.path}[/bold]{ts} [dim]({h.stage})[/dim]")
        snippet = " ".join(h.snippet.split())[:220]
        console.print(f"  {snippet}\n")


@app.command()
def ask(question: str) -> None:
    """Ask the local agent a question about your files."""
    cfg, index = _load()
    from . import agent as agent_mod

    def on_tool(name, args):
        console.print(f"[dim]→ {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})[/dim]")

    try:
        with console.status("thinking…"):
            answer = agent_mod.ask(cfg, index, question, on_tool=on_tool)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(answer)


@app.command()
def organize(
    directory: Path,
    apply: bool = typer.Option(False, "--apply", help="Apply moves/renames after per-plan confirmation."),
) -> None:
    """Propose a reorganization of DIRECTORY (read-only unless --apply)."""
    cfg, index = _load()
    from . import agent as agent_mod

    directory = directory.expanduser().resolve()
    try:
        with console.status("analyzing directory…"):
            plan = agent_mod.propose_organization(cfg, index, directory)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Plan for {directory}[/bold]\n\n{plan.summary}\n")
    table = Table(show_lines=False)
    table.add_column("action")
    table.add_column("src / target")
    table.add_column("reason", overflow="fold")
    for a in plan.actions:
        if a.action == "mkdir":
            table.add_row("mkdir", a.dst or "", a.reason)
        elif a.action == "delete_candidate":
            table.add_row("[yellow]delete candidate[/yellow]", a.src or "", a.reason + " (never auto-deleted)")
        else:
            table.add_row(a.action, f"{a.src}\n→ {a.dst}", a.reason)
    console.print(table)

    doable = [a for a in plan.actions if a.action in ("move", "rename", "mkdir")]
    if not apply:
        console.print("\n[dim]Read-only: nothing was changed. Re-run with --apply to execute moves/renames.[/dim]")
        return
    if not doable:
        console.print("nothing applicable to apply")
        return
    if not typer.confirm(f"Apply {len(doable)} operations (moves/renames/mkdirs only)?", default=False):
        console.print("aborted — nothing changed")
        return

    applied = 0
    for a in doable:
        try:
            if a.action == "mkdir":
                dst = Path(a.dst)
                _assert_inside(cfg, dst)
                dst.mkdir(parents=True, exist_ok=True)
                index.audit("mkdir", None, str(dst), a.reason)
            else:
                src, dst = Path(a.src), Path(a.dst)
                _assert_inside(cfg, src)
                _assert_inside(cfg, dst)
                if dst.exists():
                    console.print(f"[yellow]skip (target exists): {dst}[/yellow]")
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                index.audit(a.action, str(src), str(dst), a.reason)
                row = index.get_file_by_path(str(src))
                if row:
                    index.move_file(row["id"], str(dst), dst.stat().st_mtime)
                    index.commit()
            applied += 1
        except (OSError, PermissionError) as e:
            console.print(f"[red]failed: {a.action} {a.src} → {a.dst}: {e}[/red]")
    console.print(f"[green]applied {applied}/{len(doable)} operations[/green] (audit log: {cfg.audit_log_path})")
    _append_audit_file(cfg, index)


def _assert_inside(cfg: Config, p: Path) -> None:
    if not cfg.is_within_roots(p.parent if not p.exists() else p):
        raise PermissionError(f"{p} is outside the whitelisted roots")


def _append_audit_file(cfg: Config, index: Index) -> None:
    """Mirror the DB audit log to a plain-text file for easy inspection."""
    import time as _time

    with open(cfg.audit_log_path, "a") as f:
        for row in index.db.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT 200"
        ).fetchall()[::-1]:
            ts = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(row["ts"]))
            f.write(f"{ts} {row['op']} {row['before_path'] or '-'} -> {row['after_path'] or '-'} ({row['detail']})\n")


@app.command()
def exclude(
    pattern: str = typer.Argument(..., help="Directory, file, or glob to exclude"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Exclude files from the index and future scans (index-only — files on
    disk are never touched). Reversible: remove the pattern from config.yaml
    and re-run scan."""
    import fnmatch as _fnmatch

    cfg, index = _load()
    from .crawler import apply_exclude, exclude_pattern

    norm = exclude_pattern(pattern)
    n = sum(
        1 for r in index.db.execute("SELECT path FROM files WHERE deleted=0")
        if _fnmatch.fnmatch(r["path"], norm)
    )
    if n and not yes and not typer.confirm(
        f"Remove {n} indexed files matching {norm}? (files on disk are untouched)",
        default=False,
    ):
        console.print("aborted — nothing changed")
        return
    norm, removed, pending = apply_exclude(cfg, index, pattern)
    console.print(
        f"excluded [bold]{norm}[/bold]: [green]{removed} removed from index[/green] "
        f"({pending} skipped from the deep queue); future scans will skip it"
    )


@app.command()
def watch() -> None:
    """Watch roots for changes and index them incrementally (daemon)."""
    cfg, index = _load()
    from .embed import Embedder
    from .queue import Tier1Worker
    from .watch import Watcher

    watcher = Watcher(cfg, index)
    signal.signal(signal.SIGTERM, lambda s, f: watcher.stop())
    signal.signal(signal.SIGINT, lambda s, f: watcher.stop())
    console.print(f"watching: {', '.join(str(r) for r in cfg.roots)} (ctrl-c to stop)")

    worker = Tier1Worker(cfg, index, Embedder(cfg))

    def on_event(kind, path):
        console.print(f"[dim]{kind}[/dim] {path}")
        if kind == "indexed":
            worker.run()  # drain tier-1 immediately; tier-2 waits for `deep`

    watcher.run(on_event=on_event)
    console.print("stopped")


@app.command()
def browse(
    host: str = typer.Option("127.0.0.1", help="Bind address (localhost only by default)."),
    port: int = typer.Option(8765, help="Port for the web UI."),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the UI in the default browser."
    ),
) -> None:
    """Serve a local web gallery of indexed files and their captions."""
    cfg, _index = _load()
    if not cfg.db_path.exists():
        console.print("[red]No index found — run `file-index scan` first.[/red]")
        raise typer.Exit(1)
    from .web import serve

    console.print(f"browse UI at [bold]http://{host}:{port}/[/bold] (ctrl-c to stop)")
    serve(cfg, host=host, port=port, open_browser=open_browser)


@app.command()
def status() -> None:
    """Queue stats, index size, per-type counts, recent failures."""
    cfg, index = _load()
    stats = index.queue_stats()

    table = Table(title="queue")
    table.add_column("tier")
    for col in ("pending", "done", "failed"):
        table.add_column(col, justify="right")
    t1, t2 = stats["tier1"], stats["tier2"]
    table.add_row("1 (metadata)", str(t1.get("pending_metadata", 0)), str(t1.get("done", 0)), str(t1.get("failed", 0)))
    table.add_row(
        "2 (deep)",
        str(t2.get("pending_deep", 0) + t2.get("pending_summary", 0)),
        str(t2.get("done", 0)),
        str(t2.get("failed", 0)),
    )
    console.print(table)

    kinds = Table(title="files by type")
    kinds.add_column("kind")
    kinds.add_column("count", justify="right")
    kinds.add_column("total size", justify="right")
    for row in index.db.execute(
        "SELECT kind, COUNT(*) n, SUM(size) s FROM files WHERE deleted=0 GROUP BY kind ORDER BY n DESC"
    ):
        kinds.add_row(row["kind"] or "?", str(row["n"]), _human(row["s"] or 0))
    console.print(kinds)

    n_files = index.db.execute("SELECT COUNT(*) n FROM files WHERE deleted=0").fetchone()["n"]
    n_chunks = index.db.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
    n_emb = index.db.execute("SELECT COUNT(*) n FROM chunks WHERE embedding IS NOT NULL").fetchone()["n"]
    db_size = cfg.db_path.stat().st_size if cfg.db_path.exists() else 0
    console.print(
        f"{n_files} files, {n_chunks} chunks ({n_emb} embedded), index size {_human(db_size)}"
    )

    fails = index.failures(20)
    if fails:
        ft = Table(title="recent failures")
        ft.add_column("tier")
        ft.add_column("path", overflow="fold")
        ft.add_column("error", overflow="fold")
        for f in fails:
            ft.add_row(str(f["tier"]), f["path"], (f["error"] or "")[:120])
        console.print(ft)


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


if __name__ == "__main__":
    app()
