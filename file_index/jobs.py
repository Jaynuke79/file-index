"""Run CLI commands as supervised subprocesses on behalf of the web UI.

Jobs are subprocesses of `file-index` itself rather than in-process calls: they
are isolated from the server, killable, and inherit the exact semantics the
terminal already has — notably `deep`'s two-signal graceful stop, which
checkpoints and unloads models rather than dying mid-file.

Output is kept in a bounded ring buffer per job; the UI polls for the tail.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

MAX_OUTPUT_LINES = 500

# Commands the UI may launch, with the CLI arguments each maps to. Anything not
# listed here is refused — the UI never passes a free-form command line.
RUNNABLE: dict[str, list[str]] = {
    "scan": ["scan"],
    "deep": ["deep"],
    "reindex": ["reindex", "--yes"],
    "purge": ["purge", "--yes"],
}

# Only one job may run at a time: scan and deep both write the index, and deep
# expects sole use of the GPU.
EXCLUSIVE = True


class JobError(RuntimeError):
    pass


@dataclass
class Job:
    id: int
    name: str
    argv: list[str]
    started_at: float
    proc: subprocess.Popen | None = None
    finished_at: float | None = None
    returncode: int | None = None
    stopping: bool = False
    output: deque = field(default_factory=lambda: deque(maxlen=MAX_OUTPUT_LINES))

    @property
    def running(self) -> bool:
        return self.finished_at is None

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "running": self.running,
            "stopping": self.stopping,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "elapsed": round((self.finished_at or time.time()) - self.started_at, 1),
        }

    def detail(self, tail: int = MAX_OUTPUT_LINES) -> dict:
        out = self.summary()
        lines = list(self.output)
        out["output"] = lines[-tail:] if tail else lines
        return out


def _cli_argv(args: list[str]) -> list[str]:
    """Invoke this interpreter's file_index CLI, so a venv install and a source
    checkout both work without depending on `file-index` being on PATH."""
    return [sys.executable, "-m", "file_index.cli", *args]


class JobRunner:
    """Owns the lifecycle of UI-launched jobs. Thread-safe."""

    def __init__(self) -> None:
        self._jobs: dict[int, Job] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    # ---------- queries ----------

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.id, reverse=True)
            return [j.summary() for j in jobs]

    def get(self, job_id: int) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> Job | None:
        with self._lock:
            for j in self._jobs.values():
                if j.running:
                    return j
        return None

    # ---------- lifecycle ----------

    def start(self, name: str, extra: list[str] | None = None) -> Job:
        if name not in RUNNABLE:
            raise JobError(f"unknown job {name!r}")
        if EXCLUSIVE and (busy := self.active()):
            raise JobError(
                f"{busy.name} is already running — stop it before starting another"
            )
        argv = _cli_argv([*RUNNABLE[name], *(extra or [])])
        with self._lock:
            job = Job(id=self._next_id, name=name, argv=argv, started_at=time.time())
            self._next_id += 1
            self._jobs[job.id] = job
        try:
            job.proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                # Own process group so a stop signals the child, not the server.
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "COLUMNS": "100"},
            )
        except OSError as e:
            job.finished_at = time.time()
            job.returncode = -1
            job.output.append(f"failed to start: {e}")
            raise JobError(f"could not start {name}: {e}") from e
        threading.Thread(
            target=self._pump, args=(job,), name=f"job-{job.id}-{name}", daemon=True
        ).start()
        return job

    def _pump(self, job: Job) -> None:
        assert job.proc is not None and job.proc.stdout is not None
        try:
            for line in job.proc.stdout:
                line = line.rstrip("\n")
                if line.strip():
                    job.output.append(line)
        except (OSError, ValueError):  # pipe closed under us
            pass
        finally:
            job.returncode = job.proc.wait()
            job.finished_at = time.time()
            job.output.append(
                f"— finished with exit code {job.returncode} —"
                if job.returncode
                else "— finished —"
            )

    def stop(self, job_id: int, force: bool = False) -> Job:
        job = self.get(job_id)
        if job is None:
            raise JobError(f"no job {job_id}")
        if not job.running or job.proc is None:
            return job
        sig = signal.SIGKILL if force else signal.SIGINT
        try:
            # Signal the whole process group: `deep` treats the first SIGINT as
            # "finish this file and checkpoint", a second as "stop now".
            os.killpg(os.getpgid(job.proc.pid), sig)
        except (ProcessLookupError, PermissionError) as e:
            raise JobError(f"could not signal job {job_id}: {e}") from e
        job.stopping = True
        job.output.append(
            "— sent SIGKILL —" if force else "— stop requested (finishing current file) —"
        )
        return job

    def shutdown(self) -> None:
        """Best-effort stop of anything still running, for server teardown."""
        for summary in self.list():
            if summary["running"]:
                try:
                    self.stop(summary["id"])
                except JobError:
                    pass
