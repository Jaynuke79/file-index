"""Thin HTTP client for a local Ollama server. No cloud calls anywhere."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import requests

log = logging.getLogger("file_index.ollama")


class OllamaError(Exception):
    pass


class OllamaNotRunning(OllamaError):
    pass


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # When set (e.g. -1 during a deep run), sent with every request so
        # Ollama does not idle-evict the model mid-run — a long CPU stretch
        # (Whisper, scene detection) must not cost a ~20 GB model reload.
        # Callers that set this are responsible for unloading at the end.
        self.keep_alive: int | str | None = None

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/version", timeout=5)
            return r.ok
        except requests.RequestException:
            return False

    def require(self) -> None:
        if not self.ping():
            raise OllamaNotRunning(
                f"Ollama is not reachable at {self.base_url}. "
                "Install it from https://ollama.com and start it (`ollama serve`), "
                "or fix models.ollama_url in config.yaml."
            )

    def list_models(self) -> list[str]:
        r = requests.get(f"{self.base_url}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def has_model(self, name: str) -> bool:
        models = self.list_models()
        return name in models or f"{name}:latest" in models

    def pull(self, name: str, progress_cb=None) -> None:
        with requests.post(
            f"{self.base_url}/api/pull", json={"model": name}, stream=True, timeout=None
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    raise OllamaError(f"pull {name}: {msg['error']}")
                if progress_cb:
                    progress_cb(msg)

    def loaded_models(self) -> list[str]:
        """Models currently resident in memory (GPU or CPU)."""
        r = requests.get(f"{self.base_url}/api/ps", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def unload(self, model: str) -> None:
        """Ask Ollama to evict the model now (keep_alive=0) instead of after
        its idle timeout. Waits for any in-flight request on that model.
        """
        # /api/generate unloads generation models; embedding-only models
        # reject generate, so fall back to /api/embed.
        for endpoint, payload in (
            ("generate", {"model": model, "keep_alive": 0}),
            ("embed", {"model": model, "input": [], "keep_alive": 0}),
        ):
            r = requests.post(
                f"{self.base_url}/api/{endpoint}", json=payload, timeout=self.timeout
            )
            if r.ok:
                return

    def unload_all(self) -> int:
        """Best-effort eviction of every loaded model. Returns count evicted."""
        try:
            models = self.loaded_models()
            for m in models:
                self.unload(m)
            return len(models)
        except requests.RequestException as e:
            log.debug("unload_all: %s", e)
            return 0

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        payload: dict = {"model": model, "input": texts}
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        r = requests.post(
            f"{self.base_url}/api/embed",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        if "embeddings" not in data:
            raise OllamaError(f"unexpected embed response: {data}")
        return data["embeddings"]

    def generate(
        self,
        model: str,
        prompt: str,
        images: list[Path] | None = None,
        format_json: bool = False,
        options: dict | None = None,
    ) -> str:
        payload: dict = {"model": model, "prompt": prompt, "stream": False}
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if images:
            payload["images"] = [
                base64.b64encode(p.read_bytes()).decode() for p in images
            ]
        if format_json:
            payload["format"] = "json"
        if options:
            payload["options"] = options
        r = requests.post(
            f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json().get("response", "")

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        options: dict | None = None,
    ) -> dict:
        """Returns the response `message` dict (may contain tool_calls)."""
        payload: dict = {"model": model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options
        r = requests.post(
            f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json().get("message", {})
