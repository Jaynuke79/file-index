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

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        r = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": texts},
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
