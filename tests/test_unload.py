"""Unload: deep must evict Ollama models on exit so VRAM is freed."""

from unittest.mock import MagicMock, patch

from file_index.ollama_client import OllamaClient


def _resp(ok=True, json_data=None):
    r = MagicMock()
    r.ok = ok
    r.json.return_value = json_data or {}
    return r


def test_unload_all_evicts_each_loaded_model():
    client = OllamaClient()
    with patch("file_index.ollama_client.requests") as req:
        req.get.return_value = _resp(
            json_data={"models": [{"name": "qwen2.5vl:32b"}, {"name": "qwen3:30b-a3b"}]}
        )
        req.post.return_value = _resp(ok=True)
        assert client.unload_all() == 2
        assert req.post.call_count == 2
        for call in req.post.call_args_list:
            assert call.kwargs["json"]["keep_alive"] == 0


def test_unload_falls_back_to_embed_endpoint():
    client = OllamaClient()
    with patch("file_index.ollama_client.requests") as req:
        req.post.side_effect = [_resp(ok=False), _resp(ok=True)]
        client.unload("nomic-embed-text")
        urls = [c.args[0] for c in req.post.call_args_list]
        assert urls[0].endswith("/api/generate")
        assert urls[1].endswith("/api/embed")


def test_unload_all_survives_ollama_down():
    import requests as real_requests

    client = OllamaClient()
    with patch("file_index.ollama_client.requests") as req:
        req.RequestException = real_requests.RequestException
        req.get.side_effect = real_requests.ConnectionError("down")
        assert client.unload_all() == 0
