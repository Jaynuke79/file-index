"""VLM JSON validation path, with the model mocked."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from file_index.extractors.image import analyze_image, validate_vlm_json, vlm_body_text

GOOD = {
    "description": "a cat on a keyboard",
    "ocr_text": "",
    "type": "photo",
    "objects": ["cat", "keyboard"],
    "people_count": 0,
    "inferred_context": "home office photo",
}


def test_valid_json_accepted():
    out = validate_vlm_json(json.dumps(GOOD))
    assert out == GOOD


def test_markdown_fenced_json_accepted():
    out = validate_vlm_json("```json\n" + json.dumps(GOOD) + "\n```")
    assert out is not None
    assert out["type"] == "photo"


def test_prose_around_json_accepted():
    out = validate_vlm_json("Here is the analysis: " + json.dumps(GOOD) + " hope that helps!")
    assert out is not None


def test_missing_keys_defaulted():
    out = validate_vlm_json('{"description": "x"}')
    assert out["ocr_text"] == ""
    assert out["objects"] == []
    assert out["people_count"] == 0


def test_wrong_types_coerced():
    out = validate_vlm_json(
        '{"description": "x", "people_count": "3", "objects": "dog", "type": "banana"}'
    )
    assert out["people_count"] == 3
    assert out["objects"] == ["dog"]
    assert out["type"] == "other"


def test_garbage_rejected():
    assert validate_vlm_json("I cannot analyze this image.") is None
    assert validate_vlm_json("{broken json") is None
    assert validate_vlm_json("") is None


def test_analyze_retries_once_then_degrades():
    client = MagicMock()
    client.generate.side_effect = ["not json at all", "still not json"]
    data, raw, degraded = analyze_image(client, "vlm", Path("/x.jpg"))
    assert degraded is True
    assert data is None
    assert raw == "still not json"
    assert client.generate.call_count == 2


def test_analyze_retry_succeeds():
    client = MagicMock()
    client.generate.side_effect = ["garbage", json.dumps(GOOD)]
    data, raw, degraded = analyze_image(client, "vlm", Path("/x.jpg"))
    assert degraded is False
    assert data["description"] == "a cat on a keyboard"
    assert client.generate.call_count == 2


def test_body_text_flattening():
    body = vlm_body_text(GOOD)
    assert "cat" in body and "home office" in body
