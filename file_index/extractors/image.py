"""Image analysis: EXIF (tier 1-cheap, stored separately) and structured VLM
analysis (tier 2). The VLM is prompted for strict JSON; one retry on parse
failure, then the raw output is stored and the row marked degraded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..ollama_client import OllamaClient

log = logging.getLogger("file_index.extractors.image")

VERSION = "image-1.2"

# Formats Ollama's vision path handles natively; everything else (HEIC, WEBP,
# TIFF, …) is silently misread or rejected, so we transcode first.
VLM_NATIVE_FORMATS = {"JPEG", "PNG"}
VLM_MAX_DIM = 1344  # larger images OOM the 32B VLM's vision encoder on 32 GB VRAM

VLM_SCHEMA_KEYS = {
    "description": str,
    "ocr_text": str,
    "type": str,
    "objects": list,
    "people_count": int,
    "inferred_context": str,
}
VALID_TYPES = {"photo", "screenshot", "document", "meme", "diagram", "artwork", "other"}

VLM_PROMPT = """Analyze this image and return ONLY a JSON object, no other text, with exactly these keys:
{
  "description": "detailed description of the image content",
  "ocr_text": "all readable text in the image, or empty string",
  "type": "one of: photo, screenshot, document, meme, diagram, artwork, other",
  "objects": ["list", "of", "notable", "objects"],
  "people_count": 0,
  "inferred_context": "what this image is likely for / where it likely came from"
}
Return only the JSON object."""


def extract_exif(path: Path) -> dict:
    """EXIF datetime, GPS, camera — best-effort, empty dict on failure."""
    out: dict = {}
    try:
        from PIL import ExifTags, Image

        with Image.open(path) as img:
            out["width"], out["height"] = img.size
            out["format"] = img.format
            exif = img.getexif()
            if not exif:
                return out
            named = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
            for key in ("DateTime", "DateTimeOriginal", "Make", "Model", "Software"):
                if key in named:
                    out[key] = str(named[key])
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
            if gps_ifd:
                gps = {ExifTags.GPSTAGS.get(k, str(k)): v for k, v in gps_ifd.items()}
                lat, lon = _gps_decimal(gps)
                if lat is not None:
                    out["gps"] = {"lat": lat, "lon": lon}
    except Exception as e:  # noqa: BLE001 — EXIF is best-effort
        log.debug("exif failed for %s: %s", path, e)
    return out


def _gps_decimal(gps: dict):
    try:
        def conv(vals, ref, neg):
            d, m, s = (float(v) for v in vals)
            dec = d + m / 60 + s / 3600
            return -dec if ref in neg else dec

        lat = conv(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"), "S")
        lon = conv(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"), "W")
        return lat, lon
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return None, None


def validate_vlm_json(raw: str) -> dict | None:
    """Parse + validate the VLM's JSON. Returns normalized dict or None."""
    raw = raw.strip()
    # tolerate markdown fences
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    out: dict = {}
    for key, typ in VLM_SCHEMA_KEYS.items():
        val = data.get(key)
        if val is None:
            val = typ()
        if typ is int:
            try:
                val = int(val)
            except (ValueError, TypeError):
                val = 0
        elif typ is list:
            if not isinstance(val, list):
                val = [str(val)] if val else []
            val = [str(x) for x in val]
        elif typ is str and not isinstance(val, str):
            val = str(val)
        out[key] = val
    if out["type"] not in VALID_TYPES:
        out["type"] = "other"
    return out


def prepare_for_vlm(path: Path, tmpdir: Path) -> Path:
    """Return a VLM-safe image path: JPEG/PNG within VLM_MAX_DIM.

    HEIC/WEBP/TIFF etc. are transcoded to JPEG; oversized images are downscaled.
    Returns the original path unchanged when it is already safe.
    """
    from PIL import Image

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as img:
        if img.format in VLM_NATIVE_FORMATS and max(img.size) <= VLM_MAX_DIM:
            return path
        img.thumbnail((VLM_MAX_DIM, VLM_MAX_DIM))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        out = tmpdir / (path.stem + ".vlm.jpg")
        img.save(out, "JPEG", quality=90)
        return out


def analyze_image(
    client: OllamaClient, model: str, path: Path
) -> tuple[dict | None, str, bool]:
    """Run the VLM. Returns (validated_json_or_None, raw_output, degraded)."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="file-index-vlm-") as tmp:
        try:
            safe = prepare_for_vlm(path, Path(tmp))
        except Exception as e:  # noqa: BLE001 — undecodable image: let the VLM try raw
            log.warning("could not preprocess %s (%s); sending as-is", path, e)
            safe = path
        raw = client.generate(model, VLM_PROMPT, images=[safe], format_json=True)
        data = validate_vlm_json(raw)
        if data is not None:
            return data, raw, False
        log.info("VLM JSON invalid for %s, retrying once", path)
        raw = client.generate(model, VLM_PROMPT, images=[safe], format_json=True)
        data = validate_vlm_json(raw)
        if data is not None:
            return data, raw, False
        return None, raw, True


def vlm_body_text(data: dict) -> str:
    """Flatten validated VLM JSON into searchable text."""
    parts = [
        data.get("description", ""),
        data.get("ocr_text", ""),
        data.get("inferred_context", ""),
        " ".join(data.get("objects", [])),
        data.get("type", ""),
    ]
    return "\n".join(p for p in parts if p)
