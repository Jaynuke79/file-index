"""PDF text extraction via PyMuPDF. Pages without a text layer are recorded so
tier 2 can rasterize them for the VLM."""

from __future__ import annotations

from pathlib import Path

VERSION = "pdf-1.0"


def extract(path: Path, size_cap: int = 1_000_000) -> tuple[str, list[int]]:
    """Returns (text, scanned_page_numbers). Page numbers are 0-based pages
    that had no extractable text (likely scans)."""
    import fitz  # PyMuPDF

    text_parts: list[str] = []
    scanned_pages: list[int] = []
    total = 0
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            t = page.get_text().strip()
            if t:
                if total < size_cap:
                    text_parts.append(t)
                    total += len(t)
            else:
                # no text layer; if the page has images it is probably a scan
                if page.get_images(full=False):
                    scanned_pages.append(i)
    return "\n\n".join(text_parts)[:size_cap], scanned_pages


def rasterize_page(path: Path, page_number: int, out_path: Path, dpi: int = 150) -> Path:
    import fitz

    with fitz.open(path) as doc:
        page = doc[page_number]
        pix = page.get_pixmap(dpi=dpi)
        pix.save(str(out_path))
    return out_path
