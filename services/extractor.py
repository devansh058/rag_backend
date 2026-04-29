"""Extraction with junk-removal and page-by-page streaming."""
from __future__ import annotations

import os
from typing import Any, Iterator, Optional

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".bmp"}

SKIP_PAGE_KEYWORDS = (
    "table of contents",
    "contents",
    "preface",
    "introduction",
)


def clean_text(text: Optional[str]) -> Optional[str]:
    """Drop boilerplate / noisy lines. Return None when the page is junk."""
    if not text:
        return None

    text_lower = text.lower()

    if any(keyword in text_lower for keyword in SKIP_PAGE_KEYWORDS):
        return None

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 3:
            continue
        if line.lower().startswith("page "):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or None


def iter_pages(path: str) -> Iterator[dict[str, Any]]:
    """Yield cleaned `{page, text}` dicts one page at a time.

    Streaming-friendly so large PDFs don't have to live in memory.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        yield from _iter_pdf(path)
    elif ext in IMAGE_EXTENSIONS:
        yield from _iter_image(path)
    else:
        raise ValueError(f"Unsupported file type: {ext or 'unknown'}")


def extract_text(path: str) -> list[dict[str, Any]]:
    """Backward-compatible: materialise all cleaned pages into a list."""
    return list(iter_pages(path))


def _iter_pdf(path: str) -> Iterator[dict[str, Any]]:
    import fitz  # local import so optional deps don't break import-time
    from io import BytesIO

    import pytesseract
    from PIL import Image

    doc = fitz.open(path)
    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            raw = page.get_text() or ""

            # Scanned / image-only PDFs have no text layer — rasterize and OCR.
            if not (raw and raw.strip()):
                try:
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.open(BytesIO(pix.tobytes("png")))
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    raw = pytesseract.image_to_string(img) or ""
                except Exception:
                    raw = ""

            cleaned = clean_text(raw)
            if cleaned:
                yield {"page": i, "text": cleaned}
    finally:
        doc.close()


def _iter_image(path: str) -> Iterator[dict[str, Any]]:
    # `pytesseract` requires the `tesseract` binary (e.g. `brew install tesseract`).
    import pytesseract
    from PIL import Image

    image = Image.open(path)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    raw = pytesseract.image_to_string(image) or ""
    cleaned = clean_text(raw)
    if cleaned:
        yield {"page": 0, "text": cleaned}
