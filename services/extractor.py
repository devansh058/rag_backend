import os
from typing import Any

import fitz
import pytesseract  # requires the `tesseract` binary installed (e.g. `brew install tesseract` on macOS)
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".bmp"}


def extract_text(path: str) -> list[dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext in IMAGE_EXTENSIONS:
        return _extract_image(path)
    raise ValueError(f"Unsupported file type: {ext or 'unknown'}")


def _extract_pdf(path: str) -> list[dict[str, Any]]:
    doc = fitz.open(path)
    pages: list[dict[str, Any]] = []
    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text() or ""
            pages.append({"page": i, "text": text.strip()})
    finally:
        doc.close()
    return pages


def _extract_image(path: str) -> list[dict[str, Any]]:
    image = Image.open(path)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    text = pytesseract.image_to_string(image)
    return [{"page": 0, "text": text.strip()}]
