"""Extraction with junk-removal and page-by-page streaming.

OCR language selection:
- By default, Tesseract OSD (orientation / script detection) picks a script, we map
  it to `lang=` codes and filter by what `tesseract --list-langs` reports as installed.
- Set TESSERACT_LANG=eng+deu (example) to skip OSD and force those packs.
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Any, Iterator, Optional

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".bmp"}

logger = logging.getLogger(__name__)

# Tesseract OSD «Script: …» values → tessdata language code(s). Use '+' for multiple
# packs; `_filter_installed` drops missing parts. See `tesseract --list-langs`.
# Docs: https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
SCRIPT_TO_TESS_LANG: dict[str, str] = {
    "Arabic": "ara",
    "Armenian": "hye",
    "Bengali": "ben",
    "Canadian_Aboriginal": "chr",  # syllabics — may be absent; will fall back
    "Cherokee": "chr",
    "Cyrillic": "rus",
    "Devanagari": "hin",
    "Ethiopic": "amh",
    "Georgian": "kat",
    "Greek": "ell",
    "Gujarati": "guj",
    "Gurmukhi": "pan",
    "Han": "chi_sim+chi_tra",
    "Hangul": "kor",
    "Hebrew": "heb",
    "Japanese": "jpn",
    "Kannada": "kan",
    "Khmer": "khm",
    "Lao": "lao",
    "Latin": "eng",
    "Malayalam": "mal",
    "Myanmar": "mya",
    "Oriya": "ori",
    "Sinhala": "sin",
    "Syriac": "syr",
    "Tamil": "tam",
    "Telugu": "tel",
    "Thai": "tha",
    "Tibetan": "bod",
    "Mongolian": "mon",
    # Vietnamese is usually Latin script in tess OSD
    "Vietnamese": "eng",
    # Rare / generic fallbacks
    "Fraktur": "eng",
}

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
        if len(line) < 2:
            continue
        if len(line) < 3 and all(ord(ch) < 128 for ch in line):
            continue
        if line.lower().startswith("page "):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or None


@functools.lru_cache(maxsize=1)
def _installed_tesseract_langs() -> frozenset[str]:
    import pytesseract

    try:
        return frozenset(pytesseract.get_languages())
    except Exception:
        logger.warning("Could not list Tesseract languages", exc_info=True)
        return frozenset()


def _filter_installed_lang_spec(lang_spec: str, installed: frozenset[str]) -> str:
    if not lang_spec:
        return ""
    parts = [p.strip() for p in lang_spec.split("+") if p.strip() and p.strip() in installed]
    return "+".join(parts)


def _osd_script_for_image(image: Any) -> tuple[Optional[str], float]:
    """Return (script_name, script_confidence) from Tesseract OSD, or (None, 0)."""
    import pytesseract
    from pytesseract import Output

    try:
        osd = pytesseract.image_to_osd(image, output_type=Output.DICT)
    except pytesseract.TesseractError as exc:
        logger.warning("Tesseract OSD failed (need osd.traineddata?): %s", exc)
        return None, 0.0
    except Exception:
        logger.warning("Tesseract OSD unexpected error", exc_info=True)
        return None, 0.0

    script = osd.get("script")
    conf = float(osd.get("script_conf") or 0.0)
    if not script or not str(script).strip():
        return None, 0.0
    return str(script).strip(), conf


def _settings_tesseract() -> tuple[str, bool, bool, bool]:
    try:
        from django.conf import settings

        fixed = (getattr(settings, "TESSERACT_LANG", None) or "").strip()
        script_on = bool(getattr(settings, "TESSERACT_SCRIPT_DETECTION", True))
        append_eng = bool(getattr(settings, "TESSERACT_APPEND_ENG", True))
        supple = bool(getattr(settings, "PDF_OCR_SUPPLEMENT_TEXT_LAYER", False))
        return fixed, script_on, append_eng, supple
    except Exception:
        fixed = os.environ.get("TESSERACT_LANG", "").strip()
        script_on = os.environ.get("TESSERACT_SCRIPT_DETECTION", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        append_eng = os.environ.get("TESSERACT_APPEND_ENG", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        supple = os.environ.get("PDF_OCR_SUPPLEMENT_TEXT_LAYER", "").lower() in (
            "1",
            "true",
            "yes",
        )
        return fixed, script_on, append_eng, supple


def _resolve_primary_lang_spec(image: Any) -> str:
    """Single `lang=` string for Tesseract (may contain '+'), before fallbacks."""
    fixed, script_on, append_eng, _ = _settings_tesseract()
    installed = _installed_tesseract_langs()

    if fixed:
        s = _filter_installed_lang_spec(fixed, installed)
        if s:
            return s
        logger.warning("TESSERACT_LANG=%r has no installed packs; ignoring.", fixed)

    if not script_on:
        s = _filter_installed_lang_spec("eng", installed)
        return s if s else "eng"

    script, conf = _osd_script_for_image(image)
    if script:
        logger.info("Tesseract OSD script=%r confidence=%.2f", script, conf)

    mapped = SCRIPT_TO_TESS_LANG.get(script or "") or "eng"
    if script and script not in SCRIPT_TO_TESS_LANG:
        logger.warning("No tess lang mapping for OSD script %r; using %r", script, mapped)

    primary = _filter_installed_lang_spec(mapped, installed)
    if not primary:
        primary = _filter_installed_lang_spec("eng", installed) or "eng"

    if append_eng and "eng" in installed:
        parts = primary.split("+")
        if "eng" not in parts:
            primary = primary + "+eng"

    return _filter_installed_lang_spec(primary, installed) or (
        _filter_installed_lang_spec("eng", installed) or "eng"
    )


def _ocr_lang_attempts(image: Any) -> list[str]:
    """Ordered `lang` values to try for this image (deduped)."""
    primary = _resolve_primary_lang_spec(image)
    attempts: list[str] = []
    seen: set[str] = set()

    def add(spec: str) -> None:
        spec = spec.strip()
        if spec and spec not in seen:
            seen.add(spec)
            attempts.append(spec)

    add(primary)
    for p in primary.split("+"):
        add(p)
    add("eng")
    return attempts


def _ocr_pil_image(img: Any) -> str:
    import pytesseract

    for lang in _ocr_lang_attempts(img):
        try:
            text = pytesseract.image_to_string(img, lang=lang) or ""
            if text.strip():
                logger.info("OCR succeeded using lang=%r", lang)
                return text
        except pytesseract.TesseractError as exc:
            logger.warning("Tesseract OCR failed for lang=%r: %s", lang, exc)
        except Exception:
            logger.exception("OCR unexpected error for lang=%r", lang)
    return ""


def _pdf_ocr_supplement() -> bool:
    _, _, _, supple = _settings_tesseract()
    return supple


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

    from PIL import Image

    doc = fitz.open(path)
    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            try:
                raw = page.get_text(sort=True) or ""
            except TypeError:
                raw = page.get_text() or ""

            ocr_supplement = bool(raw and raw.strip() and _pdf_ocr_supplement())

            if not (raw and raw.strip()) or ocr_supplement:
                try:
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.open(BytesIO(pix.tobytes("png")))
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    ocr_txt = _ocr_pil_image(img)
                    if ocr_supplement and raw and raw.strip():
                        raw = (raw.rstrip() + "\n\n" + ocr_txt).strip()
                    else:
                        raw = ocr_txt
                except Exception:
                    if not (raw and raw.strip()):
                        raw = ""

            cleaned = clean_text(raw)
            if cleaned:
                yield {"page": i, "text": cleaned}
    finally:
        doc.close()


def _iter_image(path: str) -> Iterator[dict[str, Any]]:
    from PIL import Image

    image = Image.open(path)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    raw = _ocr_pil_image(image) or ""
    cleaned = clean_text(raw)
    if cleaned:
        yield {"page": 0, "text": cleaned}
