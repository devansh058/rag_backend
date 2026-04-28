"""Sentence-aware chunking with real overlap, plus tag and language metadata."""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
OVERLAP = 100

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


KEYWORD_TAGS: dict[str, tuple[str, ...]] = {
    "electrical": ("voltage", "wiring", "cable", "circuit", "transformer", "switchgear"),
    "plumbing": ("pipe", "water", "drain", "sanitary", "valve", "plumbing"),
    "structural": ("beam", "column", "slab", "rebar", "reinforcement", "concrete", "steel"),
    "civil": ("excavation", "earthwork", "foundation", "soil", "backfill", "grading"),
    "hvac": ("hvac", "duct", "ventilation", "air conditioning", "chiller", "boiler"),
    "safety": ("safety", "ppe", "hazard", "incident", "hse", "lockout"),
    "legal": ("contract", "agreement", "clause", "liability", "indemnity"),
    "financial": ("invoice", "payment", "boq", "cost", "budget", "quotation"),
    "scheduling": ("schedule", "milestone", "gantt", "delay", "deadline"),
    "materials": ("cement", "aggregate", "sand", "brick", "tile", "paint"),
}


def generate_tags(text: str) -> list[str]:
    """Return construction-domain tags inferred from keyword presence."""
    if not text:
        return []
    text_lower = text.lower()
    tags: list[str] = []
    for tag, words in KEYWORD_TAGS.items():
        if any(word in text_lower for word in words):
            tags.append(tag)
    return tags


def detect_language(text: str) -> str:
    """Best-effort ISO-639-1 language code, `unknown` on failure."""
    if not text or len(text.strip()) < 20:
        return "unknown"
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return "unknown"


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _build_chunks_for_text(text: str, page_num: int, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    """Greedy sentence packing into ~chunk_size buckets with character-level overlap."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[dict[str, Any]] = []
    current = ""

    def flush(buf: str) -> str:
        if not buf.strip():
            return ""
        chunks.append({"text": buf.strip(), "page": page_num})
        if overlap <= 0 or len(buf) <= overlap:
            return ""
        return buf[-overlap:]

    for sentence in sentences:
        sentence = sentence if sentence.endswith((".", "!", "?")) else sentence + "."
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        # Sentence would overflow — flush and seed next chunk with overlap tail.
        seed = flush(current)
        if len(sentence) > chunk_size:
            # Single sentence is bigger than chunk_size: hard-split it.
            i = 0
            while i < len(sentence):
                piece = sentence[i : i + chunk_size]
                seed = flush(seed + (" " if seed else "") + piece)
                i += chunk_size
            current = seed
        else:
            current = (seed + " " + sentence).strip() if seed else sentence

    if current.strip():
        chunks.append({"text": current.strip(), "page": page_num})

    return chunks


def chunk_text(
    pages: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[dict[str, Any]]:
    """Chunk a list of page dicts. Each chunk gets `tags` and `language`."""
    chunks: list[dict[str, Any]] = []
    for page in pages:
        text = (page.get("text") or "").strip()
        if not text:
            continue
        page_chunks = _build_chunks_for_text(text, page["page"], chunk_size, overlap)
        for c in page_chunks:
            c["tags"] = generate_tags(c["text"])
            c["language"] = detect_language(c["text"])
        chunks.extend(page_chunks)

    logger.info("chunker: produced %d chunks across %d pages", len(chunks), len(pages))
    return chunks
