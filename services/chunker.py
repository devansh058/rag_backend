from typing import Any

CHUNK_SIZE = 500
OVERLAP = 50


def chunk_text(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for entry in pages:
        page_num = entry["page"]
        text = entry.get("text") or ""
        if not text:
            continue
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            piece = text[start:end].strip()
            if piece:
                chunks.append({"text": piece, "page": page_num})
            if end >= len(text):
                break
            start = end - OVERLAP
    return chunks
