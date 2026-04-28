"""pgvector-backed similarity search with server-side filtering and dedup."""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from services.embedder import embed_query

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
SEARCH_LIMIT = 10  # over-fetch, then dedupe down to top_k


def retrieve(
    project_id: int,
    query: str,
    limit: int = DEFAULT_TOP_K,
    tags: Optional[Sequence[str]] = None,
    language: Optional[str] = None,
    document_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Cosine-similarity search over the `Chunk` table, scoped to a project."""
    from apps.rag.models import Chunk
    from pgvector.django import CosineDistance

    vector = embed_query(query)

    qs = Chunk.objects.filter(project_id=project_id)
    if tags:
        qs = qs.filter(tags__overlap=list(tags))
    if language:
        qs = qs.filter(language=language)
    if document_type:
        qs = qs.filter(document_type=document_type)

    over_fetch = max(SEARCH_LIMIT, limit * 2)
    qs = qs.annotate(distance=CosineDistance("embedding", vector)).order_by("distance")[:over_fetch]

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    fetched = 0
    for c in qs.iterator():
        fetched += 1
        text = (c.text or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        # Cosine distance is in [0, 2]; similarity in [-1, 1]. For typical normalized
        # embeddings (>= 0 dot product) similarity in [0, 1].
        score = 1.0 - float(c.distance) if c.distance is not None else None
        out.append(
            {
                "text": text,
                "page": c.page,
                "document_id": c.document_id,
                "document_title": c.document_title or "",
                "document_type": c.document_type or "",
                "discipline": c.discipline or "",
                "revision": c.revision or "",
                "tags": list(c.tags or []),
                "language": c.language or "unknown",
                "score": score,
            }
        )
        if len(out) >= limit:
            break

    logger.info(
        "retriever: project_id=%s query_len=%d hits=%d (after dedup, fetched=%d)",
        project_id,
        len(query),
        len(out),
        fetched,
    )
    return out
