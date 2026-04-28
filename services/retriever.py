"""Vector retrieval with Qdrant-side filtering and result deduplication."""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from django.conf import settings
from qdrant_client.http import models as qmodels

from services.embedder import embed_query
from services.vector_store import ensure_collection, get_client

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
    """Vector-search Qdrant with server-side filters; dedupe identical chunk text."""
    ensure_collection()
    vector = embed_query(query)
    client = get_client()
    name = settings.QDRANT_COLLECTION

    must: list[qmodels.FieldCondition] = [
        qmodels.FieldCondition(
            key="project_id",
            match=qmodels.MatchValue(value=project_id),
        )
    ]
    if tags:
        must.append(qmodels.FieldCondition(key="tags", match=qmodels.MatchAny(any=list(tags))))
    if language:
        must.append(qmodels.FieldCondition(key="language", match=qmodels.MatchValue(value=language)))
    if document_type:
        must.append(qmodels.FieldCondition(key="document_type", match=qmodels.MatchValue(value=document_type)))

    flt = qmodels.Filter(must=must)
    over_fetch = max(SEARCH_LIMIT, limit * 2)
    resp = client.query_points(
        collection_name=name,
        query=vector,
        query_filter=flt,
        limit=over_fetch,
    )

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for h in resp.points:
        payload = h.payload or {}
        text = (payload.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(
            {
                "text": text,
                "page": payload.get("page", 0),
                "document_id": payload.get("document_id"),
                "document_title": payload.get("document_title", ""),
                "document_type": payload.get("document_type", ""),
                "discipline": payload.get("discipline", ""),
                "revision": payload.get("revision", ""),
                "tags": payload.get("tags", []),
                "language": payload.get("language", "unknown"),
                "score": h.score,
            }
        )
        if len(out) >= limit:
            break

    logger.info(
        "retriever: project_id=%s query_len=%d hits=%d (after dedup, fetched=%d)",
        project_id,
        len(query),
        len(out),
        len(resp.points),
    )
    return out
