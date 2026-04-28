from typing import Any

from django.conf import settings
from qdrant_client.http import models as qmodels

from services.embedder import embed_query
from services.vector_store import ensure_collection, get_client


def retrieve(project_id: int, query: str, limit: int = 5) -> list[dict[str, Any]]:
    ensure_collection()
    vector = embed_query(query)
    client = get_client()
    name = settings.QDRANT_COLLECTION
    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="project_id",
                match=qmodels.MatchValue(value=project_id),
            )
        ]
    )
    resp = client.query_points(
        collection_name=name,
        query=vector,
        query_filter=flt,
        limit=limit,
    )
    out: list[dict[str, Any]] = []
    for h in resp.points:
        payload = h.payload or {}
        out.append(
            {
                "text": payload.get("text", ""),
                "page": payload.get("page", 0),
                "document_id": payload.get("document_id"),
                "document_title": payload.get("document_title", ""),
                "document_type": payload.get("document_type", ""),
                "discipline": payload.get("discipline", ""),
                "revision": payload.get("revision", ""),
                "score": h.score,
            }
        )
    return out
