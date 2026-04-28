import logging
import uuid
from typing import Any, Optional

from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)

VECTOR_SIZE = 384  # all-MiniLM-L6-v2

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    return _client


def ensure_collection() -> None:
    client = get_client()
    name = settings.QDRANT_COLLECTION
    collections = client.get_collections().collections
    names = {c.name for c in collections}
    if name not in names:
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE),
        )


def delete_document_vectors(document_id: int) -> None:
    ensure_collection()
    client = get_client()
    name = settings.QDRANT_COLLECTION
    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="document_id",
                match=qmodels.MatchValue(value=document_id),
            )
        ]
    )
    client.delete(collection_name=name, points_selector=flt)


def upsert_chunks(
    project_id: int,
    document_id: int,
    chunk_payloads: list[dict[str, Any]],
    vectors: list[list[float]],
    extra_payload: Optional[dict[str, Any]] = None,
) -> int:
    """Upsert chunks into Qdrant. Returns the number of points written."""
    if not chunk_payloads:
        return 0
    ensure_collection()
    client = get_client()
    name = settings.QDRANT_COLLECTION
    extra = extra_payload or {}
    points = []
    for payload, vec in zip(chunk_payloads, vectors):
        pid = str(uuid.uuid4())
        body = {
            "text": payload["text"],
            "page": payload["page"],
            "project_id": project_id,
            "document_id": document_id,
            "tags": payload.get("tags", []),
            "language": payload.get("language", "unknown"),
        }
        body.update(extra)
        points.append(qmodels.PointStruct(id=pid, vector=vec, payload=body))
    client.upsert(collection_name=name, points=points)
    logger.info(
        "vector_store: upserted %d points (document_id=%s, project_id=%s)",
        len(points),
        document_id,
        project_id,
    )
    return len(points)
