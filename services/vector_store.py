"""pgvector-backed vector storage. Vectors live in Postgres alongside the rest
of the relational data, so no extra service is required."""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def ensure_collection() -> None:
    """No-op for pgvector. The `vector` extension and tables are created by
    Django migrations (see apps/rag/migrations/0001_initial.py)."""
    return None


def delete_document_vectors(document_id: int) -> int:
    """Delete all chunks for a given document. Returns the number deleted."""
    from apps.rag.models import Chunk

    deleted, _ = Chunk.objects.filter(document_id=document_id).delete()
    if deleted:
        logger.info("vector_store: deleted %d chunks for document_id=%s", deleted, document_id)
    return int(deleted)


def upsert_chunks(
    project_id: int,
    document_id: int,
    chunk_payloads: list[dict[str, Any]],
    vectors: list[list[float]],
    extra_payload: Optional[dict[str, Any]] = None,
) -> int:
    """Bulk-insert chunk rows. Returns the number of rows written."""
    if not chunk_payloads:
        return 0

    from apps.rag.models import Chunk

    extra = extra_payload or {}
    rows = []
    for payload, vec in zip(chunk_payloads, vectors):
        rows.append(
            Chunk(
                document_id=document_id,
                project_id=project_id,
                text=payload["text"],
                page=int(payload.get("page", 0)),
                embedding=vec,
                tags=list(payload.get("tags", []) or []),
                language=payload.get("language", "unknown"),
                document_title=extra.get("document_title", "") or "",
                document_type=extra.get("document_type", "") or "",
                discipline=extra.get("discipline", "") or "",
                revision=extra.get("revision", "") or "",
                project_code=extra.get("project_code", "") or "",
            )
        )

    Chunk.objects.bulk_create(rows, batch_size=500)
    logger.info(
        "vector_store: inserted %d chunks (document_id=%s, project_id=%s)",
        len(rows),
        document_id,
        project_id,
    )
    return len(rows)
