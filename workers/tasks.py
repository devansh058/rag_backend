import logging
import time
import traceback

from celery import shared_task
from django.conf import settings

from apps.documents.models import Document
from services.chunker import chunk_text
from services.embedder import embed_texts
from services.extractor import iter_pages
from services.vector_store import delete_document_vectors, upsert_chunks

logger = logging.getLogger(__name__)


@shared_task
def process_document(document_id: int) -> None:
    """Streaming pipeline: extract → clean → chunk → embed → upsert, page-by-page.

    This keeps memory bounded for very large PDFs since we never materialise
    the whole document or all chunks/vectors at once.
    """
    try:
        doc = Document.objects.select_related("project").get(pk=document_id)
    except Document.DoesNotExist:
        logger.error("Document %s not found", document_id)
        return

    doc.status = Document.Status.PROCESSING
    doc.error_message = ""
    doc.save(update_fields=["status", "error_message"])

    started = time.perf_counter()
    total_chunks = 0
    total_stored = 0
    pages_processed = 0
    pages_skipped = 0

    try:
        delete_document_vectors(document_id)
        path = doc.file.path

        extra_payload = {
            "document_title": doc.title or "",
            "document_type": doc.document_type,
            "discipline": doc.discipline or "",
            "revision": doc.revision or "",
            "project_code": doc.project.project_code,
        }

        for page in iter_pages(path):
            if not page.get("text"):
                pages_skipped += 1
                continue
            pages_processed += 1

            chunks = chunk_text([page])
            if not chunks:
                continue

            texts = [c["text"] for c in chunks]
            vectors = embed_texts(texts)
            stored = upsert_chunks(
                project_id=doc.project_id,
                document_id=document_id,
                chunk_payloads=chunks,
                vectors=vectors,
                extra_payload=extra_payload,
            )
            total_chunks += len(chunks)
            total_stored += stored
            logger.info(
                "process_document: doc=%s page=%s chunks=%d stored=%d",
                document_id,
                page.get("page"),
                len(chunks),
                stored,
            )

        elapsed = time.perf_counter() - started
        logger.info(
            "process_document: doc=%s done in %.2fs (pages_kept=%d, pages_skipped=%d, "
            "chunks=%d, stored=%d)",
            document_id,
            elapsed,
            pages_processed,
            pages_skipped,
            total_chunks,
            total_stored,
        )

        doc.status = Document.Status.DONE
        doc.save(update_fields=["status"])
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("process_document failed: %s", tb)
        doc.status = Document.Status.FAILED
        doc.error_message = str(exc)[:2000]
        doc.save(update_fields=["status", "error_message"])
        if getattr(settings, "CELERY_TASK_ALWAYS_EAGER", False):
            raise
