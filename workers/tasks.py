import logging
import traceback

from celery import shared_task
from django.conf import settings

from apps.documents.models import Document
from services.chunker import chunk_text
from services.embedder import embed_texts
from services.extractor import extract_text
from services.vector_store import delete_document_vectors, upsert_chunks

logger = logging.getLogger(__name__)


@shared_task
def process_document(document_id: int) -> None:
    try:
        doc = Document.objects.select_related("project").get(pk=document_id)
    except Document.DoesNotExist:
        logger.error("Document %s not found", document_id)
        return

    doc.status = Document.Status.PROCESSING
    doc.error_message = ""
    doc.save(update_fields=["status", "error_message"])

    try:
        delete_document_vectors(document_id)
        path = doc.file.path
        pages = extract_text(path)
        chunks = chunk_text(pages)
        texts = [c["text"] for c in chunks]
        if not texts:
            doc.status = Document.Status.DONE
            doc.save(update_fields=["status"])
            return
        vectors = embed_texts(texts)
        extra_payload = {
            "document_title": doc.title or "",
            "document_type": doc.document_type,
            "discipline": doc.discipline or "",
            "revision": doc.revision or "",
            "project_code": doc.project.project_code,
        }
        upsert_chunks(
            project_id=doc.project_id,
            document_id=document_id,
            chunk_payloads=chunks,
            vectors=vectors,
            extra_payload=extra_payload,
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
