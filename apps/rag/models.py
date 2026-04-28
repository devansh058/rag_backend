"""Vector chunk storage using pgvector inside Postgres."""
from django.contrib.postgres.fields import ArrayField
from django.contrib.postgres.indexes import GinIndex
from django.db import models
from pgvector.django import HnswIndex, VectorField

VECTOR_DIMENSIONS = 384  # all-MiniLM-L6-v2


class Chunk(models.Model):
    """One retrieved-able piece of a Document, with its embedding."""

    document = models.ForeignKey(
        "documents.Document",
        on_delete=models.CASCADE,
        related_name="chunks",
    )
    project_id = models.BigIntegerField(db_index=True)

    text = models.TextField()
    page = models.IntegerField(default=0)

    embedding = VectorField(dimensions=VECTOR_DIMENSIONS)

    tags = ArrayField(models.CharField(max_length=64), default=list, blank=True)
    language = models.CharField(max_length=8, default="unknown")

    document_title = models.CharField(max_length=255, blank=True)
    document_type = models.CharField(max_length=32, blank=True)
    discipline = models.CharField(max_length=64, blank=True)
    revision = models.CharField(max_length=32, blank=True)
    project_code = models.CharField(max_length=64, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            HnswIndex(
                name="rag_chunk_embedding_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
            GinIndex(name="rag_chunk_tags_gin", fields=["tags"]),
            models.Index(fields=["document_type"]),
            models.Index(fields=["language"]),
        ]

    def __str__(self):
        return f"chunk(doc={self.document_id}, page={self.page})"
