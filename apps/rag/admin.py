from django.contrib import admin

from apps.rag.models import Chunk


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "document",
        "project_id",
        "page",
        "document_type",
        "discipline",
        "language",
        "created_at",
    )
    list_filter = ("document_type", "language", "discipline")
    search_fields = ("text", "document_title", "project_code")
