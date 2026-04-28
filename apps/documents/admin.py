from django.contrib import admin

from .models import Document


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "project",
        "title",
        "document_type",
        "discipline",
        "revision",
        "status",
        "created_at",
    )
    list_filter = ("document_type", "status", "discipline")
    search_fields = ("title", "discipline", "revision", "project__project_code", "project__name")
