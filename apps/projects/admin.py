from django.contrib import admin

from apps.projects.models import Project


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "project_code",
        "name",
        "client_name",
        "project_type",
        "status",
        "location",
        "created_at",
    )
    list_filter = ("project_type", "status")
    search_fields = ("project_code", "name", "client_name", "location")
