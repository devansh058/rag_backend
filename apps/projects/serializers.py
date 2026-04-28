from rest_framework import serializers

from apps.documents.models import Document
from apps.projects.models import Project


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = (
            "id",
            "name",
            "project_code",
            "client_name",
            "location",
            "project_type",
            "status",
            "start_date",
            "expected_end_date",
            "description",
            "created_at",
        )
        read_only_fields = ("id", "created_at")


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = (
            "id",
            "project",
            "title",
            "document_type",
            "discipline",
            "revision",
            "file",
            "status",
            "error_message",
            "created_at",
        )
        read_only_fields = ("id", "project", "status", "error_message", "created_at")
