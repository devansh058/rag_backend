from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from apps.documents.models import Document
from apps.projects.models import Project
from apps.projects.serializers import DocumentSerializer, ProjectSerializer
from apps.rag.serializers import QuerySerializer
from services.llm import generate_answer
from services.retriever import retrieve
from workers.tasks import process_document


class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.all().order_by("-created_at")
    serializer_class = ProjectSerializer
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    @action(detail=True, methods=["get"], url_path="documents")
    def documents(self, request, pk=None):
        project = self.get_object()
        qs = project.documents.all().order_by("-created_at")
        return Response(DocumentSerializer(qs, many=True, context={"request": request}).data)

    @action(detail=True, methods=["post"], url_path="upload")
    def upload(self, request, pk=None):
        project = self.get_object()
        upload = request.FILES.get("file")
        if not upload:
            return Response({"detail": "Missing file field `file`."}, status=status.HTTP_400_BAD_REQUEST)

        doc = Document.objects.create(
            project=project,
            file=upload,
            title=request.data.get("title", "") or "",
            document_type=request.data.get("document_type", Document.DocumentType.OTHER),
            discipline=request.data.get("discipline", "") or "",
            revision=request.data.get("revision", "") or "",
            status=Document.Status.UPLOADED,
        )
        process_document.delay(doc.id)
        out = DocumentSerializer(doc, context={"request": request})
        return Response(out.data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["post"], url_path="query")
    def query(self, request, pk=None):
        project = self.get_object()
        ser = QuerySerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data
        q = data["query"]

        hits = retrieve(
            project.id,
            q,
            limit=data.get("top_k") or 5,
            tags=data.get("tags") or None,
            language=(data.get("language") or "").strip() or None,
            document_type=(data.get("document_type") or "").strip() or None,
        )

        project_meta = {
            "project_code": project.project_code,
            "name": project.name,
            "client_name": project.client_name,
            "location": project.location,
            "project_type": project.get_project_type_display(),
            "status": project.get_status_display(),
        }
        answer = generate_answer(q, hits, project=project_meta)

        sources = [
            {
                "text": h["text"],
                "page": h["page"],
                "document_id": h["document_id"],
                "document_title": h.get("document_title", ""),
                "document_type": h.get("document_type", ""),
                "discipline": h.get("discipline", ""),
                "revision": h.get("revision", ""),
                "tags": h.get("tags", []),
                "language": h.get("language", "unknown"),
                "score": h.get("score"),
            }
            for h in hits
        ]
        return Response({"answer": answer, "sources": sources})
