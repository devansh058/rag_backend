from django.db import models


class Document(models.Model):
    """A construction document (drawing, contract, BOQ, RFI, report, etc.)."""

    class Status(models.TextChoices):
        UPLOADED = "uploaded", "Uploaded"
        PROCESSING = "processing", "Processing"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"

    class DocumentType(models.TextChoices):
        DRAWING = "drawing", "Drawing / Blueprint"
        SPECIFICATION = "specification", "Technical Specification"
        BOQ = "boq", "Bill of Quantities"
        CONTRACT = "contract", "Contract / Agreement"
        TENDER = "tender", "Tender Document"
        RFI = "rfi", "Request for Information"
        SUBMITTAL = "submittal", "Submittal"
        SAFETY = "safety", "Safety / HSE Report"
        INSPECTION = "inspection", "Inspection Report"
        PROGRESS = "progress", "Progress Report"
        PERMIT = "permit", "Permit / Approval"
        INVOICE = "invoice", "Invoice / Payment"
        SITE_PHOTO = "site_photo", "Site Photo"
        OTHER = "other", "Other"

    project = models.ForeignKey(
        "projects.Project",
        on_delete=models.CASCADE,
        related_name="documents",
    )
    title = models.CharField(max_length=255, blank=True)
    document_type = models.CharField(
        max_length=32,
        choices=DocumentType.choices,
        default=DocumentType.OTHER,
    )
    discipline = models.CharField(
        max_length=64,
        blank=True,
        help_text="Civil, Structural, Architectural, MEP, etc.",
    )
    revision = models.CharField(max_length=32, blank=True)
    file = models.FileField(upload_to="documents/%Y/%m/")
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.UPLOADED,
    )
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        label = self.title or (self.file.name.rsplit("/", 1)[-1] if self.file else f"doc-{self.pk}")
        return f"{label} ({self.get_document_type_display()})"
