from django.db import models


class Project(models.Model):
    """A construction project (site/job) belonging to the company."""

    class ProjectType(models.TextChoices):
        RESIDENTIAL = "residential", "Residential"
        COMMERCIAL = "commercial", "Commercial"
        INDUSTRIAL = "industrial", "Industrial"
        INFRASTRUCTURE = "infrastructure", "Infrastructure"
        RENOVATION = "renovation", "Renovation"
        OTHER = "other", "Other"

    class Status(models.TextChoices):
        PLANNING = "planning", "Planning"
        IN_PROGRESS = "in_progress", "In Progress"
        ON_HOLD = "on_hold", "On Hold"
        COMPLETED = "completed", "Completed"
        CANCELLED = "cancelled", "Cancelled"

    name = models.CharField(max_length=255)
    project_code = models.CharField(max_length=64, unique=True)
    client_name = models.CharField(max_length=255, blank=True)
    location = models.CharField(max_length=255, blank=True)
    project_type = models.CharField(
        max_length=32,
        choices=ProjectType.choices,
        default=ProjectType.COMMERCIAL,
    )
    status = models.CharField(
        max_length=32,
        choices=Status.choices,
        default=Status.PLANNING,
    )
    start_date = models.DateField(null=True, blank=True)
    expected_end_date = models.DateField(null=True, blank=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"{self.project_code} – {self.name}"
