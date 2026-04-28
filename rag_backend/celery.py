import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass

from celery import Celery  # noqa: E402

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rag_backend.settings")

app = Celery("rag_backend")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
