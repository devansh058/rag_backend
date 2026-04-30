import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except Exception:  # python-dotenv missing in some environments — fine
    pass

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-change-me-in-production-dev-only",
)

DEBUG = os.environ.get("DJANGO_DEBUG", "true").lower() in ("1", "true", "yes")

ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "apps.projects",
    "apps.documents",
    "apps.rag",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "rag_backend.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "rag_backend.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("POSTGRES_DB", "rag_db"),
        "USER": os.environ.get("POSTGRES_USER", "postgres"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
        "HOST": os.environ.get("POSTGRES_HOST", "localhost"),
        "PORT": os.environ.get("POSTGRES_PORT", "5432"),
        "OPTIONS": {},
    }
}

sslmode = os.environ.get("POSTGRES_SSLMODE")
if sslmode:
    DATABASES["default"]["OPTIONS"]["sslmode"] = sslmode

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

MEDIA_ROOT = BASE_DIR / "media"
MEDIA_URL = "/media/"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.MultiPartParser",
        "rest_framework.parsers.FormParser",
    ],
}

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = TIME_ZONE
CELERY_IMPORTS = ("workers.tasks",)
CELERY_TASK_ALWAYS_EAGER = os.environ.get("CELERY_TASK_ALWAYS_EAGER", "").lower() in ("1", "true", "yes")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "2048"))

VECTOR_DIMENSIONS = 384  # all-MiniLM-L6-v2

# -- Tesseract OCR -------------------------------------------------------------
# Automatic script detection (Tesseract OSD) maps page script → best-effort lang.
# Requires `osd.traineddata` (ships with most Tesseract installs).
# Install extra languages: `brew install tesseract-lang` then `tesseract --list-langs`.
_truthy = ("1", "true", "yes")
TESSERACT_SCRIPT_DETECTION = os.environ.get(
    "TESSERACT_SCRIPT_DETECTION", "true"
).lower() in _truthy
# When script ≠ Latin, append English if installed (common for mixed business docs).
TESSERACT_APPEND_ENG = os.environ.get("TESSERACT_APPEND_ENG", "true").lower() in _truthy
# If non-empty, skip OSD and use this `lang` string only (e.g. eng+rus).
TESSERACT_LANG = os.environ.get("TESSERACT_LANG", "").strip()
# If True, also OCR rasterized pages when a text layer exists (slower; mixed image+text).
PDF_OCR_SUPPLEMENT_TEXT_LAYER = os.environ.get(
    "PDF_OCR_SUPPLEMENT_TEXT_LAYER", ""
).lower() in _truthy
# Split each rasterized page into N horizontal bands (top→bottom); OSD/lang runs per band.
# Helps mixed layouts (e.g. English header + Hindi body). 1 = one OCR pass per page (default).
try:
    TESSERACT_HORIZONTAL_STRIPS = max(
        1,
        min(
            16,
            int(os.environ.get("TESSERACT_HORIZONTAL_STRIPS", "5").strip() or "5"),
        ),
    )
except ValueError:
    TESSERACT_HORIZONTAL_STRIPS = 1

# Celery prefork + PyTorch MPS on macOS → SIGABRT; default cpu.
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
