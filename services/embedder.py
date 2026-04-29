"""384-dim embeddings via sentence-transformers.

Default device is CPU — PyTorch MPS inside Celery prefork workers on macOS
frequently crashes with SIGABRT during model load or encode.
Set EMBEDDING_DEVICE=mps in .env only when not using fork workers (e.g. --pool=solo).
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_model = None
MODEL_NAME = "all-MiniLM-L6-v2"


def _embedding_device() -> str:
    try:
        from django.conf import settings

        dev = getattr(settings, "EMBEDDING_DEVICE", None)
        if dev:
            return str(dev)
    except Exception:
        pass
    return os.environ.get("EMBEDDING_DEVICE", "cpu")


def _get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = _embedding_device()
        # Reduce cross-thread surprises in worker children
        if device == "cpu":
            torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
        logger.info("Loading SentenceTransformer %s on device=%s", MODEL_NAME, device)
        _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    model = _get_model()
    v = model.encode(text, convert_to_numpy=True)
    return v.tolist()
