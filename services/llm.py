"""Answer generation for the construction RAG assistant (Gemini or local Ollama)."""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional

from django.conf import settings

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are a construction-domain assistant for a construction company. "
    "You help engineers, project managers, QA/QS, procurement, and site staff "
    "understand project documents (drawings, specifications, BOQs, contracts, "
    "RFIs, submittals, safety / HSE reports, inspection reports, permits, etc.).\n\n"
    "Rules:\n"
    "- Answer ONLY from the provided context.\n"
    "- If the answer is not present in the context, reply exactly: "
    "\"Not found in documents\".\n"
    "- Be precise and factual: preserve quantities, units, dates, clause numbers, "
    "drawing numbers, and revisions exactly as written.\n"
    "- Cite every fact inline as [n] using the numbered sources provided.\n"
    "- Give a substantive answer: several well-formed paragraphs or clear bullet lists "
    "when there are multiple points. Explain each major idea in enough detail that a "
    "busy site or office reader can act on it, without repeating the same sentence.\n"
    "- Do NOT invent values, do NOT speculate, do NOT use outside knowledge."
)


_client = None


def _get_gemini_client():
    global _client
    if _client is not None:
        return _client
    api_key = getattr(settings, "GEMINI_API_KEY", "") or ""
    if not api_key:
        return None
    try:
        from google import genai

        _client = genai.Client(api_key=api_key)
        return _client
    except Exception:  # pragma: no cover - best effort
        logger.exception("Failed to initialise Gemini client")
        return None


def _format_contexts(contexts: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, c in enumerate(contexts, start=1):
        title = c.get("document_title") or f"document #{c.get('document_id')}"
        dtype = c.get("document_type") or "document"
        discipline = c.get("discipline") or ""
        revision = c.get("revision") or ""
        meta_bits = [f"type={dtype}", f"page={c.get('page', '?')}"]
        if discipline:
            meta_bits.append(f"discipline={discipline}")
        if revision:
            meta_bits.append(f"rev={revision}")
        header = f"[{i}] {title} ({', '.join(meta_bits)})"
        body = (c.get("text") or "").strip()
        lines.append(f"{header}\n{body}")
    return "\n\n".join(lines)


def _build_user_prompt(
    query: str,
    contexts: list[dict[str, Any]],
    project: Optional[dict[str, Any]] = None,
) -> str:
    project_block = ""
    if project:
        project_block = (
            "Project information:\n"
            f"- Code: {project.get('project_code', '')}\n"
            f"- Name: {project.get('name', '')}\n"
            f"- Client: {project.get('client_name', '')}\n"
            f"- Location: {project.get('location', '')}\n"
            f"- Type: {project.get('project_type', '')}\n"
            f"- Status: {project.get('status', '')}\n\n"
        )
    context_block = _format_contexts(contexts) if contexts else "(no context retrieved)"
    return (
        "You are a construction domain assistant.\n\n"
        "Rules:\n"
        "- Answer ONLY from the context.\n"
        "- If not found, say \"Not found in documents\".\n"
        "- Be precise and factual.\n"
        "- Cite sources inline as [n] matching the numbers below.\n"
        "- Write a thorough answer: aim for multiple sentences or a few short paragraphs; "
        "use bullet points when several distinct requirements or items appear in the sources. "
        "Mention relevant numbers, units, drawing refs, and clauses when they appear.\n\n"
        f"{project_block}"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )


def _fallback_answer(query: str, contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return (
            "No relevant passages were found in this project's documents. "
            "Try uploading the relevant drawings, specifications, BOQ, or contract."
        )
    bullets = []
    for i, c in enumerate(contexts, start=1):
        snippet = (c.get("text") or "")[:900]
        title = c.get("document_title") or f"document #{c.get('document_id')}"
        bullets.append(f"[{i}] {title} (page {c.get('page', '?')}): {snippet}")
    joined = "\n".join(bullets)
    return (
        f"(Answer model unavailable — extractive summary for: \"{query}\")\n\n"
        f"{joined}"
    )


def _generate_ollama(
    prompt: str,
    *,
    max_tokens: int,
) -> Optional[str]:
    base = getattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = (getattr(settings, "OLLAMA_MODEL", "") or "llama3").strip()
    timeout = int(getattr(settings, "OLLAMA_TIMEOUT_SEC", 300))
    url = f"{base}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.35,
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as exc:
        logger.warning("Ollama HTTP error %s: %s", exc.code, exc.read().decode("utf-8", errors="replace")[:500])
        return None
    except urllib.error.URLError as exc:
        logger.warning("Ollama connection failed (is `ollama serve` running?): %s", exc.reason)
        return None
    except Exception:
        logger.exception("Ollama request failed")
        return None

    msg = body.get("message") or {}
    text = (msg.get("content") or "").strip()
    return text or None


def _generate_gemini(prompt: str, *, max_tokens: int) -> Optional[str]:
    client = _get_gemini_client()
    if client is None:
        return None
    model_name = getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")
    try:
        from google.genai import types

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.35,
                max_output_tokens=max_tokens,
            ),
        )
        text = (getattr(response, "text", "") or "").strip()
        return text or None
    except Exception:
        logger.exception("Gemini generate_content failed")
        return None


def generate_answer(
    query: str,
    contexts: list[dict[str, Any]],
    project: Optional[dict[str, Any]] = None,
) -> str:
    """Generate a construction-domain answer via Ollama or Gemini, else extractive fallback."""
    provider = getattr(settings, "LLM_PROVIDER", "gemini") or "gemini"
    provider = str(provider).strip().lower()

    max_tokens_gemini = getattr(settings, "GEMINI_MAX_OUTPUT_TOKENS", 2048)
    try:
        max_tokens_gemini = int(max_tokens_gemini)
    except (TypeError, ValueError):
        max_tokens_gemini = 2048

    max_tokens_ollama = getattr(settings, "OLLAMA_NUM_PREDICT", 2048)
    try:
        max_tokens_ollama = int(max_tokens_ollama)
    except (TypeError, ValueError):
        max_tokens_ollama = 2048

    prompt = _build_user_prompt(query, contexts, project=project)

    if provider == "ollama":
        text = _generate_ollama(prompt, max_tokens=max_tokens_ollama)
        if text:
            logger.info("Ollama answer ok (model=%s)", getattr(settings, "OLLAMA_MODEL", ""))
            return text
        return _fallback_answer(query, contexts)

    text = _generate_gemini(prompt, max_tokens=max_tokens_gemini)
    if text:
        return text
    return _fallback_answer(query, contexts)
