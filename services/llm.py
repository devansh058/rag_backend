"""Gemini-backed answer generation for the construction RAG assistant."""
from __future__ import annotations

import logging
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


def _get_client():
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
        f"(Gemini not configured — returning extractive summary for: \"{query}\")\n\n"
        f"{joined}"
    )


def generate_answer(
    query: str,
    contexts: list[dict[str, Any]],
    project: Optional[dict[str, Any]] = None,
) -> str:
    """Generate a construction-domain answer with Gemini, falling back to an
    extractive summary if Gemini is not configured or errors out."""
    client = _get_client()
    if client is None:
        return _fallback_answer(query, contexts)

    model_name = getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")
    max_tokens = getattr(settings, "GEMINI_MAX_OUTPUT_TOKENS", 2048)
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = 2048
    prompt = _build_user_prompt(query, contexts, project=project)

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
        if not text:
            return _fallback_answer(query, contexts)
        return text
    except Exception:
        logger.exception("Gemini generate_content failed")
        return _fallback_answer(query, contexts)
