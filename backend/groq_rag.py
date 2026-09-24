
"""
Grounded Groq answer layer for Vectorless RAG V7.

The LLM is only allowed to answer from retrieved context.
Citation validation is tolerant to harmless whitespace/case differences,
while still requiring every citation to point to an actually retrieved block.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests


DEFAULT_MODEL = "openai/gpt-oss-120b"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

CITATION_RE = re.compile(
    r"\[Source:\s*(.*?)\s*\|\s*Page:\s*(\d+)"
    r"(?:\s*\|\s*Section:\s*(.*?))?\s*\]",
    flags=re.IGNORECASE,
)


@dataclass
class GroundedAnswer:
    answer: str
    grounded: bool
    citations: List[Dict]
    reason: Optional[str] = None


def _norm(value: str) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _extract_citations(text: str) -> List[Tuple[str, int, str]]:
    found = []
    for source, page, section in CITATION_RE.findall(text or ""):
        found.append((_norm(source), int(page), _norm(section)))
    return found


def _citation_matches(citation: Tuple[str, int, str], results: List[Dict]) -> bool:
    source, page, section = citation
    for r in results:
        if _norm(r.get("source")) != source:
            continue
        if int(r.get("page", -1)) != page:
            continue

        retrieved_section = _norm(r.get("section", "General"))

        # Section is optional for compatibility, but if supplied it must match.
        if section and section != retrieved_section:
            # Accept common canonical/heading variants.
            aliases = {
                "project": "projects",
                "projects": "projects",
                "publication": "publications",
                "publications": "publications",
                "education": "education",
                "technical skill": "technical skills",
                "technical skills": "technical skills",
                "work experience": "work experience",
                "experience": "work experience",
            }
            if aliases.get(section, section) != aliases.get(retrieved_section, retrieved_section):
                continue
        return True
    return False


def _validate_citations(answer: str, results: List[Dict]) -> Tuple[bool, List[Dict]]:
    raw = _extract_citations(answer)
    if not raw:
        return False, []

    citations = []
    for source, page, section in raw:
        if not _citation_matches((source, page, section), results):
            return False, []
        citations.append({
            "source": source,
            "page": page,
            "section": section or None,
        })
    return True, citations


def _system_prompt() -> str:
    return """You are the answer-generation layer of a document-grounded RAG system.

Rules:
1. Use ONLY the retrieved context. Never invent facts.
2. Answer the user's question directly and naturally.
3. The user may ask conversationally: "qualification", "what did he study",
   "what are his projects", "published any paper", "who is the main subject",
   or "what did you analyze from the PDF". Interpret these as document questions.
4. For a profile question, combine evidence from multiple retrieved sections.
5. For a document-summary question, summarize the retrieved representative
   sections and clearly state that the summary is based on the retrieved
   document content.
6. Every factual answer must end with one or more citations using EXACTLY:
   [Source: filename | Page: N | Section: Section Name]
7. Only cite sources/pages/sections that appear in the retrieved context.
8. If the retrieved context does not contain enough evidence, say:
   "I don't have enough information in the retrieved documents to answer that."
   Do not guess.
9. Keep answers concise unless the user asks for detail.
"""


def ask_documents(index, question: str, top_k: int = 5,
                  max_context_chars: int = 6000,
                  api_key: Optional[str] = None,
                  model: Optional[str] = None) -> GroundedAnswer:
    results = index.search(question, top_k=top_k)

    if not results:
        return GroundedAnswer(
            answer="I don't have enough information in the retrieved documents to answer that.",
            grounded=False,
            citations=[],
            reason="no_retrieval_results",
        )

    context = ""
    # Import locally so this module remains usable independently.
    from rag_engine import build_context
    context = build_context(results, max_chars=max_context_chars)

    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return GroundedAnswer(
            answer="GROQ_API_KEY is not set.",
            grounded=False,
            citations=[],
            reason="missing_api_key",
        )

    payload = {
        "model": model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {
                "role": "user",
                "content": f"RETRIEVED CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}",
            },
        ],
        "temperature": 0,
        "max_tokens": 900,
    }

    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=45,
        )
    except requests.RequestException as exc:
        return GroundedAnswer(
            answer=f"Groq request failed: {exc}",
            grounded=False,
            citations=[],
            reason="request_error",
        )

    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        return GroundedAnswer(
            answer=f"Groq API error ({response.status_code}): {detail}",
            grounded=False,
            citations=[],
            reason="api_error",
        )

    try:
        answer = response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return GroundedAnswer(
            answer=f"Invalid Groq response: {exc}",
            grounded=False,
            citations=[],
            reason="invalid_response",
        )

    valid, citations = _validate_citations(answer, results)

    if not valid:
        # A model can occasionally omit a citation even with perfectly valid
        # evidence. Do not return a fabricated citation. Instead provide a
        # deterministic evidence response from the retrieved blocks.
        evidence_lines = []
        for r in results[:3]:
            evidence_lines.append(
                f"{r['text'].strip()} "
                f"[Source: {r['source']} | Page: {r['page']} | Section: {r.get('section', 'General')}]"
            )
        fallback = (
            "The generated answer could not be citation-verified. "
            "Here is the retrieved evidence instead:\n\n"
            + "\n\n".join(evidence_lines)
        )
        return GroundedAnswer(
            answer=fallback,
            grounded=True,
            citations=[
                {
                    "source": r["source"],
                    "page": r["page"],
                    "section": r.get("section", "General"),
                }
                for r in results[:3]
            ],
            reason="citation_fallback",
        )

    return GroundedAnswer(
        answer=answer,
        grounded=True,
        citations=citations,
        reason=None,
    )


# Backward-compatible alias.
def ask_groq(index, question: str, top_k: int = 5,
             max_context_chars: int = 6000,
             api_key: Optional[str] = None,
             model: Optional[str] = None) -> GroundedAnswer:
    return ask_documents(
        index=index,
        question=question,
        top_k=top_k,
        max_context_chars=max_context_chars,
        api_key=api_key,
        model=model,
    )
