
"""
Vectorless RAG Engine V7
-------------------------
BM25 retrieval + section-aware reranking for resume/document Q&A.

Design goals:
- No embeddings / vector database.
- Chunk-level BM25 retrieval.
- Explicit intent routing for resumes/documents.
- Reliable section detection.
- Profile and whole-document summary modes.
- Stable source/page/section/chunk metadata.
"""

import io
import json
import math
import re
import hashlib
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages_from_pdf(file_bytes: bytes) -> List[Dict]:
    try:
        import pypdf
    except ImportError as exc:
        raise ImportError("pypdf not installed. Run: pip install pypdf") from exc

    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append({"page": page_number, "text": text})
    return pages


def extract_pages_from_txt(file_bytes: bytes, chars_per_page: int = 2000) -> List[Dict]:
    text = clean_text(file_bytes.decode("utf-8", errors="replace"))
    if not text:
        return []

    pages = []
    for start in range(0, len(text), chars_per_page):
        chunk = text[start:start + chars_per_page].strip()
        if chunk:
            pages.append({"page": len(pages) + 1, "text": chunk})
    return pages


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","he","her",
    "his","in","is","it","its","of","on","that","the","to","was","were","will",
    "with","this","but","they","have","had","what","when","where","who","which",
    "or","not","can","do","did","we","our","you","your","i","me","my","their",
    "there","so","if","about","more","been","also","any","all","into","than",
    "then","these","those","how","why","does","doesnt","didnt","dont","very",
    "tell","u","please","is","are",
}

TOKEN_PATTERN = re.compile(
    r"""
    (?:
        [A-Za-z]+(?:[-'][A-Za-z0-9]+)*[0-9]+(?:\.[0-9]+)* |
        [0-9]+(?:[-.][A-Za-z0-9]+)+ |
        [A-Za-z]+[0-9]+(?:[-.][A-Za-z0-9]+)* |
        [A-Za-z]+(?:[-'][A-Za-z]+)* |
        [0-9]+
    )
    """,
    re.VERBOSE,
)


def _stem(token: str) -> str:
    if len(token) <= 4:
        return token
    for suffix in ("ments", "ings", "ation", "ions", "edly", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token


def tokenise(text: str) -> List[str]:
    result = []
    for m in TOKEN_PATTERN.finditer(text):
        token = m.group(0).casefold()
        if token in STOP_WORDS:
            continue
        if len(token) <= 1 and not token.isdigit():
            continue
        result.append(_stem(token))
    return result


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

SECTION_ALIASES = {
    "Technical Skills": {"technical skills", "skills", "technical skill"},
    "Work Experience": {"work experience", "experience", "professional experience",
                         "technical experience", "employment"},
    "Projects": {"projects", "project", "research & projects", "research and projects"},
    "Publications": {"publications", "publication", "papers", "paper", "research papers"},
    "Education": {"education", "academic background"},
    "Certifications": {"certifications", "certification", "certificates"},
    "Achievements": {"achievements", "achievement", "awards", "awards & honors"},
    "Summary": {"summary", "profile", "about"},
    "Research Interests": {"research interests", "research interest"},
}

COMMON_HEADINGS = {
    alias.casefold()
    for aliases in SECTION_ALIASES.values()
    for alias in aliases
}


def canonical_section(raw: str) -> str:
    normalized = re.sub(r"\s+", " ", raw).strip().rstrip(":").casefold()
    for canonical, aliases in SECTION_ALIASES.items():
        if normalized in {a.casefold() for a in aliases}:
            return canonical
    return raw.strip().rstrip(":") or "General"


def _looks_like_heading(line: str) -> bool:
    line = re.sub(r"\s+", " ", line).strip()
    if not line or len(line) > 90:
        return False
    normalized = line.casefold().rstrip(":").strip()
    if normalized in COMMON_HEADINGS:
        return True
    # ALL-CAPS resume headings
    letters = [c for c in line if c.isalpha()]
    if letters and len(line.split()) <= 7:
        ratio = sum(c.isupper() for c in letters) / len(letters)
        if ratio > 0.78 and not re.search(r"[.!?]$", line):
            return True
    return False


def sectionize_page(text: str) -> List[Dict]:
    """
    Detect headings from actual line boundaries first. PDF extractors sometimes
    collapse whitespace, so known headings are also isolated when they occur
    inline. The regex deliberately uses a single backslash word boundary.
    """
    text = clean_text(text)

    # Insert boundaries around known multi-word headings only when they occur
    # as standalone phrases. Longest first prevents "skills" from splitting
    # "technical skills".
    heading_phrases = sorted(COMMON_HEADINGS, key=len, reverse=True)
    for heading in heading_phrases:
        pattern = re.compile(
            rf"(?<![A-Za-z])({re.escape(heading)})(?=\s|:|$)",
            flags=re.IGNORECASE,
        )
        text = pattern.sub(lambda m: "\n" + m.group(1) + "\n", text)

    lines = [x.strip() for x in text.splitlines() if x.strip()]

    sections = []
    current_name = "General"
    current_lines = []

    for line in lines:
        if _looks_like_heading(line):
            if current_lines:
                sections.append({
                    "section": canonical_section(current_name),
                    "text": "\n".join(current_lines).strip(),
                })
                current_lines = []
            current_name = line.rstrip(":").strip()
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "section": canonical_section(current_name),
            "text": "\n".join(current_lines).strip(),
        })

    return [s for s in sections if s["text"]]


def chunk_text(text: str, target_chars: int = 1200, overlap_chars: int = 160) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(para) > target_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(para):
                end = min(start + target_chars, len(para))
                piece = para[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(para):
                    break
                start = max(start + 1, end - overlap_chars)
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= target_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            tail = current[-overlap_chars:].strip() if current else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        chunks.append(current.strip())

    # Deduplicate exact normalized chunks.
    seen = set()
    result = []
    for chunk in chunks:
        key = re.sub(r"\s+", " ", chunk).strip().casefold()
        if key and key not in seen:
            seen.add(key)
            result.append(chunk)
    return result


def chunk_page_sections(text: str, target_chars: int = 1200,
                        overlap_chars: int = 160) -> List[Dict]:
    sections = sectionize_page(text)
    if not sections:
        return []

    output = []
    for section in sections:
        for chunk in chunk_text(section["text"], target_chars, overlap_chars):
            output.append({"section": section["section"], "text": chunk})
    return output


# ---------------------------------------------------------------------------
# Intent routing
# ---------------------------------------------------------------------------

INTENT_TERMS = {
    "Education": [
        "education", "degree", "qualification", "qualifications", "university",
        "college", "coursework", "course", "study", "studied", "studies",
        "studying", "major", "student", "academic", "academics",
    ],
    "Projects": [
        "project", "projects", "built", "developed", "developing",
        "worked on", "work on", "project name", "project names",
    ],
    "Publications": [
        "publication", "publications", "paper", "papers", "published",
        "publication", "manuscript", "research paper", "conference",
        "journal", "authored", "co-authored", "first author",
    ],
    "Technical Skills": [
        "technical skills", "skills", "technologies", "technology", "tech stack",
        "programming languages", "tools", "frameworks", "software", "languages",
    ],
    "Work Experience": [
        "work experience", "experience", "employment", "internship",
        "internships", "job", "jobs", "role", "roles", "worked", "working",
    ],
    "Certifications": [
        "certification", "certifications", "certificate", "certificates",
    ],
    "Achievements": [
        "achievement", "achievements", "award", "awards", "honor", "honors",
        "winner", "won", "rank", "position",
    ],
}


def detect_intent(query: str) -> str:
    q = query.casefold().strip()

    # Summary/document-analysis requests.
    summary_phrases = (
        "what did you analyze", "what have you analyzed",
        "what is in the pdf", "what's in the pdf",
        "what is in this document", "what's in this document",
        "tell me what you analyzed", "tell me what u analyzed", "summarize the document",
        "summarize the pdf", "summarise the document", "summarise the pdf",
        "tell me about the document", "tell me about this resume",
        "what does this resume contain", "what does the resume contain",
    )
    if any(p in q for p in summary_phrases) or re.search(
        r"what\s+(?:did|have)\s+(?:you|u)\s+analy[sz]ed\s+from\s+(?:the\s+)?(?:pdf|document|resume)",
        q,
    ):
        return "Document Summary"

    # Profile/person questions should retrieve identity + education + work +
    # projects + publications, rather than only one section.
    profile_phrases = (
        "who is", "main subject", "who's", "tell me about ayush",
        "what does he do", "what does ayush do", "what does he study",
        "what is he", "what is ayush",
    )
    if any(p in q for p in profile_phrases):
        return "Profile"

    best = None
    best_score = 0
    for intent, terms in INTENT_TERMS.items():
        score = 0
        for term in terms:
            if term in q:
                score += 3 if " " in term else 1
        if score > best_score:
            best = intent
            best_score = score
    return best or "General"


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

class PageIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75,
                 chunk_size: int = 1200, chunk_overlap: int = 160):
        self.k1 = k1
        self.b = b
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pages: List[Dict] = []
        self.inverted = defaultdict(list)
        self.doc_lengths = []
        self.avg_dl = 0.0
        self.idf = {}

    @staticmethod
    def document_id(source_name: str, raw_text: str = "") -> str:
        payload = f"{source_name}\0{raw_text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def add_document(self, pages: List[Dict], source_name: str) -> int:
        document_id = self.document_id(
            source_name, "\n".join(p.get("text", "") for p in pages)
        )
        added = 0

        for page in pages:
            page_num = int(page["page"])
            text = clean_text(page.get("text", ""))
            if not text:
                continue

            chunks = chunk_page_sections(text, self.chunk_size, self.chunk_overlap)
            for chunk_num, item in enumerate(chunks, 1):
                section = item["section"]
                chunk_text = item["text"]

                # Section title is indexed too, which makes exact intent
                # queries very strong.
                tokens = tokenise(chunk_text) + tokenise(section)
                if not tokens:
                    continue

                idx = len(self.pages)
                self.pages.append({
                    "id": f"{document_id}:p{page_num}:c{chunk_num}",
                    "document_id": document_id,
                    "source": source_name,
                    "page": page_num,
                    "chunk": chunk_num,
                    "section": section,
                    "text": chunk_text,
                    "tokens": tokens,
                })
                self.doc_lengths.append(len(tokens))

                for term, freq in Counter(tokens).items():
                    self.inverted[term].append((idx, freq))
                added += 1

        self._recompute_stats()
        return added

    def _recompute_stats(self):
        n = len(self.pages)
        if not n:
            self.avg_dl = 0.0
            self.idf = {}
            return
        self.avg_dl = sum(self.doc_lengths) / n
        self.idf = {
            term: math.log(1.0 + (n - len(postings) + 0.5) /
                           (len(postings) + 0.5))
            for term, postings in self.inverted.items()
        }

    def _bm25_scores(self, query: str) -> Dict[int, float]:
        scores = defaultdict(float)
        q_terms = set(tokenise(query))
        for term in q_terms:
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for idx, tf in postings:
                dl = self.doc_lengths[idx]
                denom = tf + self.k1 * (
                    1 - self.b + self.b * dl / max(self.avg_dl, 1e-9)
                )
                scores[idx] += idf * ((tf * (self.k1 + 1)) / max(denom, 1e-9))
        return scores

    def _section_matches(self, section: str, intent: str) -> bool:
        return canonical_section(section).casefold() == intent.casefold()

    def _profile_search(self, top_k: int) -> List[Dict]:
        preferred = {
            "General": 3.0,
            "Education": 2.8,
            "Work Experience": 2.6,
            "Projects": 2.5,
            "Publications": 2.3,
            "Technical Skills": 2.0,
            "Research Interests": 1.8,
        }
        candidates = []
        for idx, item in enumerate(self.pages):
            section = canonical_section(item.get("section", "General"))
            boost = preferred.get(section, 1.0)
            # First chunks of each section are generally the most informative.
            if item["chunk"] == 1:
                boost += 0.4
            candidates.append((boost, idx))

        candidates.sort(reverse=True)
        return [self._result(self.pages[idx], score, "Profile")
                for score, idx in candidates[:top_k]]

    def _summary_search(self, top_k: int) -> List[Dict]:
        preferred = {
            "General": 3.0,
            "Work Experience": 2.7,
            "Projects": 2.7,
            "Publications": 2.5,
            "Education": 2.2,
            "Technical Skills": 2.2,
            "Research Interests": 2.0,
        }
        # For a one-page resume, return representative first chunks from
        # important sections. For longer documents, rank section coverage.
        candidates = []
        seen_sections = set()
        for idx, item in enumerate(self.pages):
            section = canonical_section(item.get("section", "General"))
            score = preferred.get(section, 1.0)
            if section not in seen_sections:
                score += 1.0
            if item["chunk"] == 1:
                score += 0.5
            candidates.append((score, idx, section))

        candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        selected = []
        # Coverage first: one representative chunk per section.
        for score, idx, section in candidates:
            if section not in seen_sections:
                selected.append(self._result(self.pages[idx], score, "Document Summary"))
                seen_sections.add(section)
                if len(selected) >= top_k:
                    break

        # Fill remaining slots by score.
        if len(selected) < top_k:
            selected_ids = {r["id"] for r in selected}
            for score, idx, section in candidates:
                r = self._result(self.pages[idx], score, "Document Summary")
                if r["id"] not in selected_ids:
                    selected.append(r)
                    selected_ids.add(r["id"])
                    if len(selected) >= top_k:
                        break
        return selected

    def _result(self, item: Dict, score: float, intent: str) -> Dict:
        return {
            "id": item["id"],
            "document_id": item["document_id"],
            "source": item["source"],
            "page": item["page"],
            "chunk": item["chunk"],
            "section": canonical_section(item.get("section", "General")),
            "text": item["text"],
            "score": round(float(score), 4),
            "rerank_score": round(float(score), 4),
            "section_intent": intent,
            "matched_terms": [],
        }

    def search(self, query: str, top_k: int = 5, candidate_k: Optional[int] = None,
               min_score: float = 0.0, diversify: bool = True) -> List[Dict]:
        if not self.pages or not query.strip():
            return []

        intent = detect_intent(query)

        if intent == "Profile":
            return self._profile_search(max(top_k, 6))[:top_k]

        if intent == "Document Summary":
            return self._summary_search(max(top_k, 7))[:top_k]

        scores = self._bm25_scores(query)
        candidate_k = candidate_k or max(top_k * 5, 20)

        # If intent is explicit, section matching gets a strong deterministic
        # boost. This fixes short queries like "qualification" and "papers".
        ranked = []
        used = set()

        for idx, base_score in scores.items():
            item = self.pages[idx]
            section = canonical_section(item.get("section", "General"))
            boost = 0.0
            if intent != "General" and self._section_matches(section, intent):
                boost += 5.0
                if item["chunk"] == 1:
                    boost += 0.8
            final = base_score + boost
            ranked.append((final, base_score, idx))
            used.add(idx)

        # Explicit intent must be able to retrieve the target section even if
        # the user's exact word (e.g. "qualification") does not occur inside
        # the section body. This is the key fix for short resume questions.
        if intent not in ("General", "Profile", "Document Summary"):
            for idx, item in enumerate(self.pages):
                if idx in used:
                    continue
                if self._section_matches(item.get("section", "General"), intent):
                    boost = 5.0 + (0.8 if item["chunk"] == 1 else 0.0)
                    ranked.append((boost, 0.0, idx))

        ranked.sort(reverse=True)

        results = []
        seen = set()
        page_counts = defaultdict(int)

        for final, base, idx in ranked[:candidate_k]:
            if final < min_score:
                continue
            item = self.pages[idx]
            key = re.sub(r"\s+", " ", item["text"]).strip().casefold()
            if key in seen:
                continue

            page_key = (item["source"], item["page"])
            if diversify and page_counts[page_key] >= 2:
                continue

            result = self._result(item, final, intent)
            result["score"] = round(float(base), 4)
            result["rerank_score"] = round(float(final), 4)
            results.append(result)
            seen.add(key)
            page_counts[page_key] += 1
            if len(results) >= top_k:
                break

        return results

    def stats(self) -> Dict:
        return {
            "total_pages": len({(x["source"], x["page"]) for x in self.pages}),
            "total_chunks": len(self.pages),
            "total_terms": len(self.inverted),
            "sources": sorted({x["source"] for x in self.pages}),
            "documents": len({x["document_id"] for x in self.pages}),
        }

    def clear(self):
        self.pages.clear()
        self.inverted.clear()
        self.doc_lengths.clear()
        self.idf.clear()
        self.avg_dl = 0.0

    def export(self) -> str:
        data = {
            "version": 7,
            "k1": self.k1,
            "b": self.b,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "pages": [{k: v for k, v in p.items() if k != "tokens"} for p in self.pages],
            "doc_lengths": self.doc_lengths,
            "inverted": {k: v for k, v in self.inverted.items()},
            "idf": self.idf,
            "avg_dl": self.avg_dl,
        }
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def load(cls, json_str: str) -> "PageIndex":
        data = json.loads(json_str)
        idx = cls(
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
            chunk_size=data.get("chunk_size", 1200),
            chunk_overlap=data.get("chunk_overlap", 160),
        )
        idx.pages = []
        for p in data.get("pages", []):
            p = dict(p)
            p["tokens"] = tokenise(p.get("text", "")) + tokenise(p.get("section", ""))
            idx.pages.append(p)
        idx.doc_lengths = data.get("doc_lengths", [])
        idx.inverted = defaultdict(list, {
            k: [tuple(x) for x in v] for k, v in data.get("inverted", {}).items()
        })
        idx.idf = data.get("idf", {})
        idx.avg_dl = data.get("avg_dl", 0.0)
        return idx


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

def build_context(results: List[Dict], max_chars: int = 6000,
                  include_scores: bool = False) -> str:
    if not results:
        return ""

    blocks = []
    total = 0
    for rank, r in enumerate(results, 1):
        score = f" | BM25: {r['score']}" if include_scores else ""
        header = (
            f"[Retrieved {rank} | Source: {r['source']} | Page: {r['page']} | "
            f"Section: {r.get('section', 'General')} | Chunk: {r.get('chunk', 1)}{score}]"
        )
        block = header + "\n" + r["text"]
        sep = "\n\n---\n\n"
        if total + len(block) + (len(sep) if blocks else 0) > max_chars:
            remaining = max_chars - total - (len(sep) if blocks else 0)
            if remaining > len(header) + 100:
                block = block[:remaining].rstrip() + "\n[truncated]"
                blocks.append(block)
            break
        blocks.append(block)
        total += len(block) + (len(sep) if blocks else 0)
    return "\n\n---\n\n".join(blocks)
