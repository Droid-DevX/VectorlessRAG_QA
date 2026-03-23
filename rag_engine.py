"""
Vectorless RAG Pipeline using Page Index
- No vector DB required
- Uses TF-IDF + BM25 for retrieval
- Page-level indexing for precise context
"""

import re
import math
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import io

# ── PDF text extraction 
def extract_pages_from_pdf(file_bytes: bytes) -> List[Dict]:
    """Extract text page-by-page from a PDF."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = clean_text(text)
            if text.strip():
                pages.append({"page": i + 1, "text": text})
        return pages
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")


def extract_pages_from_txt(file_bytes: bytes, chars_per_page: int = 2000) -> List[Dict]:
    """Chunk plain text into 'pages'."""
    text = file_bytes.decode("utf-8", errors="ignore")
    text = clean_text(text)
    chunks = []
    for i in range(0, len(text), chars_per_page):
        chunk = text[i : i + chars_per_page].strip()
        if chunk:
            chunks.append({"page": len(chunks) + 1, "text": chunk})
    return chunks


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    return text.strip()


# ── Tokeniser 
STOP_WORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","he","in",
    "is","it","its","of","on","that","the","to","was","were","will","with",
    "this","but","they","have","had","what","when","where","who","which","or",
    "not","can","do","did","we","our","you","your","i","me","my","their",
    "there","so","if","about","more","been","also","any","all","into","than",
}

def tokenise(text: str) -> List[str]:
    tokens = re.findall(r'[a-z]+', text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


# ── Page Index (TF-IDF + BM25) 
class PageIndex:
    """
    Inverted index over document pages.
    Retrieval uses BM25 (k1=1.5, b=0.75) — no vectors, no embeddings.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.pages: List[Dict] = []          # [{page, text, source, tokens}]
        self.inverted: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
                                              # term -> [(page_idx, freq)]
        self.doc_lengths: List[int] = []
        self.avg_dl: float = 0.0
        self.idf: Dict[str, float] = {}

    def add_document(self, pages: List[Dict], source_name: str):
        """Add extracted pages from one document."""
        for page in pages:
            tokens = tokenise(page["text"])
            idx = len(self.pages)
            self.pages.append({
                "page": page["page"],
                "text": page["text"],
                "source": source_name,
                "tokens": tokens,
            })
            self.doc_lengths.append(len(tokens))
            freq_map: Dict[str, int] = defaultdict(int)
            for t in tokens:
                freq_map[t] += 1
            for term, freq in freq_map.items():
                self.inverted[term].append((idx, freq))
        self._recompute_stats()

    def _recompute_stats(self):
        N = len(self.pages)
        if N == 0:
            return
        self.avg_dl = sum(self.doc_lengths) / N
        self.idf = {}
        for term, postings in self.inverted.items():
            df = len(postings)
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """BM25 search. Returns top-k page dicts with score."""
        q_terms = tokenise(query)
        if not q_terms or not self.pages:
            return []

        scores: Dict[int, float] = defaultdict(float)
        for term in q_terms:
            if term not in self.inverted:
                continue
            idf = self.idf.get(term, 0.0)
            for (doc_idx, tf) in self.inverted[term]:
                dl = self.doc_lengths[doc_idx]
                norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                )
                scores[doc_idx] += idf * norm

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_idx, score in ranked:
            p = self.pages[doc_idx]
            results.append({
                "source": p["source"],
                "page": p["page"],
                "text": p["text"],
                "score": round(score, 4),
            })
        return results

    def stats(self) -> Dict:
        sources = list({p["source"] for p in self.pages})
        return {
            "total_pages": len(self.pages),
            "total_terms": len(self.inverted),
            "sources": sources,
        }

    def clear(self):
        self.pages.clear()
        self.inverted.clear()
        self.doc_lengths.clear()
        self.idf.clear()
        self.avg_dl = 0.0

    def export(self) -> str:
        """Serialize the index to JSON (for session persistence)."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "pages": [
                {k: v for k, v in p.items() if k != "tokens"}
                for p in self.pages
            ],
            "doc_lengths": self.doc_lengths,
            "inverted": {t: lst for t, lst in self.inverted.items()},
            "idf": self.idf,
            "avg_dl": self.avg_dl,
        }
        return json.dumps(data)

    @classmethod
    def load(cls, json_str: str) -> "PageIndex":
        data = json.loads(json_str)
        idx = cls(k1=data["k1"], b=data["b"])
        idx.avg_dl = data["avg_dl"]
        idx.doc_lengths = data["doc_lengths"]
        idx.idf = data["idf"]
        idx.inverted = defaultdict(list, {
            t: [tuple(x) for x in v] for t, v in data["inverted"].items()
        })
        for i, p in enumerate(data["pages"]):
            tokens = tokenise(p.get("text", ""))
            idx.pages.append({**p, "tokens": tokens})
        return idx


# ── Context builder 
def build_context(results: List[Dict], max_chars: int = 4000) -> str:
    parts = []
    total = 0
    for r in results:
        snippet = f"[Source: {r['source']} | Page {r['page']} | Score: {r['score']}]\n{r['text']}"
        if total + len(snippet) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                snippet = snippet[:remaining] + "…"
                parts.append(snippet)
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts)
