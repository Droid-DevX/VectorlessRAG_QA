"""FastAPI backend for the Vectorless RAG React UI."""
from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).with_name(".env"))

from rag_engine import (
    PageIndex,
    extract_pages_from_pdf,
    extract_pages_from_txt,
)
from groq_rag import ask_documents

app = FastAPI(title="Vectorless RAG API", version="1.0.0")


allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Production frontend URL from environment variable
frontend_url = os.getenv("FRONTEND_URL", "").strip()

if frontend_url:
    allowed_origins.extend(
        origin.strip()
        for origin in frontend_url.split(",")
        if origin.strip()
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(allowed_origins)),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

index = PageIndex()
indexed_files: set[str] = set()
lock = Lock()


class QueryRequest(BaseModel):
    query: str


@app.get("/api/stats")
def stats():
    return index.stats()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()

    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    with lock:
        if filename in indexed_files:
            return {"source": filename, "pages": 0, "already_indexed": True}

        raw = await file.read()

        try:
            if suffix == ".pdf":
                pages = extract_pages_from_pdf(raw)
            else:
                pages = extract_pages_from_txt(raw)

            if not pages:
                raise ValueError("No readable content was extracted from the file.")

            index.add_document(pages, filename)
            indexed_files.add(filename)

            return {
                "source": filename,
                "pages": len(pages),
                "chunks": index.stats()["total_chunks"],
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to index {filename}: {exc}") from exc


@app.post("/api/query")
def query(payload: QueryRequest):
    question = payload.query.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if index.stats()["total_pages"] == 0:
        raise HTTPException(status_code=400, detail="Upload and index at least one document first.")

    # Retrieve evidence separately for the React UI.  GroundedAnswer does
    # not expose a `.retrieved` attribute; the current RAG engine owns the
    # retrieval results.
    retrieved = index.search(
        question,
        top_k=5,
        candidate_k=20,
    )

    # Generate the grounded answer from the same indexed document.
    try:
        result = ask_documents(index, question, top_k=5)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {exc}",
        ) from exc

    # Keep the UI contract simple while exposing section/chunk metadata.
    results = [
        {
            "source": r.get("source", ""),
            "page": r.get("page", 0),
            "section": r.get("section", "General"),
            "chunk": r.get("chunk", 1),
            "score": r.get("rerank_score", r.get("score", 0.0)),
            "text": r.get("text", ""),
        }
        for r in retrieved
    ]

    return {
        "answer": result.answer,
        "grounded": result.grounded,
        "citations": result.citations,
        "reason": result.reason,
        "results": results,
    }


@app.post("/api/clear")
def clear():
    with lock:
        index.clear()
        indexed_files.clear()
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok"}
