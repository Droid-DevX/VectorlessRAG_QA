# Vectorless RAG - Semantic Document Search

A full-stack, **vector-database-free RAG system** for uploading
documents, retrieving relevant evidence, and generating grounded
answers.

The project uses a custom **BM25 retrieval engine** instead of
embeddings/vector storage. Documents are extracted into page- and
section-aware chunks, indexed locally, and retrieved using lexical
relevance with deterministic query/section routing. A Groq-hosted LLM
then generates the final answer strictly from the retrieved evidence.

##  Features

-   PDF and TXT document upload
-   Custom BM25 inverted-index retrieval
-   Page-aware and section-aware chunking
-   Query intent routing for Technical Skills, Work Experience,
    Projects, Publications, Education, Certifications, and Achievements

-   Source/page/section citations
-   Fast local retrieval without a vector database
-   React + TypeScript frontend
-   FastAPI backend
-   Clear/reset indexed documents
-   Live index statistics

##  Architecture

``` text
React + TypeScript UI
        │ REST API
        ▼
FastAPI
        │
        ▼
rag_engine.py
PDF/TXT extraction
→ section detection + chunking
→ tokenization
→ inverted index
→ BM25 retrieval
→ section-aware reranking
        │
        ▼
groq_rag.py
retrieved context → Groq LLM → grounded answer + citation
```

### Core pipeline

``` text
User Query
    ↓
Query Intent Detection
    ↓
BM25 Candidate Retrieval
    ↓
Section-aware Reranking
    ↓
Top Evidence Chunks
    ↓
Grounded LLM Generation
    ↓
Answer + Source/Page/Section Citation
```

##  Project Structure

``` text
Vectorless_RAG/
│
├── backend/
│   ├── api.py                 # FastAPI REST API
│   ├── rag_engine.py          # Custom BM25 retrieval engine
│   ├── groq_rag.py            # Grounded Groq generation + citation validation
│   ├── test_rag.py            # Retrieval/backend tests
│   ├── requirements.txt       # Backend dependencies
│   └── .env                   # Local API credentials (not committed)
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Main application UI
│   │   ├── rag_ui.tsx         # RAG interface components
│   │   ├── index.css          # Global styling
│   │   └── main.tsx           # React entry point
│   ├── public/
│   ├── package.json
│   ├── vite.config.ts
│   └── index.html
│
├── README.md
└── .gitignore
```

##  Retrieval Engine

The retrieval layer is intentionally **vectorless**.

The custom engine uses an inverted index and BM25 scoring with:

``` text
k1 = 1.5
b  = 0.75
```

The retrieval pipeline:

1.  Extract document text.
2.  Detect document sections.
3.  Split content into retrieval chunks.
4.  Tokenize chunks.
5.  Build an inverted index.
6.  Calculate BM25 relevance.
7.  Detect query intent.
8.  Apply section-aware reranking.
9.  Return the strongest evidence chunks.

No embedding model or vector database is required for retrieval.

##  Section-Aware Retrieval

Short questions can have very little lexical overlap with the actual
document text. The engine therefore detects likely section intent.

  Query                                      Detected Intent
  ------------------------------------------ ------------------
  `What are his projects?`                   Projects
  `What is his qualification?`               Education
  `What did he study?`                       Education
  `What are his technical skills?`           Technical Skills
  `Has he published a paper?`                Publications
  `Where did he work?`                       Work Experience
  `What does he do?`                         Work Experience
  `Tell me what you analyzed from the PDF`   Document Summary

##  Grounded Generation

`backend/groq_rag.py` sends only retrieved document evidence to the Groq
API.

The generation layer is instructed to:

-   answer only from retrieved evidence;
-   avoid unsupported facts;
-   handle short/conversational questions;
-   include source citations;
-   avoid fabricated citations.

Expected citation format:

``` text
[Source: Resume.pdf | Page: 1 | Section: Education]
```

Generated citations are validated against the retrieved document
metadata. If the answer cannot be citation-verified, the backend can
fall back to retrieved evidence rather than inventing a citation.

##  Frontend

The frontend uses React, TypeScript, and Vite.

The UI communicates with:

``` text
/api/stats
/api/upload
/api/query
/api/clear
```

The interface displays document statistics, upload state, generated
answers, retrieved evidence, and source/page/section information.

##  Requirements

### Backend

-   Python 3.10+
-   FastAPI
-   Uvicorn
-   PyMuPDF
-   Requests
-   python-dotenv
-   Pydantic

Install:

``` powershell
cd backend
pip install -r requirements.txt
```

### Frontend

-   Node.js
-   npm

Install:

``` powershell
cd frontend
npm install
```

##  Environment Variables

Create:

``` text
backend/.env
```

and add:

``` env
GROQ_API_KEY=gsk_your_api_key_here
```

Never commit `.env` or API keys to GitHub.

##  Run the Project

### 1. Start FastAPI

From `backend`:

``` powershell
python -m uvicorn api:app --reload
```

Backend:

``` text
http://127.0.0.1:8000
```

Health check:

``` text
http://127.0.0.1:8000/health
```

Expected:

``` json
{"status": "ok"}
```

### 2. Start React

In another terminal:

``` powershell
cd frontend
npm install
npm run dev
```

Vite normally serves the frontend at:

``` text
http://localhost:5173
```

## API Endpoints

### `GET /api/stats`

Returns the current index statistics.

### `POST /api/upload`

Uploads and indexes a PDF or TXT file using multipart form data:

``` text
file=<document>
```

### `POST /api/query`

Request:

``` json
{
  "query": "What are his projects?"
}
```

Response:

``` json
{
  "answer": "...",
  "grounded": true,
  "citations": [],
  "reason": null,
  "results": [
    {
      "source": "Resume.pdf",
      "page": 1,
      "section": "Projects",
      "chunk": 1,
      "score": 5.8,
      "text": "..."
    }
  ]
}
```

### `POST /api/clear`

Clears the current in-memory document index.

##  Example Queries

``` text
What is the qualification?
What did he study?
What are his projects?
What are his technical skills?
Where did he work?
Has he published any paper?
What does he do?
Tell me what you analyzed from the PDF.
```

##  Why Vectorless?

Traditional RAG commonly follows:

``` text
Document
   ↓
Embeddings
   ↓
Vector Database
   ↓
Similarity Search
   ↓
LLM
```

This project follows:

``` text
Document
   ↓
Text Extraction
   ↓
Tokenization
   ↓
Inverted Index
   ↓
BM25
   ↓
Section-aware Reranking
   ↓
LLM
```

### Advantages

-   No vector database.
-   No embedding model required for retrieval.
-   Minimal storage overhead.
-   Fast indexing for small/medium document collections.
-   Transparent retrieval scores.
-   Easy to inspect retrieved evidence.
-   Exact source/page/section metadata can be preserved.

### Trade-off

Lexical retrieval depends on terminology overlap and query wording.
Section-aware routing and query expansion are used to improve
conversational queries with weak direct keyword overlap.

