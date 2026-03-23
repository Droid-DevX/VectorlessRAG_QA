"""
Vectorless RAG Pipeline — Streamlit App
BM25 Page-Index retrieval + Groq LLaMA generation
"""

import os
import streamlit as st
import requests
from rag_engine import (
    PageIndex,
    extract_pages_from_pdf,
    extract_pages_from_txt,
    build_context,
)
api_key = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="RAG · Semantic Search",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
  font-family: 'DM Sans', sans-serif !important;
  background: #F7F4EE !important;
  color: #1a1814;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #1C1917 !important;
  border-right: none !important;
  padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 0 !important;
}
[data-testid="stSidebar"] * { color: #E8E0D4 !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label { color: #9A9185 !important; }

/* sidebar inner scroll container */
[data-testid="stSidebarContent"] {
  padding: 2rem 1.5rem !important;
}

/* ── Sidebar brand ── */
.sidebar-brand {
  padding-bottom: 1.5rem;
  border-bottom: 1px solid #2E2A26;
  margin-bottom: 1.5rem;
}
.sidebar-title {
  font-family: 'DM Serif Display', serif;
  font-size: 1.55rem;
  color: #F7F4EE !important;
  line-height: 1.15;
  margin: 0 0 0.3rem 0;
}
.sidebar-title em {
  font-style: italic;
  color: #D4A853 !important;
}
.sidebar-sub {
  font-size: 0.68rem;
  color: #5C5549 !important;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-weight: 500;
}

/* ── Stat pills ── */
.stat-row {
  display: flex;
  gap: 8px;
  margin: 0 0 1.2rem 0;
}
.stat-pill {
  flex: 1;
  background: #252220;
  border: 1px solid #2E2A26;
  border-radius: 10px;
  padding: 10px 8px;
  text-align: center;
}
.stat-num {
  font-family: 'DM Serif Display', serif;
  font-size: 1.6rem;
  color: #D4A853 !important;
  line-height: 1;
  display: block;
}
.stat-lbl {
  font-size: 0.6rem;
  color: #5C5549 !important;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-top: 3px;
  display: block;
}

/* ── Source tags ── */
.src-tag {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: #1E2E1E;
  border: 1px solid #2E4A2E;
  color: #6FBA6F !important;
  border-radius: 6px;
  font-size: 0.65rem;
  font-weight: 600;
  padding: 3px 9px;
  margin: 2px 0;
  letter-spacing: 0.04em;
  word-break: break-all;
}
.src-tag::before { content: "✓"; font-size: 0.7rem; }
.no-docs {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: #2E1E1E;
  border: 1px solid #4A2E2E;
  color: #BA6F6F !important;
  border-radius: 6px;
  font-size: 0.65rem;
  font-weight: 600;
  padding: 3px 9px;
  letter-spacing: 0.04em;
}

/* ── Sidebar divider ── */
.sdiv { border: none; border-top: 1px solid #2E2A26; margin: 1.2rem 0; }

/* ── Sidebar section label ── */
.slabel {
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: #5C5549 !important;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

/* ── File uploader override ── */
div[data-testid="stFileUploader"] {
  background: #252220 !important;
  border: 1.5px dashed #3E3830 !important;
  border-radius: 12px !important;
  padding: 0.2rem !important;
}
div[data-testid="stFileUploader"] * { color: #9A9185 !important; }
div[data-testid="stFileUploader"] small { color: #5C5549 !important; }

/* ── Slider overrides (sidebar) ── */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
  background: #D4A853 !important;
  border-color: #D4A853 !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
  color: #D4A853 !important;
}

/* ── Clear button ── */
[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  color: #5C5549 !important;
  border: 1px solid #2E2A26 !important;
  border-radius: 8px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.75rem !important;
  font-weight: 500 !important;
  padding: 0.4rem 0.8rem !important;
  transition: all 0.2s !important;
  width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  border-color: #BA6F6F !important;
  color: #BA6F6F !important;
  background: #2E1E1E !important;
}

/* ── Main area ── */
.main .block-container {
  max-width: 820px !important;
  padding: 3rem 2.5rem 4rem !important;
}

/* ── Page header ── */
.page-header {
  margin-bottom: 2.5rem;
}
.page-eyebrow {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: #9A8E7E;
  font-weight: 600;
  margin-bottom: 0.5rem;
}
.page-title {
  font-family: 'DM Serif Display', serif;
  font-size: 2.6rem;
  line-height: 1.1;
  color: #1a1814;
  margin: 0 0 0.6rem 0;
}
.page-title em {
  font-style: italic;
  color: #8B6914;
}
.page-desc {
  font-size: 0.88rem;
  color: #6B6157;
  line-height: 1.6;
  max-width: 520px;
  font-weight: 300;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1.5px solid #E0D8CC !important;
  gap: 0 !important;
  margin-bottom: 1.8rem !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: #9A8E7E !important;
  padding: 0.6rem 1.2rem !important;
  background: transparent !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -1.5px !important;
  letter-spacing: 0.01em;
}
.stTabs [aria-selected="true"] {
  color: #1a1814 !important;
  border-bottom-color: #1a1814 !important;
  background: transparent !important;
}

/* ── Chat elements ── */
[data-testid="stChatMessage"] {
  background-color: transparent !important;
}
[data-testid="stChatMessage"] * {
  color: #1a1814 !important;
}
[data-testid="stChatInput"] {
  border-radius: 12px !important;
  background-color: #ffffff !important;
  border: 1px solid #E0D8CC !important;
}
[data-testid="stChatInput"] * {
  color: #1a1814 !important;
  background-color: transparent !important;
}
[data-testid="stChatInputSubmitButton"] svg {
  fill: #1a1814 !important;
  stroke: #1a1814 !important;
}

/* ── Empty state ── */
.empty-state {
  text-align: center;
  padding: 4rem 2rem;
  color: #B8AFA4;
}
.empty-icon {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  opacity: 0.4;
}
.empty-text {
  font-size: 0.9rem;
  font-weight: 300;
  line-height: 1.6;
  font-style: italic;
  font-family: 'DM Serif Display', serif;
}

/* ── Source result cards ── */
.res-card {
  background: #fff;
  border: 1px solid #E8E0D4;
  border-radius: 14px;
  padding: 16px 20px;
  margin: 10px 0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  transition: box-shadow 0.2s;
}
.res-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.res-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.res-source {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #8B6914;
}
.res-score {
  background: #FBF5E6;
  border: 1px solid #E8D5A0;
  color: #8B6914;
  font-size: 0.62rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 20px;
  letter-spacing: 0.06em;
}
.res-page {
  font-size: 0.68rem;
  color: #9A8E7E;
  margin-bottom: 8px;
  font-weight: 500;
}
.res-text {
  font-size: 0.82rem;
  color: #4A4238;
  line-height: 1.65;
  font-weight: 300;
}

/* ── Warning/info overrides ── */
.stAlert {
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.84rem !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #D4A853 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D0C8BC; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Session state 
if "index" not in st.session_state:
    st.session_state.index = PageIndex()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()


# ── LLM (Groq) 
def ask_groq(system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "⚠️ GROQ_API_KEY not set. In PowerShell: $env:GROQ_API_KEY='gsk_...'"
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 1000,
            },
            timeout=30,
        )
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Error: {e}"


# ── Sidebar 
with st.sidebar:
    stats = st.session_state.index.stats()

    st.markdown(f"""
    <div class="sidebar-brand">
      <div class="sidebar-title">Vectorless<br><em>RAG</em></div>
      <div class="sidebar-sub">BM25 · Page Index · No Embeddings</div>
    </div>

    <div class="stat-row">
      <div class="stat-pill">
        <span class="stat-num">{stats['total_pages']}</span>
        <span class="stat-lbl">Pages</span>
      </div>
      <div class="stat-pill">
        <span class="stat-num">{stats['total_terms']:,}</span>
        <span class="stat-lbl">Terms</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if stats["sources"]:
        for s in stats["sources"]:
            st.markdown(f'<div class="src-tag">{s[:30]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="no-docs">No documents indexed</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.markdown('<div class="slabel">Upload documents</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.indexed_files:
                with st.spinner(f"Indexing {f.name}…"):
                    try:
                        raw = f.read()
                        pages = extract_pages_from_pdf(raw) if f.name.lower().endswith(".pdf") else extract_pages_from_txt(raw)
                        st.session_state.index.add_document(pages, f.name)
                        st.session_state.indexed_files.add(f.name)
                        st.success(f"✓ {f.name} — {len(pages)} pages indexed")
                    except Exception as e:
                        st.error(f"Failed: {e}")
                st.rerun()

    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)

    if st.button("↺  Clear everything"):
        st.session_state.index.clear()
        st.session_state.chat_history.clear()
        st.session_state.last_results.clear()
        st.session_state.indexed_files.clear()
        st.rerun()


# ── Main area 
st.markdown("""
<div class="page-header">
  <div class="page-eyebrow">✦ Semantic Document Search</div>
  <h1 class="page-title">Ask your <em>documents</em></h1>
  <p class="page-desc">Upload PDFs or text files, then ask questions in plain English. Powered by BM25 page-level retrieval — no vector database required.</p>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_sources = st.tabs(["  Chat  ", "  Retrieved Pages  "])

# ── Chat tab 
with tab_chat:
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">✦</div>
          <div class="empty-text">Upload a document and ask your first question.<br>Answers are grounded in your files.</div>
        </div>
        """, unsafe_allow_html=True)
        
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask a question about your documents…"):
        if stats["total_pages"] == 0:
            st.warning("Please upload and index at least one document first.")
        else:
            # Display user message instantly
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.chat_history.append({"role": "user", "content": query})

            # Assistant response
            with st.chat_message("assistant"):
                with st.spinner("Retrieving pages & generating answer…"):
                    results = st.session_state.index.search(query, top_k=4)
                    st.session_state.last_results = results
                    context = build_context(results, max_chars=3500)
                    answer = ask_groq(
                        system_prompt=(
                            "You are a precise, helpful assistant. Answer using ONLY the provided context. "
                            "If the answer is not present, say so clearly. "
                            "Cite the source filename and page number in square brackets."
                        ),
                        user_prompt=f"Context:\n{context}\n\nQuestion: {query}",
                    )
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ── Sources tab 
with tab_sources:
    results = st.session_state.last_results
    if not results:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">◎</div>
          <div class="empty-text">Retrieved pages will appear here<br>after your first query.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='font-size:0.8rem;color:#9A8E7E;margin-bottom:1rem;'><strong>{len(results)}</strong> pages retrieved for last query</p>", unsafe_allow_html=True)
        for r in results:
            preview = r["text"][:450].replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ")
            st.markdown(f"""
            <div class="res-card">
              <div class="res-header">
                <span class="res-source">{r['source']}</span>
                <span class="res-score">BM25 · {r['score']}</span>
              </div>
              <div class="res-page">Page {r['page']}</div>
              <div class="res-text">{preview}{'…' if len(r['text']) > 450 else ''}</div>
            </div>
            """, unsafe_allow_html=True)