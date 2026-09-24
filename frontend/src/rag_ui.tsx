import React, { useEffect, useRef, useState } from "react";

/**
 * Vectorless RAG — UI
 * Port of the original Streamlit app.py to React/TSX.
 * Assumes a backend exposing:
 *   GET    /api/stats                -> { total_pages, total_terms, sources: string[] }
 *   POST   /api/upload  (multipart)  -> { source, pages }
 *   POST   /api/query   { query }    -> { answer, results: SearchResult[] }
 *   POST   /api/clear                -> {}
 */
const API_BASE =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/api";

interface Stats {
  total_pages: number;
  total_terms: number;
  sources: string[];
}

interface SearchResult {
  source: string;
  page: number;
  score: number;
  text: string;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

const EMPTY_STATS: Stats = { total_pages: 0, total_terms: 0, sources: [] };

export default function RagUI() {
  const [stats, setStats] = useState<Stats>(EMPTY_STATS);
  const [indexedFiles, setIndexedFiles] = useState<Set<string>>(new Set());
  const [uploading, setUploading] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [lastResults, setLastResults] = useState<SearchResult[]>([]);
  const [tab, setTab] = useState<"chat" | "sources">("chat");
  const [query, setQuery] = useState("");
  const [thinking, setThinking] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const refreshStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      if (res.ok) setStats(await res.json());
    } catch {
      /* backend unreachable — keep last known stats */
    }
  };

  useEffect(() => {
    refreshStats();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, thinking]);

  const handleUpload = async (files: FileList | null) => {
    if (!files) return;
    for (const file of Array.from(files)) {
      if (indexedFiles.has(file.name)) continue;
      setUploading(file.name);
      try {
        const form = new FormData();
        form.append("file", file);
        const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
        if (!res.ok) throw new Error(await res.text());
        setIndexedFiles((prev) => new Set(prev).add(file.name));
        await refreshStats();
      } catch (e) {
        setWarning(`Failed to index ${file.name}: ${(e as Error).message}`);
      } finally {
        setUploading(null);
      }
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleClear = async () => {
    try {
      await fetch(`${API_BASE}/clear`, { method: "POST" });
    } catch {
      /* proceed with local reset regardless */
    }
    setStats(EMPTY_STATS);
    setChatHistory([]);
    setLastResults([]);
    setIndexedFiles(new Set());
    setWarning(null);
  };

  const handleAsk = async () => {
    const q = query.trim();
    if (!q) return;
    if (stats.total_pages === 0) {
      setWarning("Upload and index at least one document first.");
      return;
    }
    setWarning(null);
    setQuery("");
    setChatHistory((prev) => [...prev, { role: "user", content: q }]);
    setThinking(true);
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      const data = await res.json();
      setLastResults(data.results ?? []);
      setChatHistory((prev) => [...prev, { role: "assistant", content: data.answer ?? "⚠️ No answer returned." }]);
    } catch (e) {
      setChatHistory((prev) => [...prev, { role: "assistant", content: `⚠️ Error: ${(e as Error).message}` }]);
    } finally {
      setThinking(false);
    }
  };

  return (
    <div className="rag-app">
      <GlobalStyle />

      <aside className="rag-sidebar">
        <div className="brand">
          <div className="brand-mark">V</div>
          <div>
            <div className="brand-name">Vectorless<span>RAG</span></div>
            <div className="brand-caption">BM25 · page index · no embeddings</div>
          </div>
        </div>

        <div className="sidebar-section">
          <div className="section-label">Knowledge base</div>

          <div className="stats-grid">
            <div className="stat-card">
              <span className="stat-value">{stats.total_pages}</span>
              <span className="stat-label">Pages</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">{stats.total_terms.toLocaleString()}</span>
              <span className="stat-label">Terms</span>
            </div>
          </div>
        </div>

        <div className="sidebar-section documents-section">
          <div className="section-heading">
            <span className="section-label">Indexed documents</span>
            <span className="doc-count">{stats.sources.length}</span>
          </div>

          <div className="document-list">
            {stats.sources.length > 0 ? (
              stats.sources.map((s) => (
                <div className="document-item" key={s} title={s}>
                  <div className="document-icon">↳</div>
                  <div className="document-meta">
                    <span className="document-name">{s}</span>
                    <span className="document-status">Indexed</span>
                  </div>
                  <span className="status-dot" />
                </div>
              ))
            ) : (
              <div className="empty-docs">
                <div className="empty-doc-icon">+</div>
                <span>No documents indexed</span>
              </div>
            )}
          </div>
        </div>

        <div className="sidebar-bottom">
          <div className="upload-label">Add documents</div>
          <label className={`upload-zone ${uploading ? "uploading" : ""}`}>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.txt"
              multiple
              style={{ display: "none" }}
              onChange={(e) => handleUpload(e.target.files)}
            />
            <span className="upload-icon">{uploading ? "…" : "↑"}</span>
            <span className="upload-title">
              {uploading ? `Indexing ${uploading}` : "Upload PDF or TXT"}
            </span>
            <span className="upload-hint">Click to browse files</span>
          </label>

          <button className="clear-button" onClick={handleClear}>
            <span>↻</span>
            Clear knowledge base
          </button>
        </div>
      </aside>

      <main className="rag-main">
        <header className="topbar">
          <div className="topbar-context">
            <span className="online-dot" />
            Local document intelligence
          </div>
          <div className="engine-pill">
            <span>BM25</span>
            <span className="pill-separator">•</span>
            Vectorless retrieval
          </div>
        </header>

        <div className="workspace">
          <section className="hero">
            <div className="hero-badge">
              <span>✦</span>
              DOCUMENT AI
            </div>
            <h1>
              Ask your <span>documents.</span>
            </h1>
            <p>
              Search your indexed files and get answers grounded directly in
              the pages you uploaded.
            </p>
          </section>

          <div className="workspace-tabs">
            <button
              className={`workspace-tab ${tab === "chat" ? "active" : ""}`}
              onClick={() => setTab("chat")}
            >
              <span>◈</span>
              Chat
            </button>
            <button
              className={`workspace-tab ${tab === "sources" ? "active" : ""}`}
              onClick={() => setTab("sources")}
            >
              <span>⌁</span>
              Retrieved pages
              {lastResults.length > 0 && (
                <span className="tab-count">{lastResults.length}</span>
              )}
            </button>
          </div>

          {warning && (
            <div className="warning">
              <span>!</span>
              {warning}
            </div>
          )}

          {tab === "chat" ? (
            <div className="chat-workspace">
              {chatHistory.length === 0 && !thinking ? (
                <div className="welcome-card">
                  <div className="welcome-orb">
                    <span>✦</span>
                  </div>
                  <h2>What would you like to know?</h2>
                  <p>
                    Upload a document, then ask anything about its contents.
                    Your answers are retrieved from the indexed pages.
                  </p>

                  <div className="suggestion-grid">
                    <button
                      onClick={() => setQuery("Summarize the main points")}
                    >
                      <span>⌘</span>
                      Summarize the main points
                    </button>
                    <button
                      onClick={() => setQuery("What are the key findings?")}
                    >
                      <span>◉</span>
                      What are the key findings?
                    </button>
                    <button
                      onClick={() => setQuery("Find the most important details")}
                    >
                      <span>⌕</span>
                      Find important details
                    </button>
                  </div>
                </div>
              ) : (
                <div className="chat-log">
                  {chatHistory.map((m, i) => (
                    <div
                      key={i}
                      className={`message-row ${m.role === "user" ? "user-row" : ""}`}
                    >
                      <div className={`avatar ${m.role}`}>
                        {m.role === "user" ? "Y" : "V"}
                      </div>
                      <div className="message-body">
                        <div className="message-name">
                          {m.role === "user" ? "You" : "Vectorless"}
                        </div>
                        <div className="message-content">{m.content}</div>
                      </div>
                    </div>
                  ))}

                  {thinking && (
                    <div className="message-row">
                      <div className="avatar assistant">V</div>
                      <div className="message-body">
                        <div className="message-name">Vectorless</div>
                        <div className="thinking">
                          <span />
                          <span />
                          <span />
                          <em>Searching indexed pages…</em>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>
              )}

              <div className="composer-wrap">
                <div className="composer">
                  <button
                    className="composer-attach"
                    title="Upload document"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    +
                  </button>
                  <input
                    className="composer-input"
                    placeholder="Ask anything about your documents…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleAsk()}
                  />
                  <button
                    className="send-button"
                    onClick={handleAsk}
                    disabled={thinking || !query.trim()}
                    title="Send"
                  >
                    ↑
                  </button>
                </div>
                <div className="composer-footer">
                  <span>Answers are grounded in your indexed documents</span>
                  <span>Enter ↵ to send</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="sources-workspace">
              {lastResults.length === 0 ? (
                <div className="sources-empty">
                  <div className="source-empty-icon">⌁</div>
                  <h2>No retrieved pages yet</h2>
                  <p>
                    Ask a question in Chat and the pages used to generate the
                    answer will appear here.
                  </p>
                </div>
              ) : (
                <>
                  <div className="sources-summary">
                    <div>
                      <span className="summary-number">{lastResults.length}</span>
                      <span className="summary-text">
                        pages retrieved for the last query
                      </span>
                    </div>
                    <span className="retrieval-method">BM25 retrieval</span>
                  </div>

                  <div className="results-list">
                    {lastResults.map((r, i) => {
                      const preview = r.text.slice(0, 450);
                      return (
                        <article className="result-card" key={i}>
                          <div className="result-top">
                            <div className="result-source">
                              <span className="file-icon">▤</span>
                              {r.source}
                            </div>
                            <div className="score-badge">
                              BM25 <strong>{r.score}</strong>
                            </div>
                          </div>
                          <div className="result-page">PAGE {r.page}</div>
                          <p className="result-text">
                            {preview}
                            {r.text.length > 450 ? "…" : ""}
                          </p>
                        </article>
                      );
                    })}
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

/** Global styles for the redesigned Vectorless RAG interface. */
function GlobalStyle() {
  return (
    <style>{`
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

      :root {
        color-scheme: dark;
        --bg: #090a0c;
        --panel: #101216;
        --panel-2: #15171c;
        --panel-3: #1b1e24;
        --border: #272b33;
        --border-soft: #1e2127;
        --text: #f1f3f5;
        --muted: #8e96a3;
        --muted-2: #626a76;
        --accent: #c9ff52;
        --accent-soft: rgba(201,255,82,.10);
        --danger: #ff8f8f;
      }

      * { box-sizing: border-box; }
      html, body, #root { min-height: 100%; }
      body {
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font-family: "DM Sans", sans-serif;
      }
      button, input { font: inherit; }
      button { -webkit-tap-highlight-color: transparent; }
      ::selection { background: rgba(201,255,82,.22); }
      ::-webkit-scrollbar { width: 7px; height: 7px; }
      ::-webkit-scrollbar-track { background: transparent; }
      ::-webkit-scrollbar-thumb { background: #30343d; border-radius: 999px; }

      .rag-app {
        min-height: 100vh;
        display: flex;
        background:
          radial-gradient(circle at 70% -10%, rgba(201,255,82,.055), transparent 28rem),
          var(--bg);
      }

      .rag-sidebar {
        width: 292px;
        min-height: 100vh;
        flex: 0 0 292px;
        display: flex;
        flex-direction: column;
        padding: 24px 18px 18px;
        background: rgba(12,13,16,.96);
        border-right: 1px solid var(--border);
      }

      .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 8px 24px;
        border-bottom: 1px solid var(--border-soft);
      }
      .brand-mark {
        width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        border-radius: 10px;
        background: var(--accent);
        color: #10120d;
        font-weight: 800;
        box-shadow: 0 0 30px rgba(201,255,82,.10);
      }
      .brand-name {
        font-size: 16px;
        font-weight: 700;
        letter-spacing: -.02em;
      }
      .brand-name span { color: var(--accent); }
      .brand-caption {
        margin-top: 3px;
        color: var(--muted-2);
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: .11em;
      }

      .sidebar-section { margin-top: 22px; }
      .section-label, .upload-label {
        color: var(--muted-2);
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .13em;
      }
      .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-top: 10px;
      }
      .stat-card {
        padding: 13px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--panel);
      }
      .stat-value {
        display: block;
        color: var(--text);
        font-size: 21px;
        font-weight: 700;
        letter-spacing: -.04em;
      }
      .stat-label {
        display: block;
        margin-top: 3px;
        color: var(--muted-2);
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: .1em;
      }

      .documents-section { flex: 1; min-height: 0; }
      .section-heading {
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .doc-count {
        min-width: 22px;
        height: 22px;
        display: grid;
        place-items: center;
        border-radius: 7px;
        background: var(--panel-2);
        color: var(--muted);
        font-size: 10px;
      }
      .document-list {
        display: flex;
        flex-direction: column;
        gap: 5px;
        margin-top: 10px;
        max-height: 38vh;
        overflow: auto;
      }
      .document-item {
        display: flex;
        align-items: center;
        gap: 9px;
        padding: 9px;
        border: 1px solid transparent;
        border-radius: 10px;
        transition: .18s ease;
      }
      .document-item:hover {
        background: var(--panel);
        border-color: var(--border);
      }
      .document-icon {
        width: 28px;
        height: 28px;
        display: grid;
        place-items: center;
        flex: 0 0 28px;
        border-radius: 8px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 13px;
      }
      .document-meta {
        min-width: 0;
        flex: 1;
      }
      .document-name {
        display: block;
        overflow: hidden;
        color: #dce0e5;
        font-size: 11px;
        font-weight: 500;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .document-status {
        display: block;
        margin-top: 2px;
        color: var(--muted-2);
        font-size: 9px;
      }
      .status-dot {
        width: 6px;
        height: 6px;
        flex: 0 0 6px;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 8px rgba(201,255,82,.35);
      }
      .empty-docs {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 14px 8px;
        color: var(--muted-2);
        font-size: 11px;
      }
      .empty-doc-icon {
        width: 24px;
        height: 24px;
        display: grid;
        place-items: center;
        border: 1px dashed #343943;
        border-radius: 7px;
      }

      .sidebar-bottom { margin-top: 18px; }
      .upload-zone {
        min-height: 106px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 3px;
        margin-top: 9px;
        padding: 14px;
        border: 1px dashed #353a44;
        border-radius: 13px;
        background: var(--panel);
        cursor: pointer;
        text-align: center;
        transition: .2s ease;
      }
      .upload-zone:hover, .upload-zone.uploading {
        border-color: #596746;
        background: rgba(201,255,82,.035);
      }
      .upload-icon {
        width: 28px;
        height: 28px;
        display: grid;
        place-items: center;
        margin-bottom: 3px;
        border-radius: 8px;
        background: var(--panel-3);
        color: var(--accent);
        font-size: 17px;
      }
      .upload-title { font-size: 11px; font-weight: 600; color: #dfe2e7; }
      .upload-hint { font-size: 9px; color: var(--muted-2); }
      .clear-button {
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 7px;
        margin-top: 9px;
        padding: 9px;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: transparent;
        color: var(--muted);
        cursor: pointer;
        font-size: 10px;
        transition: .18s ease;
      }
      .clear-button:hover { color: var(--text); background: var(--panel); }

      .rag-main {
        min-width: 0;
        flex: 1;
      }
      .topbar {
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 34px;
        border-bottom: 1px solid var(--border-soft);
      }
      .topbar-context {
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--muted);
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: .1em;
      }
      .online-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 10px rgba(201,255,82,.4);
      }
      .engine-pill {
        padding: 6px 9px;
        border: 1px solid var(--border);
        border-radius: 7px;
        color: var(--muted);
        font-family: "DM Mono", monospace;
        font-size: 9px;
      }
      .engine-pill span:first-child { color: var(--accent); }
      .pill-separator { margin: 0 5px; color: #3e434c; }

      .workspace {
        width: min(900px, calc(100% - 56px));
        min-height: calc(100vh - 60px);
        margin: 0 auto;
        padding: 58px 0 44px;
      }
      .hero { max-width: 680px; margin-bottom: 38px; }
      .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        color: var(--accent);
        font-size: 9px;
        font-weight: 700;
        letter-spacing: .16em;
      }
      .hero h1 {
        margin: 12px 0 9px;
        font-size: clamp(36px, 5vw, 58px);
        line-height: .98;
        letter-spacing: -.055em;
      }
      .hero h1 span { color: var(--accent); }
      .hero p {
        max-width: 590px;
        margin: 0;
        color: var(--muted);
        font-size: 14px;
        line-height: 1.65;
      }

      .workspace-tabs {
        display: flex;
        gap: 4px;
        margin-bottom: 18px;
        border-bottom: 1px solid var(--border);
      }
      .workspace-tab {
        position: relative;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 11px 13px;
        border: 0;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
        background: transparent;
        color: var(--muted-2);
        cursor: pointer;
        font-size: 11px;
        font-weight: 600;
      }
      .workspace-tab.active {
        color: var(--text);
        border-bottom-color: var(--accent);
      }
      .workspace-tab:hover { color: var(--text); }
      .tab-count {
        min-width: 18px;
        padding: 2px 5px;
        border-radius: 5px;
        background: var(--panel-3);
        color: var(--muted);
        font-size: 9px;
      }

      .warning {
        display: flex;
        align-items: center;
        gap: 9px;
        margin-bottom: 16px;
        padding: 10px 12px;
        border: 1px solid #493636;
        border-radius: 10px;
        background: rgba(255,143,143,.055);
        color: var(--danger);
        font-size: 11px;
      }
      .warning span {
        width: 20px;
        height: 20px;
        display: grid;
        place-items: center;
        border-radius: 6px;
        background: rgba(255,143,143,.1);
      }

      .chat-workspace { min-height: 500px; display: flex; flex-direction: column; }
      .welcome-card {
        padding: 46px 24px 35px;
        border: 1px solid var(--border);
        border-radius: 18px;
        background:
          radial-gradient(circle at 50% 0%, rgba(201,255,82,.045), transparent 18rem),
          var(--panel);
        text-align: center;
      }
      .welcome-orb {
        width: 54px;
        height: 54px;
        display: grid;
        place-items: center;
        margin: 0 auto 17px;
        border: 1px solid rgba(201,255,82,.25);
        border-radius: 16px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 22px;
      }
      .welcome-card h2 {
        margin: 0;
        font-size: 19px;
        letter-spacing: -.025em;
      }
      .welcome-card > p {
        max-width: 470px;
        margin: 8px auto 24px;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.65;
      }
      .suggestion-grid {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 7px;
      }
      .suggestion-grid button {
        display: flex;
        align-items: center;
        gap: 7px;
        padding: 8px 10px;
        border: 1px solid var(--border);
        border-radius: 9px;
        background: var(--panel-2);
        color: #b9bec7;
        cursor: pointer;
        font-size: 10px;
        transition: .18s ease;
      }
      .suggestion-grid button:hover {
        border-color: #444a54;
        color: var(--text);
        transform: translateY(-1px);
      }
      .suggestion-grid button span { color: var(--accent); }

      .chat-log {
        display: flex;
        flex-direction: column;
        gap: 28px;
        padding: 10px 2px 26px;
      }
      .message-row {
        display: flex;
        gap: 12px;
        max-width: 84%;
      }
      .message-row.user-row {
        align-self: flex-end;
        flex-direction: row-reverse;
      }
      .avatar {
        width: 30px;
        height: 30px;
        flex: 0 0 30px;
        display: grid;
        place-items: center;
        border-radius: 9px;
        font-size: 10px;
        font-weight: 800;
      }
      .avatar.assistant {
        background: var(--accent);
        color: #11130e;
      }
      .avatar.user {
        background: var(--panel-3);
        border: 1px solid var(--border);
        color: #c8cdd5;
      }
      .message-body { min-width: 0; }
      .message-name {
        margin-bottom: 6px;
        color: var(--muted-2);
        font-size: 9px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .1em;
      }
      .message-content {
        padding: 12px 14px;
        border: 1px solid var(--border);
        border-radius: 4px 13px 13px 13px;
        background: var(--panel);
        color: #d9dde3;
        font-size: 13px;
        line-height: 1.7;
        white-space: pre-wrap;
      }
      .user-row .message-content {
        border-radius: 13px 4px 13px 13px;
        background: var(--accent-soft);
        border-color: rgba(201,255,82,.13);
      }

      .thinking {
        display: flex;
        align-items: center;
        gap: 4px;
        color: var(--muted);
        font-size: 11px;
      }
      .thinking span {
        width: 4px;
        height: 4px;
        border-radius: 50%;
        background: var(--accent);
        animation: pulse 1.2s infinite;
      }
      .thinking span:nth-child(2) { animation-delay: .15s; }
      .thinking span:nth-child(3) { animation-delay: .3s; }
      .thinking em { margin-left: 6px; font-style: normal; }
      @keyframes pulse { 0%,100% { opacity: .25; transform: scale(.8); } 50% { opacity: 1; transform: scale(1); } }

      .composer-wrap { margin-top: auto; padding-top: 12px; }
      .composer {
        display: flex;
        align-items: center;
        gap: 9px;
        padding: 8px;
        border: 1px solid #343944;
        border-radius: 15px;
        background: rgba(20,22,27,.96);
        box-shadow: 0 14px 45px rgba(0,0,0,.25);
        transition: .2s ease;
      }
      .composer:focus-within {
        border-color: #505861;
        box-shadow: 0 14px 50px rgba(0,0,0,.32), 0 0 0 2px rgba(201,255,82,.035);
      }
      .composer-attach, .send-button {
        width: 34px;
        height: 34px;
        flex: 0 0 34px;
        display: grid;
        place-items: center;
        border-radius: 10px;
        cursor: pointer;
      }
      .composer-attach {
        border: 1px solid var(--border);
        background: var(--panel-2);
        color: var(--muted);
        font-size: 19px;
      }
      .composer-attach:hover { color: var(--text); }
      .composer-input {
        min-width: 0;
        flex: 1;
        border: 0;
        outline: 0;
        background: transparent;
        color: var(--text);
        font-size: 13px;
      }
      .composer-input::placeholder { color: #5f6672; }
      .send-button {
        border: 0;
        background: var(--accent);
        color: #11130e;
        font-weight: 800;
        font-size: 17px;
        transition: .18s ease;
      }
      .send-button:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 5px 18px rgba(201,255,82,.16); }
      .send-button:disabled { opacity: .28; cursor: not-allowed; }
      .composer-footer {
        display: flex;
        justify-content: space-between;
        padding: 7px 4px 0;
        color: #4f5661;
        font-size: 9px;
      }

      .sources-workspace { min-height: 500px; }
      .sources-empty {
        padding: 72px 25px;
        border: 1px dashed var(--border);
        border-radius: 18px;
        text-align: center;
        background: rgba(16,18,22,.55);
      }
      .source-empty-icon {
        color: var(--accent);
        font-size: 30px;
        margin-bottom: 10px;
      }
      .sources-empty h2 {
        margin: 0;
        font-size: 17px;
      }
      .sources-empty p {
        max-width: 430px;
        margin: 8px auto 0;
        color: var(--muted);
        font-size: 11px;
        line-height: 1.6;
      }
      .sources-summary {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
      }
      .summary-number { margin-right: 6px; font-size: 17px; font-weight: 700; }
      .summary-text { color: var(--muted); font-size: 11px; }
      .retrieval-method {
        padding: 5px 8px;
        border: 1px solid var(--border);
        border-radius: 6px;
        color: var(--muted-2);
        font-family: "DM Mono", monospace;
        font-size: 8px;
        text-transform: uppercase;
      }
      .results-list { display: flex; flex-direction: column; gap: 9px; }
      .result-card {
        padding: 16px;
        border: 1px solid var(--border);
        border-radius: 13px;
        background: var(--panel);
        transition: .18s ease;
      }
      .result-card:hover {
        border-color: #3a404a;
        transform: translateY(-1px);
      }
      .result-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }
      .result-source {
        min-width: 0;
        display: flex;
        align-items: center;
        gap: 8px;
        overflow: hidden;
        color: #d8dce2;
        font-size: 11px;
        font-weight: 600;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .file-icon {
        width: 25px;
        height: 25px;
        display: grid;
        place-items: center;
        flex: 0 0 25px;
        border-radius: 7px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 11px;
      }
      .score-badge {
        flex: 0 0 auto;
        padding: 4px 7px;
        border: 1px solid rgba(201,255,82,.13);
        border-radius: 6px;
        background: rgba(201,255,82,.055);
        color: #aebd91;
        font-family: "DM Mono", monospace;
        font-size: 8px;
      }
      .score-badge strong { color: var(--accent); }
      .result-page {
        margin: 11px 0 7px;
        color: var(--muted-2);
        font-size: 8px;
        font-weight: 700;
        letter-spacing: .13em;
      }
      .result-text {
        margin: 0;
        color: #aeb4bd;
        font-size: 11px;
        line-height: 1.7;
      }

      @media (max-width: 800px) {
        .rag-sidebar { width: 235px; flex-basis: 235px; }
        .workspace { width: min(100% - 32px, 900px); }
        .topbar { padding: 0 18px; }
      }
      @media (max-width: 650px) {
        .rag-app { display: block; }
        .rag-sidebar {
          width: 100%;
          min-height: auto;
          border-right: 0;
          border-bottom: 1px solid var(--border);
        }
        .documents-section { display: none; }
        .sidebar-bottom { margin-top: 16px; }
        .upload-zone { min-height: 75px; }
        .topbar-context { display: none; }
        .topbar { justify-content: flex-end; }
        .workspace { width: calc(100% - 24px); padding-top: 35px; }
        .hero h1 { font-size: 40px; }
        .message-row { max-width: 94%; }
        .composer-footer span:last-child { display: none; }
      }
    `}</style>
  );
}

