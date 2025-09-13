import React, { useState } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [status, setStatus] = useState("");
  const [q, setQ] = useState("");
  const [ans, setAns] = useState(null);
  const [sources, setSources] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);

  const BASE_URL = "http://localhost:8000";

  const styles = {
    page: {
      minHeight: "100vh",
      background: "#0f172a",
      padding: 24,
      boxSizing: "border-box",
      color: "#e2e8f0",
      fontFamily: "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
    },
    container: {
      maxWidth: 960,
      margin: "0 auto",
    },
    header: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      marginBottom: 20,
    },
    title: {
      margin: 0,
      fontSize: 24,
      fontWeight: 700,
      color: "#f1f5f9",
    },
    subtitle: {
      margin: 0,
      fontSize: 13,
      color: "#94a3b8",
    },
    grid: {
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: 16,
    },
    card: {
      background: "#111827",
      border: "1px solid #1f2937",
      borderRadius: 12,
      padding: 16,
    },
    sectionTitle: {
      marginTop: 0,
      marginBottom: 12,
      fontSize: 16,
      color: "#f8fafc",
    },
    label: {
      display: "block",
      fontSize: 13,
      color: "#cbd5e1",
      marginBottom: 8,
    },
    input: {
      width: "100%",
      background: "#0b1220",
      border: "1px solid #1f2937",
      color: "#e5e7eb",
      padding: "10px 12px",
      borderRadius: 8,
      outline: "none",
    },
    buttonRow: {
      display: "flex",
      gap: 8,
      marginTop: 12,
    },
    button: (variant = "primary", disabled = false) => ({
      padding: "10px 14px",
      borderRadius: 8,
      border: "1px solid transparent",
      background: variant === "primary" ? "#2563eb" : "#334155",
      color: "white",
      cursor: disabled ? "not-allowed" : "pointer",
      opacity: disabled ? 0.6 : 1,
      transition: "background 120ms ease",
    }),
    status: (tone = "neutral") => ({
      marginTop: 12,
      padding: "10px 12px",
      borderRadius: 8,
      fontSize: 13,
      background:
        tone === "error"
          ? "#7f1d1d"
          : tone === "success"
          ? "#064e3b"
          : "#1e293b",
      border:
        tone === "error"
          ? "1px solid #b91c1c"
          : tone === "success"
          ? "1px solid #10b981"
          : "1px solid #334155",
      color: "#e2e8f0",
      whiteSpace: "pre-wrap",
    }),
    answerCard: {
      background: "#0b1220",
      border: "1px solid #1f2937",
      borderRadius: 12,
      padding: 16,
      marginTop: 16,
    },
    answerBlock: {
      marginTop: 8,
      background: "#0a0f1a",
      border: "1px solid #1f2937",
      borderRadius: 8,
      padding: 12,
      maxHeight: 320,
      overflow: "auto",
      color: "#e5e7eb",
      lineHeight: 1.6,
      whiteSpace: "pre-wrap",
      wordBreak: "break-word",
    },
    sourcesList: {
      marginTop: 8,
      paddingLeft: 18,
    },
    footerNote: {
      marginTop: 8,
      fontSize: 12,
      color: "#64748b",
    },
  };

  const setStatusTone = (text) => {
    if (!text) return "neutral";
    if (/fail|error|network/i.test(text)) return "error";
    if (/done|success|ingest|uploaded|query/i.test(text)) return "success";
    return "neutral";
  };

  const upload = async () => {
    if (!file) {
      setStatus("Upload failed: choose a file first");
      return;
    }
    setIsUploading(true);
    setStatus("Uploading...");
    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await fetch(`${BASE_URL}/upload`, { method: "POST", body: fd });
      const j = await res.json();
      if (j.file_id) {
        setFileId(j.file_id);
        setStatus(j.status || "Ingestion done");
      } else {
        setStatus("Upload failed: no file_id returned");
      }
    } catch (err) {
      console.error(err);
      setStatus("Upload failed: network error");
    } finally {
      setIsUploading(false);
    }
  };

  const ask = async () => {
    if (!q.trim()) {
      setStatus("Query failed: enter a question");
      return;
    }
    if (!fileId) {
      setStatus("Query failed: upload and ingest a PDF first");
      return;
    }
    setIsAsking(true);
    setStatus("Querying...");
    const fd = new FormData();
    fd.append("question", q);
    fd.append("file_id", fileId);

    try {
      const res = await fetch(`${BASE_URL}/ask`, { method: "POST", body: fd });
      const j = await res.json();
      setAns(j.answer || "No answer received");
      setSources(j.sources || []);
      setStatus("Done");
    } catch (err) {
      console.error(err);
      setStatus("Query failed: network error");
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <div style={styles.header}>
          <div>
            <h2 style={styles.title}>Multimodal RAG Demo</h2>
            <p style={styles.subtitle}>Upload a PDF, then ask questions about its content.</p>
          </div>
        </div>

        <div style={styles.grid}>
          <section style={styles.card}>
            <h3 style={styles.sectionTitle}>1. Upload & Ingest</h3>
            <label style={styles.label} htmlFor="file-input">Choose a PDF</label>
            <input
              id="file-input"
              type="file"
              accept="application/pdf,.pdf"
              onChange={(e) => setFile(e.target.files && e.target.files[0] ? e.target.files[0] : null)}
              style={styles.input}
            />
            <div style={styles.buttonRow}>
              <button
                onClick={upload}
                disabled={isUploading || !file}
                style={styles.button("primary", isUploading || !file)}
              >
                {isUploading ? "Uploading…" : "Upload & Ingest"}
              </button>
              {fileId && (
                <span style={styles.footerNote}>Ingested File ID: {fileId}</span>
              )}
            </div>
            {status && (
              <div style={styles.status(setStatusTone(status))}>{status}</div>
            )}
          </section>

          <section style={styles.card}>
            <h3 style={styles.sectionTitle}>2. Ask a Question</h3>
            <label style={styles.label} htmlFor="question">Your question</label>
            <textarea
              id="question"
              rows={6}
              placeholder="Ask anything about the uploaded document…"
              style={{ ...styles.input, resize: "vertical", minHeight: 120 }}
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
            <div style={styles.buttonRow}>
              <button
                onClick={ask}
                disabled={isAsking || !fileId || !q.trim()}
                style={styles.button("primary", isAsking || !fileId || !q.trim())}
              >
                {isAsking ? "Querying…" : "Ask"}
              </button>
              {!fileId && (
                <span style={styles.footerNote}>Upload and ingest a PDF before asking.</span>
              )}
            </div>
          </section>
        </div>

        {(ans || (sources && sources.length > 0)) && (
          <section style={styles.answerCard}>
            <h3 style={styles.sectionTitle}>Answer</h3>
            <div style={styles.answerBlock}>{ans}</div>

            {sources && sources.length > 0 && (
              <div style={{ marginTop: 14 }}>
                <h4 style={{ ...styles.sectionTitle, fontSize: 14, marginBottom: 8 }}>Sources</h4>
                <ul style={styles.sourcesList}>
                  {sources.map((s, i) => (
                    <li key={i} style={{ color: "#cbd5e1" }}>{s}</li>
                  ))}
                </ul>
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
