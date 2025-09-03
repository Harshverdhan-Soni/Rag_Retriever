import React, { useState } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [status, setStatus] = useState("");
  const [q, setQ] = useState("");
  const [ans, setAns] = useState(null);
  const [sources, setSources] = useState([]);

  const BASE_URL = "http://localhost:8000";

  const upload = async () => {
    if (!file) return alert("Choose a file");
    setStatus("Uploading...");
    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await fetch(`${BASE_URL}/upload`, { method: "POST", body: fd });
      const j = await res.json();
      console.log(j);
      if (j.file_id) {
        setFileId(j.file_id);
        setStatus(j.status || "Ingestion done");
      } else {
        setStatus("Upload failed: no file_id returned");
      }
    } catch (err) {
      console.error(err);
      setStatus("Upload failed: network error");
    }
  };

  const ask = async () => {
    if (!q.trim()) return alert("Enter a question");
    if (!fileId) return alert("Upload a PDF first");
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
      setStatus("Query failed");
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Multimodal RAG Demo</h2>

      <div>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={upload} style={{ marginLeft: 8 }}>Upload & Ingest</button>
      </div>

      <div style={{ marginTop: 12 }}>{status}</div>
      <hr />

      <textarea
        rows={3}
        style={{ width: "100%" }}
        value={q}
        onChange={(e) => setQ(e.target.value)}
      />
      <button onClick={ask}>Ask</button>

      {ans && (
        <div style={{ marginTop: 12 }}>
          <h3>Answer:</h3>
          <pre style={{ background: "#f6f6f6", padding: 10 }}>{ans}</pre>

          {sources.length > 0 && (
            <>
              <h4>Sources:</h4>
              <ul>
                {sources.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}
