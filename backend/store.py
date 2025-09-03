# backend/store.py
import faiss
import numpy as np
import os
from PyPDF2 import PdfReader
from embeddings import embed_texts

INDEX_PATH = "vector.index"
META_PATH = "meta.npy"

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest_pdf(pdf_path: str):
    """Extract text from PDF and build FAISS index."""
    print(f"Reading {pdf_path} ...")
    reader = PdfReader(pdf_path)
    texts, metas = [], []

    for page_num, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        if not raw_text.strip():
            continue

        chunks = chunk_text(raw_text)
        for i, ch in enumerate(chunks):
            texts.append(ch)
            metas.append({
                "type": "text",
                "source": pdf_path,
                "page": page_num + 1,
                "chunk": i,
                "text": ch
            })

    # Create embeddings
    print("Embedding texts...")
    embs = embed_texts(texts)

    # Build FAISS index
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)

    # Save index + metadata
    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, np.array(metas, dtype=object), allow_pickle=True)
    print(f"Indexed {len(texts)} chunks")

def init_store():
    """Load FAISS index + metadata if available."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return None, None, None
    index = faiss.read_index(INDEX_PATH)
    meta_list = np.load(META_PATH, allow_pickle=True).tolist()
    return index, meta_list, None
