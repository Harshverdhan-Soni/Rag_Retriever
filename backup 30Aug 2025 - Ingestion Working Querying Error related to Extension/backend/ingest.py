# backend/ingest.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import tempfile, shutil
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from docx import Document as DocxDocument
from langchain.text_splitter import CharacterTextSplitter
from embeddings import embed_texts
from store import init_store, save_index, save_meta, save_graph, add_vectors

router = APIRouter()

def extract_text_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, p in enumerate(reader.pages):
        t = p.extract_text() or ""
        pages.append({"page": i+1, "text": t})
    return pages

def ocr_pdf(path, dpi=200):
    imgs = convert_from_path(path, dpi=dpi)
    pages=[]
    for i, img in enumerate(imgs):
        txt = pytesseract.image_to_string(img)
        pages.append({"page": i+1, "text": txt})
    return pages

def chunk_texts(records, chunk_size=600, overlap=100):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs=[]
    for r in records:
        splits = splitter.split_text(r["text"])
        for i, s in enumerate(splits):
            docs.append({"source": r["source"], "page": r["page"], "chunk_id": i, "text": s})
    return docs

@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    allowed = {".pdf"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="unsupported file")

    tmpdir = tempfile.mkdtemp()
    tmp_path = Path(tmpdir)/file.filename
    content = await file.read()
    tmp_path.write_bytes(content)

    try:
        records=[]
        if suffix == ".pdf":
            pages = extract_text_pdf(str(tmp_path))
            total = sum(len(p["text"].strip()) for p in pages)
            if total < 50:
                pages = ocr_pdf(str(tmp_path))
            for p in pages:
                records.append({"source": file.filename, "page": p["page"], "text": p["text"]})

        # split records into text chunks
        text_chunks = chunk_texts(records) if records else []

        # embeddings
        index, meta_list, graph = init_store()
        if text_chunks:
            texts = [c["text"] for c in text_chunks]
            metas = [{"type":"text","source":c["source"],"page":c["page"],"chunk":c["chunk_id"], "text": c["text"]} for c in text_chunks]
            text_emb = embed_texts(texts)
            add_vectors(index, meta_list, graph, text_emb, metas)

        # persist
        save_index(index)
        save_meta(meta_list)
        save_graph(graph)

        return {"status":"ok", "ingested_text_chunks": len(text_chunks)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
