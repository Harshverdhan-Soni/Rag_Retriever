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
from embeddings import embed_texts, embed_images
from store import init_store, save_index, save_meta, save_graph, add_vectors
import numpy as np

router = APIRouter()

def extract_text_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i,p in enumerate(reader.pages):
        t = p.extract_text() or ""
        pages.append({"page": i+1, "text": t})
    return pages

def ocr_pdf(path, dpi=200):
    imgs = convert_from_path(path, dpi=dpi)
    pages=[]
    for i,img in enumerate(imgs):
        txt = pytesseract.image_to_string(img)
        pages.append({"page": i+1, "text": txt})
    return pages

def extract_docx(path):
    doc = DocxDocument(path)
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return [{"page":1, "text": "\n".join(paras)}]

def chunk_texts(records, chunk_size=600, overlap=100):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs=[]
    for r in records:
        d = {"source": r["source"], "page": r["page"], "text": r["text"]}
        splits = splitter.split_text(r["text"])
        for i, s in enumerate(splits):
            docs.append({"source": r["source"], "page": r["page"], "chunk_id": i, "text": s})
    return docs

@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    allowed = {".pdf",".txt",".docx",".png",".jpg",".jpeg",".mp4"}
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
            # if very little text -> OCR
            total = sum(len(p["text"].strip()) for p in pages)
            if total < 50:
                pages = ocr_pdf(str(tmp_path))
            for p in pages:
                records.append({"source": file.filename, "page": p["page"], "text": p["text"]})
        elif suffix in {".png",".jpg",".jpeg"}:
            txt = pytesseract.image_to_string(Image.open(str(tmp_path)))
            records.append({"source": file.filename, "page":1, "text": txt})
        elif suffix == ".docx":
            pages = extract_docx(str(tmp_path))
            for p in pages:
                records.append({"source": file.filename, "page": p["page"], "text": p["text"]})
        elif suffix == ".txt":
            txt = tmp_path.read_text(encoding="utf-8", errors="ignore")
            records.append({"source": file.filename, "page":1, "text": txt})
        elif suffix == ".mp4":
            # simple: sample N frames (first pass). For production use ffmpeg + smarter frame extraction.
            import cv2
            vid = cv2.VideoCapture(str(tmp_path))
            fps = vid.get(cv2.CAP_PROP_FPS) or 25
            frames_to_sample = [0, int(fps*1), int(fps*2)]  # sample first 3 seconds
            for f in frames_to_sample:
                vid.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = vid.read()
                if ret:
                    pil = Image.fromarray(frame[:,:,::-1])
                    # OCR on frame
                    txt = pytesseract.image_to_string(pil)
                    records.append({"source": file.filename, "page": f, "text": txt, "image": pil})
            vid.release()

        # split records to text chunks and images
        text_records = [r for r in records if r.get("text")]
        text_chunks = chunk_texts(text_records)

        image_records = [r for r in records if r.get("image")]  # from video frames
        # (we don't expect images for plain uploads here; extend if you pass actual images separately)

        # embeddings
        index, meta_list, graph = init_store()

        # text embeddings via CLIP text encoder
        texts = [c["text"] for c in text_chunks] if text_chunks else []
        metas = [{"type":"text","source":c["source"],"page":c["page"],"chunk":c["chunk_id"], "text": c["text"]} for c in text_chunks]
        if texts:
            text_emb = embed_texts(texts)
            ids = add_vectors(index, meta_list, graph, text_emb, metas)

        # images (if any)
        if image_records:
            imgs = [r["image"] for r in image_records]
            img_metas = [{"type":"image","source":r["source"],"page":r["page"]} for r in image_records]
            img_emb = embed_images(imgs)
            ids2 = add_vectors(index, meta_list, graph, img_emb, img_metas)

        # persist
        save_index(index)
        save_meta(meta_list)
        save_graph(graph)

        return {"status":"ok", "ingested_text_chunks": len(texts), "ingested_images": len(image_records)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
