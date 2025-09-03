# backend/query.py
from fastapi import APIRouter
import numpy as np
from embeddings import embed_texts
from store import init_store

router = APIRouter()

index, metas, _ = init_store()

@router.get("/")
def query_root():
    return {"msg": "Query endpoint is ready"}

@router.get("/search")
def search(q: str, k: int = 3):
    global index, metas
    if index is None:
        return {"error": "No index found. Ingest a PDF first."}

    # Embed query
    q_emb = embed_texts([q])
    D, I = index.search(q_emb, k)

    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx == -1:
            continue
        results.append({
            "text": metas[idx]["text"],
            "page": metas[idx]["page"],
            "score": float(dist)
        })
    return {"query": q, "results": results}
