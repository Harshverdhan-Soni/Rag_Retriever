# backend/query.py
from transformers import AutoTokenizer, AutoModelForCausalLM  # pyright: ignore[reportMissingImports]
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.store import init_store
import numpy as np
from transformers import pipeline  # pyright: ignore[reportMissingImports]
import os
from backend.embeddings import embed_texts

router = APIRouter()

class QueryIn(BaseModel):
    question: str
    top_k: int = 5

# simple LLM runner
def run_llm(prompt: str):
    model_id = os.getenv("LOCAL_HF_MODEL", "distilgpt2")  # default to distilgpt2
    try:
        # ✅ force fast tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False,
        )
        out = gen(prompt, max_new_tokens=256)[0]["generated_text"]
        return out
    except Exception as e:
        return f"LLM error: {str(e)}. Ensure {model_id} is installed."
        
@router.post("/ask")
async def ask(q: QueryIn):
    index, meta_list, graph = init_store()
    if index is None:
        raise HTTPException(400, "No index found, ingest first")

    # embed query
    q_emb = embed_texts([q.question])

    # FAISS search
    D, I = index.search(q_emb, q.top_k)
    ids = I[0].tolist()
    results = []
    for idx, score in zip(ids, D[0].tolist()):
        if idx == -1:
            continue
        meta = meta_list[idx]
        results.append({"id": idx, "score": float(score), "meta": meta})

    # build context
    context = "\n\n".join([
        f"[{r['meta'].get('type')}] Source:{r['meta'].get('source')} "
        f"page:{r['meta'].get('page')} chunk:{r['meta'].get('chunk')}\n"
        f"{r['meta'].get('text')[:1000]}"
        for r in results if r['meta'].get('type') == 'text'
    ])

    prompt = (
        "Use the following context to answer the question. "
        "If not present, say 'I don't know'.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {q.question}\n\n"
        "Answer concisely and list sources used."
    )

    answer = run_llm(prompt)
    return {"answer": answer, "retrieved": results}
