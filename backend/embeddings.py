# backend/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a lightweight text embedding model
_model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading text embedding model: {_model_name}")
_model = SentenceTransformer(_model_name)

def embed_texts(texts: list[str]) -> np.ndarray:
    """Return (n, d) numpy array of embeddings for a list of texts."""
    embeddings = _model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")
