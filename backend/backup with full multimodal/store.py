# backend/store.py
import faiss
import numpy as np
from pathlib import Path
import pickle
import networkx as nx

BASE = Path(__file__).resolve().parent
INDEX_PATH = BASE / "faiss.index"
META_PATH = BASE / "meta.pkl"
GRAPH_PATH = BASE / "graph.pkl"

_dim = 512  # CLIP base dim (clip-vit-base-patch32 -> 512)

def create_index(dim=_dim):
    idx = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine
    id_map = faiss.IndexIDMap(idx)
    return id_map

def save_index(index):
    faiss.write_index(index, str(INDEX_PATH))

def load_index():
    if INDEX_PATH.exists():
        return faiss.read_index(str(INDEX_PATH))
    return None

def save_meta(meta):
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

def load_meta():
    if META_PATH.exists():
        with open(META_PATH, "rb") as f:
            return pickle.load(f)
    return []

def save_graph(G):
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)

def load_graph():
    if GRAPH_PATH.exists():
        with open(GRAPH_PATH, "rb") as f:
            return pickle.load(f)
    return nx.DiGraph()

def init_store():
    idx = load_index()
    if idx is None:
        idx = create_index()
    meta = load_meta()
    G = load_graph()
    return idx, meta, G

def add_vectors(index, meta_list, graph, vectors: np.ndarray, metas: list):
    """vectors shape (n,d), metas a list of dicts with metadata, returns assigned ids"""
    n = vectors.shape[0]
    start = len(meta_list)
    ids = np.arange(start, start + n).astype("int64")
    index.add_with_ids(vectors, ids)
    meta_list.extend(metas)
    # add to graph: each meta becomes a node id_<id>
    for i, m in zip(ids.tolist(), metas):
        graph.add_node(f"id_{i}", **m)
    return ids
