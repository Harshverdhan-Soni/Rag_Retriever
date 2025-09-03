# backend/embeddings.py
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from typing import List
import numpy as np

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model_name = "openai/clip-vit-base-patch32"  # HuggingFace CLIP

print("Loading CLIP model on", _device)
_clip_model = CLIPModel.from_pretrained(_model_name).to(_device)
_processor = CLIPProcessor.from_pretrained(_model_name)

def embed_texts(texts: List[str]) -> np.ndarray:
    """Return (n, d) numpy array — CLIP text embeddings (normalized)."""
    inputs = _processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(_device)
    with torch.no_grad():
        out = _clip_model.get_text_features(**inputs)
    arr = out.cpu().numpy()
    # normalize
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10)
    return arr.astype("float32")

def embed_images(pil_images: List[Image.Image]) -> np.ndarray:
    inputs = _processor(images=pil_images, return_tensors="pt").to(_device)
    with torch.no_grad():
        out = _clip_model.get_image_features(**inputs)
    arr = out.cpu().numpy()
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10)
    return arr.astype("float32")
