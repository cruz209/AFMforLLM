# embedding.py
from __future__ import annotations

from typing import List
from utils import simple_tokenize, l2_normalize_inplace

class Embedder:
    """Interface for producing vector embeddings for text."""
    dim: int

    def encode(self, text: str) -> List[float]:  # pragma: no cover
        raise NotImplementedError

class HashingEmbedder(Embedder):
    """
    Dependency-free feature hashing embedder.
    Fast/stable for demos; swap for OpenAI/HF in prod.
    """
    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for i, tok in enumerate(simple_tokenize(text)):
            bucket = (hash(tok) ^ (i * 0x9E3779B1)) % self.dim
            sign = -1.0 if (hash(tok + "$") & 1) else 1.0
            vec[bucket] += sign
        l2_normalize_inplace(vec)
        return vec
