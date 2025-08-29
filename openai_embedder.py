# openai_embedder.py
import os
from typing import List
from embedding import Embedder
from utils import l2_normalize_inplace

try:
    from openai import OpenAI
    _OPENAI_V1 = True
except ImportError:
    import openai
    _OPENAI_V1 = False

class OpenAIEmbedder(Embedder):
    """OpenAI embeddings with L2-normalization."""
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        if _OPENAI_V1:
            self.client = OpenAI(api_key=api_key)
        else:
            openai.api_key = api_key
            self.client = None
        self.dim = 1536

    def encode(self, text: str) -> List[float]:
        if _OPENAI_V1:
            out = self.client.embeddings.create(model=self.model, input=text)
            vec = list(out.data[0].embedding)
        else:
            out = openai.Embedding.create(model=self.model, input=text)
            vec = list(out["data"][0]["embedding"])
        self.dim = len(vec)
        l2_normalize_inplace(vec)
        return vec
