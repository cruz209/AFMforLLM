# utils.py
from __future__ import annotations

import math
from typing import List

def simple_tokenize(text: str) -> List[str]:
    """Lowercase + simple split + strip punctuation."""
    toks = [t.strip(".,;:!?()[]{}\"'`).-_/") for t in text.lower().split()]
    return [t for t in toks if t]

def l2_normalize_inplace(v: List[float]) -> None:
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    for i in range(len(v)):
        v[i] /= s

def cosine(a: List[float], b: List[float]) -> float:
    # assumes L2-normalized inputs
    return float(sum(x * y for x, y in zip(a, b)))

def split_sentences(text: str) -> List[str]:
    parts: List[str] = []
    buff: List[str] = []
    for ch in text:
        buff.append(ch)
        if ch in ".!?\n":
            s = "".join(buff).strip()
            if s:
                parts.append(s)
            buff = []
    if buff:
        s = "".join(buff).strip()
        if s:
            parts.append(s)
    return parts

def truncate_to_tokens(text: str, target_tokens: int, counter) -> str:
    """Greedy word-based truncation as a proxy for tokens."""
    toks = text.split()
    out: List[str] = []
    used = 0
    for w in toks:
        need = counter.count(w)
        if used + need > target_tokens:
            break
        out.append(w)
        used += need
    return " ".join(out)
