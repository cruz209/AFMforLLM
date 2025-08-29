# compression.py
from __future__ import annotations

from typing import List, Optional, Tuple
from token_counter import TokenCounter
from utils import split_sentences, truncate_to_tokens, simple_tokenize

class Compressor:
    """Interface for compressing long text to fit a target token budget."""
    def __init__(self, token_counter: TokenCounter):
        self.tc = token_counter

    def compress(self, text: str, target_tokens: int, hint: Optional[str] = None) -> str:  # pragma: no cover
        raise NotImplementedError

class HeuristicCompressor(Compressor):
    """
    Extractive-ish compressor with no external models.
    Scores sentences by hint-overlap, length prior, and early-position bias.
    """
    def compress(self, text: str, target_tokens: int, hint: Optional[str] = None) -> str:
        if self.tc.count(text) <= target_tokens:
            return text

        sentences = split_sentences(text)
        if not sentences:
            return truncate_to_tokens(text, target_tokens, self.tc)

        hint_toks = set(simple_tokenize(hint or ""))
        scored: List[Tuple[float, str]] = []
        for idx, s in enumerate(sentences):
            toks = set(simple_tokenize(s))
            overlap = len(toks & hint_toks)
            len_penalty = max(1, len(toks)) ** 0.15
            pos_bias = 1.0 / (1 + idx * 0.05)
            score = (1 + overlap) * pos_bias / len_penalty
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[str] = []
        used = 0
        for _, s in scored:
            need = self.tc.count(s)
            if used + need > target_tokens:
                continue
            out.append(s)
            used += need
            if used >= target_tokens:
                break

        if not out:
            return truncate_to_tokens(sentences[0], target_tokens, self.tc)
        return " ".join(out)
