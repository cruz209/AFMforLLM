# focus.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from token_counter import TokenCounter
from embedding import Embedder
from compression import Compressor, HeuristicCompressor
from utils import cosine, truncate_to_tokens

class Fidelity:
    FULL = "FULL"
    COMPRESSED = "COMPRESSED"
    PLACEHOLDER = "PLACEHOLDER"

@dataclass
class MemoryItem:
    id: int
    role: str   # "user" | "assistant" | "system"
    content: str
    created_at_s: float = field(default_factory=lambda: time.time())
    token_len: int = 0
    embedding: Optional[List[float]] = None
    compression_state: str = Fidelity.FULL
    compressed_text: Optional[str] = None
    last_score: float = 0.0  # diagnostic

@dataclass
class FocusConfig:
    high_threshold: float = 0.55
    mid_threshold: float = 0.30
    recency_half_life: int = 12     # turns; lower = more recency bias
    max_placeholder_tokens: int = 12
    default_compress_ratio: float = 0.35  # target compressed size vs original

class FocusManager:
    """
    Dynamic focus controller:
    - Scores each past turn by (cosine similarity to query) * recency weight.
    - Assigns fidelity (FULL/COMPRESSED/PLACEHOLDER).
    - Packs messages under a token budget, expanding/contracting as needed.
    """

    def __init__(
        self,
        embedder: Embedder,
        token_counter: TokenCounter,
        compressor: Optional[Compressor] = None,
        config: Optional[FocusConfig] = None,
    ):
        self.embedder = embedder
        self.tc = token_counter
        self.compressor = compressor or HeuristicCompressor(token_counter)
        self.cfg = config or FocusConfig()
        self._items: List[MemoryItem] = []
        self._next_id = 1

    # ---- Public API ----

    def add_message(self, role: str, content: str) -> MemoryItem:
        item = MemoryItem(id=self._next_id, role=role, content=content)
        self._next_id += 1
        item.token_len = self.tc.count(content)
        item.embedding = self.embedder.encode(content)
        self._items.append(item)
        return item

    def build_context(
        self,
        current_query: str,
        budget_tokens: int,
        system_preamble: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, str]], Dict[str, float]]:
        """
        Returns (context_messages, stats).
        context_messages: list[(role, content)] ready for a chat API.
        """
        q_emb = self.embedder.encode(current_query)

        # Score items by relevance * recency
        n = len(self._items)
        scored: List[Tuple[float, MemoryItem]] = []
        for idx, it in enumerate(self._items):
            if not it.embedding:
                it.embedding = self.embedder.encode(it.content)
            sim = cosine(q_emb, it.embedding)
            turns_ago = (n - 1) - idx
            half = max(1, self.cfg.recency_half_life)
            recency_weight = 0.5 ** (turns_ago / half)
            score = max(0.0, sim) * (0.25 + 0.75 * recency_weight)
            it.last_score = score
            scored.append((score, it))

        # Plan fidelity by thresholds
        desired_map: Dict[int, str] = {}
        for score, it in scored:
            if score >= self.cfg.high_threshold:
                desired_map[it.id] = Fidelity.FULL
            elif score >= self.cfg.mid_threshold:
                desired_map[it.id] = Fidelity.COMPRESSED
            else:
                desired_map[it.id] = Fidelity.PLACEHOLDER

        messages: List[Tuple[str, str]] = []
        budget_left = budget_tokens

        def try_add(role: str, text: str) -> bool:
            nonlocal budget_left
            need = self.tc.count(text)
            if need <= budget_left:
                messages.append((role, text))
                budget_left -= need
                return True
            return False

        # Include system preamble first
        if system_preamble:
            try_add("system", system_preamble)

        used_tokens = raw_tokens = compressed_tokens = placeholder_tokens = 0
        expanded_count = compressed_count = stub_count = 0

        def compress_item(it: MemoryItem) -> str:
            target = max(1, int(it.token_len * self.cfg.default_compress_ratio))
            return self.compressor.compress(it.content, target, hint=current_query)

        # Emit in chronological order; fidelity decided by relevance plan
        ordered_items = sorted(self._items, key=lambda x: x.id)

        for it in ordered_items:
            desired = desired_map[it.id]

            if desired == Fidelity.FULL:
                if try_add(it.role, it.content):
                    used_tokens += it.token_len
                    raw_tokens += it.token_len
                    expanded_count += 1
                    it.compression_state = Fidelity.FULL
                    continue
                # fallback to compressed
                comp = it.compressed_text or compress_item(it)
                it.compressed_text = comp
                if try_add(it.role, comp):
                    ctok = self.tc.count(comp)
                    used_tokens += ctok
                    compressed_tokens += ctok
                    compressed_count += 1
                    it.compression_state = Fidelity.COMPRESSED
                    continue
                # fallback to stub
                stub = self._make_stub(it)
                if try_add(it.role, stub):
                    stok = self.tc.count(stub)
                    used_tokens += stok
                    placeholder_tokens += stok
                    stub_count += 1
                    it.compression_state = Fidelity.PLACEHOLDER

            elif desired == Fidelity.COMPRESSED:
                comp = it.compressed_text or compress_item(it)
                it.compressed_text = comp
                if try_add(it.role, comp):
                    ctok = self.tc.count(comp)
                    used_tokens += ctok
                    compressed_tokens += ctok
                    compressed_count += 1
                    it.compression_state = Fidelity.COMPRESSED
                else:
                    stub = self._make_stub(it)
                    if try_add(it.role, stub):
                        stok = self.tc.count(stub)
                        used_tokens += stok
                        placeholder_tokens += stok
                        stub_count += 1
                        it.compression_state = Fidelity.PLACEHOLDER

            else:  # PLACEHOLDER
                stub = self._make_stub(it)
                if try_add(it.role, stub):
                    stok = self.tc.count(stub)
                    used_tokens += stok
                    placeholder_tokens += stok
                    stub_count += 1
                    it.compression_state = Fidelity.PLACEHOLDER

        stats = {
            "budget": float(budget_tokens),
            "used": float(used_tokens),
            "raw_tokens": float(raw_tokens),
            "compressed_tokens": float(compressed_tokens),
            "placeholder_tokens": float(placeholder_tokens),
            "expanded_count": float(expanded_count),
            "compressed_count": float(compressed_count),
            "stub_count": float(stub_count),
            "items_total": float(len(self._items)),
            "items_planned_full": float(sum(1 for _, it in scored if desired_map[it.id] == Fidelity.FULL)),
            "items_planned_compressed": float(sum(1 for _, it in scored if desired_map[it.id] == Fidelity.COMPRESSED)),
            "items_planned_stub": float(sum(1 for _, it in scored if desired_map[it.id] == Fidelity.PLACEHOLDER)),
        }

        return messages, stats

    # ---- Internals ----

    def _make_stub(self, it: MemoryItem) -> str:
        head = it.content.strip().split("\n", 1)[0][:200]
        prefix = f"[ref #{it.id} • {it.role}] "
        room = max(0, self.cfg.max_placeholder_tokens - self.tc.count(prefix))
        if room <= 0:
            return prefix.strip()
        snippet = truncate_to_tokens(head, room, self.tc)
        return prefix + snippet

    def items(self) -> List[MemoryItem]:
        return list(self._items)
