# demo.py
from __future__ import annotations

from token_counter import TokenCounter
from embedding import HashingEmbedder
from compression import HeuristicCompressor
from focus import FocusManager, FocusConfig

def main():
    tc = TokenCounter("gpt-4o-mini")
    emb = HashingEmbedder(dim=512)

    # Looser thresholds so hashing embeddings pass the bar
    cfg = FocusConfig(
        high_threshold=0.20,   # was 0.55
        mid_threshold=0.05,    # was 0.30
        recency_half_life=10,  # modest recency bias
        max_placeholder_tokens=12,
        default_compress_ratio=0.35,
        # If your FocusConfig doesn't have these, ignore. Otherwise:
        # min_topk_full=1,
        # min_topk_compressed=3,
    )

    fm = FocusManager(
        embedder=emb,
        token_counter=tc,
        compressor=HeuristicCompressor(tc),
        config=cfg
    )

    # Seed conversation
    fm.add_message("system", "You are a helpful assistant that writes short, punchy answers.")
    fm.add_message("user", "Plan a weekend trip to Chicago focused on architecture and deep-dish pizza.")
    fm.add_message("assistant", "Itinerary: river architecture tour, Art Institute, Lou Malnati's.")
    fm.add_message("user", "Keep total budget under $500 and use only public transit.")
    fm.add_message("assistant", "Updated: CTA passes, hostels, discount boat tour, free days.")
    fm.add_message("user", "Switch to Seattle coffee crawl and indie bookstores.")
    fm.add_message("assistant", "Coffee: Victrola, Vita, Milstead. Books: Elliott Bay, Twice Sold Tales.")
    fm.add_message("user", "My girlfriend is gluten-free—adjust food accordingly.")

    query = "Make the Seattle plan kid-friendly and still gluten-free."
    context, stats = fm.build_context(
        current_query=query,
        budget_tokens=220,
        system_preamble="Stay concise. Use bullets."
    )

    print("=== Context (role: content) ===")
    for r, c in context:
        preview = c.replace("\n", " ")
        print(f"{r}: {preview[:120]}{'...' if len(preview) > 120 else ''}")

    print("\n=== Stats ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
