# # chat_llm.py
# from __future__ import annotations
# import os
#
# from token_counter import TokenCounter
# from openai_embedder import OpenAIEmbedder
# from llm_compressor import LLMCompressor
# from focus import FocusManager, FocusConfig, Fidelity
#
# try:
#     from openai import OpenAI
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# except ImportError:
#     import openai
#     client = None
#
# def main():
#     tc = TokenCounter("gpt-4o-mini")
#     emb = OpenAIEmbedder("text-embedding-3-small")
#     compressor = LLMCompressor(tc, model="gpt-4o-mini")
#
#     cfg = FocusConfig(
#         high_threshold=0.45,
#         mid_threshold=0.25,
#         recency_half_life=10,
#         max_placeholder_tokens=12,
#         default_compress_ratio=0.35,
#     )
#     fm = FocusManager(embedder=emb, token_counter=tc, compressor=compressor, config=cfg)
#
#     print("=== Interactive chat with Adaptive Focus Memory ===")
#     print("Type 'exit' to finish and see memory summary.\n")
#
#     # Start with system
#     system_preamble = "You are a helpful, concise assistant."
#
#     while True:
#         user_in = input("You: ").strip()
#         if user_in.lower() in {"exit", "quit"}:
#             break
#
#         fm.add_message("user", user_in)
#
#         # Build context within token budget
#         ctx, stats = fm.build_context(
#             current_query=user_in,
#             budget_tokens=800,   # can tune this
#             system_preamble=system_preamble,
#         )
#
#         # Call OpenAI LLM with the packed context
#         if client:
#             out = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[{"role": r, "content": c} for r, c in ctx],
#                 temperature=0.7,
#             )
#             reply = out.choices[0].message.content
#         else:
#             out = openai.ChatCompletion.create(
#                 model="gpt-4o-mini",
#                 messages=[{"role": r, "content": c} for r, c in ctx],
#                 temperature=0.7,
#             )
#             reply = out["choices"][0]["message"]["content"]
#
#         print(f"Assistant: {reply}\n")
#         fm.add_message("assistant", reply)
#
#     # After chat ends, inspect what was stored
#     print("\n=== Conversation ended ===")
#     for it in fm.items():
#         print(f"[{it.id:02d}] {it.role:9s} | {it.compression_state:11s} | {it.content[:80]}{'...' if len(it.content)>80 else ''}")
#
# if __name__ == "__main__":
#     main()
# chat_llm.py
from __future__ import annotations
import os

from token_counter import TokenCounter
from embedding import HashingEmbedder
from compression import HeuristicCompressor
from focus import FocusManager, FocusConfig

USE_OPENAI = os.getenv("OPENAI_API_KEY") is not None
if USE_OPENAI:
    from openai_embedder import OpenAIEmbedder
    from llm_compressor import LLMCompressor
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except ImportError:
        import openai
        client = None
else:
    client = None

END_WORDS = {"exit", "quit", "end", "end the convo", "end conversation"}

def is_meaningful_query(text: str) -> bool:
    t = text.strip().lower()
    if not t: return False
    if t in END_WORDS: return False
    return len(t) >= 6  # avoid 1-2 word filler like "hi"

def main():
    tc = TokenCounter("gpt-4o-mini")

    if USE_OPENAI:
        print("🔑 Using OpenAIEmbedder + LLMCompressor")
        emb = OpenAIEmbedder("text-embedding-3-small")
        compressor = LLMCompressor(tc, model="gpt-4o-mini")
    else:
        print("⚡ No API key found → Using HashingEmbedder + HeuristicCompressor")
        emb = HashingEmbedder(dim=512)
        compressor = HeuristicCompressor(tc)

    cfg = FocusConfig(
        high_threshold=0.45 if USE_OPENAI else 0.20,
        mid_threshold=0.25 if USE_OPENAI else 0.05,
        recency_half_life=10,
        max_placeholder_tokens=12,
        default_compress_ratio=0.35,
    )

    fm = FocusManager(embedder=emb, token_counter=tc, compressor=compressor, config=cfg)
    system_preamble = "You are a helpful, concise assistant."

    print("=== Interactive chat with Adaptive Focus Memory ===")
    print("Type 'exit' to finish and see final packed context & memory summary.\n")

    last_meaningful_query = None

    while True:
        user_in = input("You: ").strip()
        if user_in.lower() in END_WORDS:
            break

        fm.add_message("user", user_in)
        if is_meaningful_query(user_in):
            last_meaningful_query = user_in

        ctx, stats = fm.build_context(
            current_query=user_in,
            budget_tokens=800,
            system_preamble=system_preamble,
        )

        if USE_OPENAI:
            if client:
                out = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": r, "content": c} for r, c in ctx],
                    temperature=0.7,
                )
                reply = out.choices[0].message.content
            else:
                out = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": r, "content": c} for r, c in ctx],
                    temperature=0.7,
                )
                reply = out["choices"][0]["message"]["content"]
        else:
            reply = f"[stub reply] You said: {user_in}"

        print(f"Assistant: {reply}\n")
        fm.add_message("assistant", reply)

    # ---------- POST-CHAT REPORT ----------
    print("\n=== Conversation ended ===")

    # 1) Final packed context for your last meaningful question (not 'end the convo')
    if last_meaningful_query is None:
        # fallback to last user message if all were trivial
        # (this matches your earlier run)
        last_meaningful_query = "end the convo"

    final_ctx, final_stats = fm.build_context(
        current_query=last_meaningful_query,
        budget_tokens=800,
        system_preamble=system_preamble,
    )

    print(f"\n--- Final packed context for query: \"{last_meaningful_query}\" ---")
    for role, content in final_ctx:
        preview = content.replace("\n", " ")
        print(f"{role:9s} | {preview[:160]}{'...' if len(preview) > 160 else ''}")

    print("\n--- Token stats ---")
    for k, v in final_stats.items():
        print(f"{k}: {v}")

    # 2) Memory summary with fidelity states (FULL/COMPRESSED/PLACEHOLDER)
    print("\n--- Memory summary ---")
    for it in fm.items():
        print(f"[{it.id:02d}] {it.role:9s} | {it.compression_state:11s} | {it.content[:80]}{'...' if len(it.content)>80 else ''}")

if __name__ == "__main__":
    main()
