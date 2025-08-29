

# Adaptive Focus Memory (AFM)

**Adaptive Focus Memory (AFM)** is a lightweight Python framework for **token-efficient LLM conversations**.
Instead of dumping an entire chat history into the prompt, AFM dynamically decides whether each past message should be included as:

* **FULL** → kept verbatim
* **COMPRESSED** → summarized by an LLM
* **PLACEHOLDER** → left as a short stub reference

This lets you stretch small context windows, reduce costs, and keep conversations relevant without losing fidelity.

---

## ✨ Features

* **Dynamic context packing** → each message is scored and either kept full, compressed, or stubbed
* **LLM-based compression** → long messages summarized with OpenAI models
* **Embedding-based relevance** → past messages scored against the current query for semantic importance
* **Token budgeting** → ensures context fits within a user-defined token budget
* **Pluggable design** → supports both heuristic and LLM compression, with fallback offline mode

---

## 📂 Project Structure

```
afm/
  focus.py            # Core AFM logic (message scoring, packing)
  llm_compressor.py   # Compression using OpenAI LLMs
  openai_embedder.py  # Embedding wrapper for OpenAI
  embedding.py        # Base embedder interface
  token_counter.py    # Token counting utilities
  utils.py            # Shared helpers

examples/
  chat_llm.py         # Interactive LLM chatbot w/ AFM memory
  demo.py             # Offline demo with heuristic compression

requirements.txt
README.md
LICENSE
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
# Linux / Mac
export OPENAI_API_KEY="sk-..."

# Windows (Powershell)
setx OPENAI_API_KEY "sk-..."
```

### 3. Run the interactive chatbot

```bash
python examples/chat_llm.py
```

You should see:

```
🔑 Using OpenAIEmbedder + LLMCompressor
=== Interactive chat with Adaptive Focus Memory ===
```

---

## 📊 Example Output

AFM builds a packed context and reports token usage:

```
--- Final packed context ---
system    | You are a helpful assistant
user      | Should I use Klarna for a $23 lunch
assistant | Before financing a $23 lunch with Klarna, consider...

--- Token stats ---
budget: 800
used: 371
raw_tokens: 41
compressed_tokens: 330
items_total: 4
items_full: 1
items_compressed: 3
items_stubbed: 0
```

---

## 🛠 Use Cases

* Chatbots that need **long conversations** without blowing token budgets
* **Multi-agent systems** where each agent shares compressed context
* **Cost-saving AI infra** — reducing API spend in production pipelines

---

## 📜 License

Released under the **Apache 2.0 License** — free for commercial and research use.
Patent rights are preserved (see [LICENSE](LICENSE)).

---

⚡ **TL;DR:** AFM lets you talk to LLMs without wasting tokens — by remembering what matters, compressing what doesn’t, and stubbing the rest.

---
