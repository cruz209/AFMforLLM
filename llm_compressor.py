# llm_compressor.py
import os
from typing import Optional
from compression import Compressor
from token_counter import TokenCounter

try:
    from openai import OpenAI
    _OPENAI_V1 = True
except ImportError:
    import openai
    _OPENAI_V1 = False

_SYSTEM = (
    "You are a compression module. Rewrite the provided text to preserve key facts "
    "and task-relevant details while staying under the specified token budget."
)

class LLMCompressor(Compressor):
    """LLM-backed compressor that targets an approximate token budget."""
    def __init__(self, token_counter: TokenCounter, model: str = "gpt-4o-mini"):
        super().__init__(token_counter)
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        if _OPENAI_V1:
            self.client = OpenAI(api_key=api_key)
        else:
            openai.api_key = api_key
            self.client = None

    def compress(self, text: str, target_tokens: int, hint: Optional[str] = None) -> str:
        if self.tc.count(text) <= target_tokens:
            return text

        prompt = (
            f"Target token budget: ~{target_tokens} tokens.\n"
            f"Compression hint: {hint or 'N/A'}\n\n{text}"
        )

        if _OPENAI_V1:
            out = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": _SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            comp = out.choices[0].message.content.strip()
        else:
            out = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": _SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            comp = out["choices"][0]["message"]["content"].strip()

        return comp
