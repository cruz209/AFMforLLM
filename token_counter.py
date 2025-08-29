# token_counter.py
from __future__ import annotations

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None

class TokenCounter:
    """
    Token length estimator with optional tiktoken backend.
    Defaults to 'gpt-4o-mini' encoding, falls back to cl100k_base,
    then to naive whitespace count.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self._enc = None
        if tiktoken is not None:
            try:
                self._enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                try:
                    self._enc = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self._enc = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            try:
                return len(self._enc.encode(text))
            except Exception:
                pass
        # Fallback proxy
        return max(1, len(text.split()))
