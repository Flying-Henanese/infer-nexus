"""Lightweight token counting for benchmark runs."""

from __future__ import annotations

from typing import Any


class EstimatedTokenizer:
    """Dependency-free tokenizer approximation used when model tokenizers are unavailable."""

    def count_text(self, text: str) -> int:
        """Estimate token count from text."""
        if not text:
            return 0
        return max(1, len(text.split()))

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Estimate token count for OpenAI-style chat messages."""
        total = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total += self.count_text(content)
            elif isinstance(content, list):
                total += sum(self.count_text(str(part)) for part in content)
            else:
                total += self.count_text(str(content))
        return total
