"""Native vLLM OpenAI embedding serving adapter contracts."""

from __future__ import annotations

from typing import Any, Protocol


class OpenAIEmbeddingServingAdapter(Protocol):
    """Backend-local contract for vLLM OpenAI embeddings serving passthrough."""

    async def embedding(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Return a full OpenAI-compatible embeddings payload."""
