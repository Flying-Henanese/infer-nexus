"""Native vLLM OpenAI embedding serving adapter."""

from __future__ import annotations

import inspect
import json
from typing import Any, Protocol


class OpenAIEmbeddingServingAdapter(Protocol):
    """Backend-local contract for vLLM OpenAI embeddings serving passthrough."""

    async def embedding(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Return a full OpenAI-compatible embeddings payload."""


class DynamicVLLMOpenAIEmbeddingServingAdapter:
    """Wrap native vLLM embedding serving objects behind the local adapter contract."""

    def __init__(
        self,
        *,
        embedding_request_cls: type[Any],
        serving_embedding: Any,
    ) -> None:
        self.embedding_request_cls = embedding_request_cls
        self.serving_embedding = serving_embedding

    def _build_request(self, request_payload: dict[str, Any]) -> Any:
        if hasattr(self.embedding_request_cls, "model_validate"):
            return self.embedding_request_cls.model_validate(request_payload)
        return self.embedding_request_cls(**request_payload)

    def _normalize_payload(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="json", exclude_none=True)

        body = getattr(payload, "body", None)
        if isinstance(body, bytes | bytearray):
            return json.loads(bytes(body).decode("utf-8"))
        if isinstance(body, str):
            return json.loads(body)

        raise RuntimeError(
            "vLLM OpenAI embeddings serving returned an unexpected payload type."
        )

    async def embedding(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        request = self._build_request(request_payload)

        if hasattr(self.serving_embedding, "create_embedding"):
            result = self.serving_embedding.create_embedding(request, raw_request=None)
        elif hasattr(self.serving_embedding, "create_pooling"):
            result = self.serving_embedding.create_pooling(request, raw_request=None)
        elif callable(self.serving_embedding):
            result = self.serving_embedding(request, raw_request=None)
        else:
            raise RuntimeError(
                "vLLM OpenAI embeddings serving object does not expose a compatible entrypoint."
            )

        if inspect.isawaitable(result):
            result = await result
        return self._normalize_payload(result)
