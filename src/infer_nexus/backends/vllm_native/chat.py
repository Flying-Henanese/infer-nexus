"""Native vLLM OpenAI chat serving adapter."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any, Protocol


class OpenAIChatServingAdapter(Protocol):
    """Backend-local contract for vLLM OpenAI serving passthrough."""

    async def chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Return a full OpenAI-compatible chat completion payload."""

    async def chat_completion_stream(
        self,
        request_payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
        """Yield OpenAI-compatible chat completion chunks."""


class DynamicVLLMOpenAIChatServingAdapter:
    """Wrap native vLLM serving objects behind the local adapter contract."""

    def __init__(
        self,
        *,
        chat_request_cls: type[Any],
        serving_chat: Any,
    ) -> None:
        self.chat_request_cls = chat_request_cls
        self.serving_chat = serving_chat

    def _build_request(self, request_payload: dict[str, Any]) -> Any:
        if hasattr(self.chat_request_cls, "model_validate"):
            return self.chat_request_cls.model_validate(request_payload)
        return self.chat_request_cls(**request_payload)

    def _normalize_payload(self, payload: Any) -> dict[str, Any] | bytes | str:
        if isinstance(payload, dict | bytes | str):
            return payload
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="json", exclude_none=True)
        return payload

    async def chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        request = self._build_request(request_payload)
        result = self.serving_chat.create_chat_completion(request, raw_request=None)
        if inspect.isawaitable(result):
            result = await result
        if hasattr(result, "__aiter__"):
            raise RuntimeError(
                "vLLM OpenAI serving returned a stream for a non-streaming chat request."
            )
        normalized = self._normalize_payload(result)
        if not isinstance(normalized, dict):
            raise RuntimeError(
                "vLLM OpenAI serving returned an unexpected non-streaming chat payload type."
            )
        return normalized

    async def chat_completion_stream(
        self,
        request_payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
        request = self._build_request(request_payload)
        result = self.serving_chat.create_chat_completion(request, raw_request=None)
        if inspect.isawaitable(result):
            result = await result

        if hasattr(result, "__aiter__"):
            async for chunk in result:
                yield self._normalize_payload(chunk)
            return

        yield self._normalize_payload(result)
