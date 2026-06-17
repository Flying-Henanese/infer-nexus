"""Runtime worker client interface used to isolate Ray Serve handle calls."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from starlette.responses import Response

from infer_nexus.core.schemas import (
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
)
from infer_nexus.runtime.types import RuntimeTarget


class RuntimeWorkerClient(Protocol):
    """Client boundary between HTTP gateway workers and runtime worker processes."""

    async def chat_completion(
        self,
        *,
        target: RuntimeTarget,
        request: ChatCompletionsRequest,
    ) -> ChatCompletionsResponse | Response:
        """Execute one non-streaming chat completion in a runtime worker."""

    async def chat_completion_stream(
        self,
        *,
        target: RuntimeTarget,
        request: ChatCompletionsRequest,
    ) -> AsyncIterator[dict | bytes | str]:
        """Execute one streaming chat completion in a runtime worker."""

    async def embedding(
        self,
        *,
        target: RuntimeTarget,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse | Response:
        """Execute one embedding request in a runtime worker."""

    async def rerank(
        self,
        *,
        target: RuntimeTarget,
        request: RerankRequest,
    ) -> RerankResponse | Response:
        """Execute one rerank request in a runtime worker."""
