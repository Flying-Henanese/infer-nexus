"""OpenAI-compatible benchmark HTTP client."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from infer_nexus.benchmark.models import BenchmarkRequest, BenchmarkResult
from infer_nexus.benchmark.tokenizer import EstimatedTokenizer


class OpenAICompatibleBenchmarkClient:
    """Minimal OpenAI-compatible chat completions benchmark client."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        tokenizer: EstimatedTokenizer | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.tokenizer = tokenizer or EstimatedTokenizer()

    async def complete(
        self,
        request: BenchmarkRequest,
        *,
        timeout_seconds: float,
    ) -> BenchmarkResult:
        """Execute a chat completion request."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "stream": request.stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        timeout = httpx.Timeout(timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            if request.stream:
                return await self._complete_streaming(client, headers, payload)
            return await self._complete_non_streaming(client, headers, payload)

    async def _complete_non_streaming(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> BenchmarkResult:
        """Execute a non-streaming chat completion request."""
        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        end_time = time.perf_counter()
        text = _extract_non_streaming_text(data)
        usage = data.get("usage") or {}
        return BenchmarkResult(
            first_token_time=end_time,
            end_time=end_time,
            output_text=text,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    async def _complete_streaming(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> BenchmarkResult:
        """Execute a streaming chat completion request and record chunk timing."""
        text_parts: list[str] = []
        chunk_times: list[float] = []
        usage: dict[str, Any] = {}

        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line.removeprefix("data:").strip()
                if raw == "[DONE]":
                    break
                if not raw:
                    continue
                now = time.perf_counter()
                chunk_times.append(now)
                event = json.loads(raw)
                usage = event.get("usage") or usage
                text = _extract_streaming_text(event)
                if text:
                    text_parts.append(text)

        end_time = time.perf_counter()
        output_text = "".join(text_parts)
        return BenchmarkResult(
            first_token_time=chunk_times[0] if chunk_times else end_time,
            end_time=end_time,
            output_text=output_text,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens")
            or self.tokenizer.count_text(output_text),
            chunk_times=chunk_times,
        )


def _extract_non_streaming_text(data: dict[str, Any]) -> str:
    """Extract text from a non-streaming OpenAI-compatible response."""
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _extract_streaming_text(data: dict[str, Any]) -> str:
    """Extract text delta from a streaming OpenAI-compatible event."""
    choices = data.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""
