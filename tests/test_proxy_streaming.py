"""Proxy streaming passthrough tests."""

import asyncio
from typing import Any

import httpx

from infer_nexus.catalog.models import ProxyConfig
from infer_nexus.runtime.executor import RuntimeExecutor


class FakeStreamingUpstreamResponse:
    """Small async response double for exercising SSE passthrough."""

    status_code = 200
    headers = {"content-type": "text/event-stream; charset=utf-8"}

    def __init__(self) -> None:
        self.closed = False

    async def aiter_bytes(self) -> Any:
        yield b"data: first\n\n"
        yield b"data: second\n\n"

    async def aclose(self) -> None:
        self.closed = True


class FakeAsyncClient:
    """httpx.AsyncClient replacement that returns a fake streaming response."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.response = FakeStreamingUpstreamResponse()
        self.closed = False

    def build_request(self, method: str, url: str, **kwargs: Any) -> httpx.Request:
        return httpx.Request(method, url, **kwargs)

    async def send(self, request: httpx.Request, *, stream: bool) -> FakeStreamingUpstreamResponse:
        assert stream is True
        return self.response

    async def aclose(self) -> None:
        self.closed = True


def test_proxy_streaming_preserves_sse_bytes_and_content_type(
    monkeypatch,
) -> None:
    """Streaming proxy should not parse or rewrite upstream SSE chunks."""
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    executor = RuntimeExecutor()
    proxy_config = ProxyConfig(upstream_base_url="http://upstream.local/v1")

    response = asyncio.run(
        executor._execute_proxy_stream(
            proxy_config=proxy_config,
            path="/chat/completions",
            payload={"model": "mineru", "stream": True},
            request_id="req-test",
        )
    )

    async def collect() -> bytes:
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return b"".join(chunks)

    body = asyncio.run(collect())

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert response.headers["X-Infer-Nexus-Request-ID"] == "req-test"
    assert body == b"data: first\n\ndata: second\n\n"
