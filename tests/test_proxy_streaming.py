"""Proxy streaming passthrough tests."""

import asyncio
import logging
from typing import Any

import httpx
import pytest

from infer_nexus.catalog.models import ProxyConfig
from infer_nexus.core.request_context import (
    RequestContext,
    RequestLifecycleState,
    bind_request_state,
)
from infer_nexus.observability.metrics import render_prometheus_metrics
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
            model_label="mineru",
            stream_start_time=0.0,
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
    metrics = render_prometheus_metrics()[0].decode("utf-8")
    assert 'infer_nexus_stream_ttft_seconds_count{model="mineru"}' in metrics
    assert 'infer_nexus_stream_tpot_seconds_count{model="mineru"}' in metrics
    assert 'infer_nexus_stream_completions_total{model="mineru",status="success"}' in metrics


def test_proxy_request_forwards_request_id_and_logs_safe_upstream_failure(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Proxy failures expose the canonical ID and safe host/status metadata only."""
    captured: dict[str, Any] = {}

    class ErrorResponseClient:
        def __init__(self, *, timeout: httpx.Timeout) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "ErrorResponseClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            json: dict[str, Any],
            headers: dict[str, str],
        ) -> httpx.Response:
            captured["url"] = url
            captured["headers"] = headers
            return httpx.Response(
                503,
                request=httpx.Request("POST", url),
                headers={"content-type": "application/json"},
                json={"error": "unavailable"},
            )

    monkeypatch.setattr(httpx, "AsyncClient", ErrorResponseClient)
    config = ProxyConfig(
        upstream_base_url="https://user:secret@upstream.local/v1?token=query-secret"
    )
    state = RequestLifecycleState(context=RequestContext(request_id="req-proxy-1"))
    executor = RuntimeExecutor()

    async def request() -> httpx.Response:
        return await executor._request_proxy(
            proxy_config=config,
            path="/chat/completions",
            payload={"model": "test", "messages": [{"content": "private"}]},
            request_id="req-proxy-1",
        )

    with bind_request_state(state), caplog.at_level(
        logging.WARNING,
        logger="infer_nexus.runtime.executor",
    ):
        response = asyncio.run(request())

    assert response.status_code == 503
    assert captured["headers"]["X-Request-ID"] == "req-proxy-1"
    assert "Authorization" not in captured["headers"]
    assert state.upstream_status == 503
    event = next(
        record
        for record in caplog.records
        if getattr(record, "structured_event", None) == "proxy.request.failed"
    )
    fields = event.structured_fields
    assert fields["upstream_host"] == "upstream.local"
    assert fields["upstream_status"] == 503
    assert fields["error_code"] == "upstream_http_503"
    assert "secret" not in repr(fields)
    assert "query-secret" not in repr(fields)
