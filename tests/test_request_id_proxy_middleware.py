from __future__ import annotations

import asyncio
import re

from infer_nexus.runtime.request_id_proxy_middleware import RequestIdProxyMiddleware


def test_proxy_replaces_invalid_request_id_before_forwarding() -> None:
    observed_headers: list[tuple[bytes, bytes]] = []
    response_messages: list[dict[str, object]] = []

    async def app(scope, receive, send) -> None:
        nonlocal observed_headers
        observed_headers = list(scope["headers"])
        request_id = next(
            value
            for name, value in observed_headers
            if name.lower() == b"x-request-id"
        )
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"x-request-id", request_id)],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        response_messages.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/models",
        "headers": [(b"x-request-id", b"invalid id")],
    }

    asyncio.run(RequestIdProxyMiddleware(app)(scope, receive, send))

    forwarded_ids = [
        value for name, value in observed_headers if name.lower() == b"x-request-id"
    ]
    returned_ids = [
        value
        for name, value in response_messages[0]["headers"]
        if name.lower() == b"x-request-id"
    ]
    assert len(forwarded_ids) == 1
    assert forwarded_ids[0] != b"invalid id"
    assert re.fullmatch(rb"[0-9a-f]{32}", forwarded_ids[0])
    assert returned_ids == forwarded_ids


def test_proxy_collapses_duplicate_request_ids_before_forwarding() -> None:
    observed_headers: list[tuple[bytes, bytes]] = []
    response_messages: list[dict[str, object]] = []

    async def app(scope, receive, send) -> None:
        nonlocal observed_headers
        observed_headers = list(scope["headers"])
        request_id = next(
            value
            for name, value in observed_headers
            if name.lower() == b"x-request-id"
        )
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"x-request-id", request_id)],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        response_messages.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/models",
        "headers": [
            (b"x-request-id", b"first-id"),
            (b"X-Request-ID", b"second-id"),
        ],
    }

    asyncio.run(RequestIdProxyMiddleware(app)(scope, receive, send))

    forwarded_ids = [
        value for name, value in observed_headers if name.lower() == b"x-request-id"
    ]
    returned_ids = [
        value
        for name, value in response_messages[0]["headers"]
        if name.lower() == b"x-request-id"
    ]
    assert len(forwarded_ids) == 1
    assert forwarded_ids[0] not in {b"first-id", b"second-id"}
    assert re.fullmatch(rb"[0-9a-f]{32}", forwarded_ids[0])
    assert returned_ids == forwarded_ids
