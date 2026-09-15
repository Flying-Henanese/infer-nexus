"""Normalize request IDs before Ray Serve's proxy middleware handles them."""

from __future__ import annotations

from starlette.middleware import Middleware
from starlette.types import ASGIApp, Receive, Scope, Send

from infer_nexus.core.request_ids import (
    REQUEST_ID_HEADER,
    new_request_id,
    validate_request_id,
)


class RequestIdProxyMiddleware:
    """Give Ray Serve exactly one validated request ID at its HTTP boundary."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        supplied_ids = [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == REQUEST_ID_HEADER
        ]
        request_id = None
        if len(supplied_ids) == 1:
            try:
                request_id = validate_request_id(supplied_ids[0].decode("ascii"))
            except UnicodeDecodeError:
                pass

        canonical_id = request_id or new_request_id()
        headers = [
            (name, value)
            for name, value in scope.get("headers", [])
            if name.lower() != REQUEST_ID_HEADER
        ]
        headers.append((REQUEST_ID_HEADER, canonical_id.encode("ascii")))
        normalized_scope = {**scope, "headers": headers}
        await self.app(normalized_scope, receive, send)


def ray_serve_request_id_middleware() -> Middleware:
    """Build the middleware entry Ray Serve installs outside its request-ID handler."""
    return Middleware(RequestIdProxyMiddleware)
