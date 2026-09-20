"""ASGI middleware for process-local gateway worker admission."""

from __future__ import annotations

import json
from typing import Final

from starlette.types import ASGIApp, Receive, Scope, Send

from infer_nexus.api.request_logging_middleware import mark_admission_rejection
from infer_nexus.control.worker_admission import (
    WorkerAdmissionController,
    WorkerOverloadedError,
)
from infer_nexus.observability.logging import get_logger


PROTECTED_INFERENCE_PATHS: Final[frozenset[str]] = frozenset(
    {
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/rerank",
        "/rerank",
    }
)


class WorkerAdmissionMiddleware:
    """Reject excess inference requests before they enter FastAPI routing."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.logger = get_logger(__name__)

    @staticmethod
    def _controller(scope: Scope) -> WorkerAdmissionController:
        application = scope["app"]
        return application.state.worker_admission

    async def _send_overloaded(self, send: Send, *, retry_after_seconds: int) -> None:
        body = json.dumps(
            {
                "error": {
                    "message": "Gateway worker is overloaded.",
                    "type": "service_unavailable_error",
                    "code": "gateway_worker_overloaded",
                }
            },
            separators=(",", ":"),
        ).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"retry-after", str(retry_after_seconds).encode("ascii")),
        ]
        await send({"type": "http.response.start", "status": 503, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") not in PROTECTED_INFERENCE_PATHS:
            await self.app(scope, receive, send)
            return

        controller = self._controller(scope)
        try:
            await controller.acquire()
        except WorkerOverloadedError:
            mark_admission_rejection(
                "gateway_worker_overloaded",
                admission_layer="gateway_worker",
            )
            self.logger.warning(
                "admission.rejected",
                reason="gateway_worker_overloaded",
                retry_after_seconds=controller.retry_after_seconds,
            )
            await self._send_overloaded(
                send,
                retry_after_seconds=controller.retry_after_seconds,
            )
            return

        try:
            await self.app(scope, receive, send)
        finally:
            await controller.release()
