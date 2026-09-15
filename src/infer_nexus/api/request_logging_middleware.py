"""Outer ASGI request identity, logging, and metric lifecycle."""

from __future__ import annotations

import asyncio
import json
import random
import re
from time import perf_counter
from typing import Any
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from infer_nexus.core.request_context import (
    RequestContext,
    RequestLifecycleState,
    bind_request_state,
)
from infer_nexus.observability.logging import bind_log_context, get_logger
from infer_nexus.observability.metrics import GATEWAY_METRICS

logger = get_logger(__name__)
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_LEGACY_REQUEST_ID_HEADER = b"x-infer-nexus-request-id"
_REQUEST_ID_HEADER = b"x-request-id"
_SUPPRESSED_SUCCESS_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})


def _validated_request_id(value: str | None) -> str | None:
    if value is None or _REQUEST_ID_PATTERN.fullmatch(value) is None:
        return None
    return value


def _client_request_id(scope: Scope) -> str | None:
    for name, value in scope.get("headers", []):
        if name.lower() == _REQUEST_ID_HEADER:
            try:
                return _validated_request_id(value.decode("ascii"))
            except UnicodeDecodeError:
                return None
    return None


def _serve_request_id() -> str | None:
    """Read Ray Serve's proxy request ID behind one version-sensitive adapter."""
    try:
        from ray.serve.context import _get_serve_request_context

        context = _get_serve_request_context()
        return _validated_request_id(getattr(context, "request_id", None))
    except (ImportError, RuntimeError, AttributeError):
        return None


def _request_identity(scope: Scope) -> tuple[str, bool]:
    """Return the canonical ID and whether the Ray proxy owns the response header."""
    serve_request_id = _serve_request_id()
    request_id = _client_request_id(scope) or serve_request_id or uuid4().hex
    return request_id, serve_request_id is not None


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    template = getattr(route, "path_format", None) or getattr(route, "path", None)
    if isinstance(template, str) and template:
        return template
    path = scope.get("path", "")
    if path in _SUPPRESSED_SUCCESS_PATHS or path in {
        "/v1/models",
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/rerank",
        "/rerank",
    }:
        return path
    return "unmatched"


def _response_error_code(body: bytes) -> str | None:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) and code else None


def mark_request_error(error_code: str) -> None:
    """Attach a stable domain error code to the active request lifecycle."""
    state = _current_state()
    if state is not None:
        state.error_code = error_code


def _current_state() -> RequestLifecycleState | None:
    from infer_nexus.core.request_context import current_request_state

    return current_request_state()


def _outcome(state: RequestLifecycleState) -> str:
    if state.outcome == "cancelled":
        return "cancelled"
    if isinstance(state.exception, asyncio.CancelledError):
        return "cancelled"
    if state.exception is not None and type(state.exception).__name__ in {
        "ClientDisconnect",
        "DisconnectError",
    }:
        return "cancelled"
    if state.exception is not None:
        return "timeout" if state.error_code == "upstream_timeout" else "error"
    if not state.response_complete:
        return "error"
    if state.status_code is not None and state.status_code >= 500:
        return "timeout" if state.error_code == "upstream_timeout" else "error"
    if state.status_code is not None and state.status_code >= 400:
        if state.status_code == 429 or state.error_code in {
            "gateway_worker_overloaded",
            "admission_rejected",
        }:
            return "rejected"
        return "error"
    return "success"


def _record_request_metrics(state: RequestLifecycleState, duration_seconds: float) -> None:
    context = state.context
    if context.task is None:
        return
    model = context.model or "unknown"
    status = state.outcome if state.outcome in {
        "success",
        "error",
        "rejected",
        "cancelled",
        "timeout",
    } else "error"
    GATEWAY_METRICS.observe_request(
        model=model,
        task=context.task,
        endpoint=state.endpoint,
        status=status,
        latency_seconds=duration_seconds,
    )
    if state.error_code:
        GATEWAY_METRICS.observe_error(
            model=model,
            task=context.task,
            code=state.error_code,
        )
    elif state.status_code == 422:
        GATEWAY_METRICS.observe_error(
            model=model,
            task=context.task,
            code="invalid_request",
        )
    if state.outcome == "rejected":
        GATEWAY_METRICS.observe_admission_rejection(
            model=model,
            reason=state.error_code or "admission_rejected",
        )
    GATEWAY_METRICS.observe_token_usage(
        model=model,
        task=context.task,
        usage=state.token_usage,
    )
    if state.inflight_started:
        GATEWAY_METRICS.dec_inflight(
            model=model,
            task=context.task,
            endpoint=state.endpoint,
        )


def _finalize(state: RequestLifecycleState, settings: Any) -> None:
    if state.finalized:
        return
    state.finalized = True
    duration_seconds = max(0.0, perf_counter() - state.started_at)
    state.outcome = _outcome(state)
    context = state.context
    _record_request_metrics(state, duration_seconds)

    path = context.route or ""
    if path in _SUPPRESSED_SUCCESS_PATHS and state.outcome == "success":
        return
    slow_request = duration_seconds * 1000 >= settings.slow_request_ms
    if (
        state.outcome == "success"
        and not slow_request
        and random.random() >= settings.success_sample_rate
    ):
        return

    event_fields = {
        **context.to_log_fields(),
        "outcome": state.outcome,
        "status_code": state.status_code or (500 if state.exception else 200),
        "duration_ms": round(duration_seconds * 1000, 3),
    }
    if state.error_code:
        event_fields["error_code"] = state.error_code
    if state.upstream_status is not None:
        event_fields["upstream_status"] = state.upstream_status
    if slow_request:
        event_fields["slow"] = True

    if state.outcome in {"success", "cancelled"}:
        logger.info("request.completed", **event_fields)
    elif state.exception is not None:
        if state.stream_terminal_logged:
            logger.error("request.failed", **event_fields)
        else:
            logger.error(
                "request.failed",
                exc_info=(type(state.exception), state.exception, state.exception.__traceback__),
                **event_fields,
            )
    elif state.outcome == "rejected":
        logger.warning("request.completed", **event_fields)
    else:
        logger.warning("request.completed", **event_fields)


class RequestLoggingMiddleware:
    """Wrap the whole FastAPI ASGI app so even generated 500s gain a request ID."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __getattr__(self, name: str) -> Any:
        # Keep FastAPI's state/router surface available to lifespan and test clients.
        try:
            app = object.__getattribute__(self, "app")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(app, name)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id, ray_proxy_owns_request_id_header = _request_identity(scope)
        path = scope.get("path", "")
        context = RequestContext(
            request_id=request_id,
            method=scope.get("method"),
            route=path if path in _SUPPRESSED_SUCCESS_PATHS else None,
        )
        state = RequestLifecycleState(context=context)
        scope.setdefault("state", {})["request_lifecycle"] = state
        settings = self.app.state.settings.observability.logging
        error_body = bytearray()
        json_error_response = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal json_error_response
            if message["type"] == "http.response.start":
                state.status_code = int(message["status"])
                state.context.route = _route_template(scope)
                content_type = next(
                    (
                        value.split(b";", 1)[0].lower()
                        for name, value in message.get("headers", [])
                        if name.lower() == b"content-type"
                    ),
                    b"",
                )
                json_error_response = (
                    state.status_code >= 400 and content_type == b"application/json"
                )
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() not in {_REQUEST_ID_HEADER, _LEGACY_REQUEST_ID_HEADER}
                ]
                if not ray_proxy_owns_request_id_header:
                    headers.append((_REQUEST_ID_HEADER, request_id.encode("ascii")))
                content_type = next(
                    (
                        value.decode("latin-1").lower()
                        for name, value in message.get("headers", [])
                        if name.lower() == b"content-type"
                    ),
                    "",
                )
                if state.context.stream or "text/event-stream" in content_type:
                    headers.append((_LEGACY_REQUEST_ID_HEADER, request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)
            if (
                message["type"] == "http.response.body"
                and json_error_response
                and len(error_body) < 65_536
            ):
                body = message.get("body", b"")
                if isinstance(body, bytes):
                    error_body.extend(body[: 65_536 - len(error_body)])
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                state.response_complete = True
                if json_error_response:
                    state.error_code = state.error_code or _response_error_code(bytes(error_body))

        with bind_request_state(state), bind_log_context(
            request_id=request_id,
            method=context.method,
        ):
            try:
                await self.app(scope, receive, send_with_request_id)
            except asyncio.CancelledError as exc:
                state.exception = exc
                raise
            except BaseException as exc:
                state.exception = exc
                if type(exc).__name__ in {"ClientDisconnect", "DisconnectError"} or (
                    state.context.stream and isinstance(exc, OSError)
                ):
                    state.outcome = "cancelled"
                raise
            finally:
                _finalize(state, settings)
