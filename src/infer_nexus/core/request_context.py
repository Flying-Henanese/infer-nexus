"""Allowlisted request metadata shared across gateway and runtime boundaries."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


FAILURE_STAGES = frozenset(
    {
        "request_validation",
        "catalog_lookup",
        "gateway_admission",
        "model_admission",
        "serve_handle",
        "model_backend",
        "proxy_upstream",
        "response_stream",
        "unknown",
    }
)
TIMEOUT_KINDS = frozenset(
    {
        "admission_wait",
        "serve_handle",
        "stream_idle",
        "stream_lifetime",
        "upstream_connect",
        "upstream_read",
        "unknown",
    }
)


@dataclass(slots=True)
class RequestContext:
    """Serializable request identity and low-cardinality routing metadata."""

    request_id: str
    method: str | None = None
    route: str | None = None
    model: str | None = None
    task: str | None = None
    stream: bool = False
    backend: str | None = None
    compat_mode: str | None = None
    serve_app: str | None = None
    deployment: str | None = None

    def to_log_fields(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "request_id": self.request_id,
                "method": self.method,
                "route": self.route,
                "model": self.model,
                "task": self.task,
                "stream": self.stream,
                "backend": self.backend,
                "compat_mode": self.compat_mode,
                "serve_app": self.serve_app,
                "deployment": self.deployment,
            }.items()
            if value is not None
        }


@dataclass(slots=True)
class RequestLifecycleState:
    """Mutable observations owned by the outer ASGI request lifecycle."""

    context: RequestContext
    started_at: float = field(default_factory=perf_counter)
    endpoint: str = "unknown"
    status_code: int | None = None
    upstream_status: int | None = None
    error_code: str | None = None
    admission_rejected: bool = False
    token_usage: Any = None
    exception: BaseException | None = None
    outcome: str | None = None
    response_complete: bool = False
    inflight_started: bool = False
    stream_terminal_logged: bool = False
    finalized: bool = False
    failure_stage: str | None = None
    timeout_kind: str | None = None
    admission_layer: str | None = None
    admission_wait_ms: float | None = None
    serve_handle_ms: float | None = None
    ttft_ms: float | None = None
    emitted_chunk_count: int = 0
    response_headers_sent: bool = False
    proxy_attempt_count: int | None = None
    proxy_failure_class: str | None = None
    usage_source: str | None = None


_CURRENT_REQUEST: ContextVar[RequestLifecycleState | None] = ContextVar(
    "infer_nexus_request_lifecycle",
    default=None,
)


def current_request_state() -> RequestLifecycleState | None:
    """Return the active request's mutable lifecycle state, if any."""
    return _CURRENT_REQUEST.get()


@contextmanager
def bind_request_state(state: RequestLifecycleState) -> Iterator[None]:
    """Bind one lifecycle state to the current async task and restore it on exit."""
    token = _CURRENT_REQUEST.set(state)
    try:
        yield
    finally:
        _CURRENT_REQUEST.reset(token)
