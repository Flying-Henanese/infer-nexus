"""Application logging interface and private serialization implementation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import math
import os
import sys
import threading
import traceback
from typing import Any

from infer_nexus.core.config import LoggingSettings


_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("infer_nexus_log_context", default={})
_CONFIG_LOCK = threading.RLock()
_PROCESS_IDENTITY: dict[str, Any] = {}
_INCLUDE_TRACEBACK = True
_SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "password",
    "secret",
    "credential",
    "access_token",
    "refresh_token",
    "prompt",
    "content",
    "text",
    "messages",
    "embedding_input",
    "input",
    "rerank_query",
    "rerank_documents",
    "document",
    "query",
    "documents",
    "token_ids",
    "cookies",
    "cookie",
    "headers",
    "request_headers",
    "client_headers",
    "request_body",
    "response_body",
    "extra_body",
    "body",
    "upstream_url",
}
_RESERVED_FIELDS = {
    "timestamp",
    "level",
    "event",
    "message",
    "logger",
    "service",
    "service_version",
    "environment",
    "process_role",
    "pid",
    "source",
    "exception",
    "request_id",
}
_LOGGING_KWARGS = {"exc_info", "stack_info", "stacklevel"}


def _sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS


def _safe_value(value: Any) -> Any:
    """Convert supported values to JSON-safe data without serializing objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Enum):
        return _safe_value(value.value)
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            sanitized[normalized_key] = (
                "[REDACTED]" if _sensitive_key(normalized_key) else _safe_value(item)
            )
        return sanitized
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_value(item) for item in value]
    return f"<{type(value).__name__}>"


def _source_for_logger(logger_name: str) -> str:
    if logger_name == "infer_nexus" or logger_name.startswith("infer_nexus."):
        return "infer_nexus"
    if logger_name.startswith(("ray.serve", "ray.serve_logging")):
        return "ray_serve"
    if logger_name.startswith("ray"):
        return "ray_core"
    if logger_name == "vllm" or logger_name.startswith("vllm."):
        return "vllm"
    return "third_party"


def _exception_payload(exc_info: Any) -> dict[str, str] | None:
    if not exc_info:
        return None
    exc_type, _, tb = exc_info
    if exc_type is None:
        return None
    result = {"type": exc_type.__name__}
    if _INCLUDE_TRACEBACK:
        frames = "\n".join(
            f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}'
            for frame in traceback.extract_tb(tb)
        )
        result["stack_trace"] = (
            f"Traceback (most recent call last):\n{frames}" if frames else ""
        )
    return result


class _JsonFormatter(logging.Formatter):
    """Serialize each record to exactly one JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        structured = getattr(record, "structured_fields", {})
        structured_record = hasattr(record, "structured_event")
        event = getattr(record, "structured_event", "log.record")
        message = (
            getattr(record, "structured_message", None)
            if structured_record
            else record.getMessage()
        )
        exception = _exception_payload(record.exc_info)
        if exception is None and record.stack_info:
            exception = {"stack_trace": record.stack_info}

        protected = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "event": str(event),
            "logger": record.name,
            **{
                key: value
                for key, value in _PROCESS_IDENTITY.items()
                if key in _RESERVED_FIELDS
            },
            "source": _source_for_logger(record.name),
        }
        request_id = _CONTEXT.get().get("request_id")
        if request_id is not None:
            protected["request_id"] = request_id
        fields: dict[str, Any] = {}
        for identity_field, value in _PROCESS_IDENTITY.items():
            if identity_field not in _RESERVED_FIELDS:
                fields[identity_field] = value
        for context_field, value in _CONTEXT.get().items():
            if context_field not in _RESERVED_FIELDS:
                fields[context_field] = value
        if isinstance(structured, Mapping):
            for field_name, value in structured.items():
                if field_name not in _RESERVED_FIELDS:
                    fields[field_name] = value
        result = dict(fields)
        result.update(protected)
        if message:
            result["message"] = message
        if exception is not None:
            result["exception"] = exception
        sanitized = _safe_value(result)
        return json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class _ConsoleFormatter(logging.Formatter):
    """Render concise key-value console records for local development."""

    def format(self, record: logging.LogRecord) -> str:
        structured = getattr(record, "structured_fields", {})
        structured_record = hasattr(record, "structured_event")
        event = getattr(record, "structured_event", "log.record")
        message = (
            getattr(record, "structured_message", None)
            if structured_record
            else record.getMessage()
        )
        fields: dict[str, Any] = dict(_CONTEXT.get())
        request_id = _CONTEXT.get().get("request_id")
        if isinstance(structured, Mapping):
            fields.update(structured)
        if request_id is not None:
            fields["request_id"] = request_id
        fields = {
            key: value
            for key, value in fields.items()
            if key not in _RESERVED_FIELDS or key == "request_id"
        }
        field_text = " ".join(
            f"{key}={json.dumps(_safe_value(value), ensure_ascii=False)}"
            for key, value in sorted(fields.items())
        )
        prefix = (
            f"{datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec='seconds')} "
            f"{record.levelname:<8} {record.name} {event}"
        )
        suffix = f" {message}" if message else ""
        if field_text:
            suffix = f"{suffix} {field_text}"
        exception = _exception_payload(record.exc_info)
        if exception and exception.get("stack_trace"):
            suffix = f"{suffix}\n{exception['stack_trace']}"
        return f"{prefix}{suffix}"


class _StructuredLogger(logging.LoggerAdapter):
    """Allow application events to pass structured fields as keyword arguments."""

    def log(self, level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
        if not self.isEnabledFor(level):
            return
        logging_kwargs = {
            key: kwargs.pop(key)
            for key in tuple(kwargs)
            if key in _LOGGING_KWARGS
        }
        explicit_extra = kwargs.pop("extra", None)
        fields: dict[str, Any] = {}
        if isinstance(self.extra, Mapping):
            fields.update(self.extra)
        if isinstance(explicit_extra, Mapping):
            fields.update(explicit_extra)
        fields.update(kwargs)

        if isinstance(msg, str) and "." in msg and not args:
            event = msg
            message = None
        else:
            event = fields.pop("event", "log.record")
            try:
                message = str(msg) % args if args else str(msg)
            except Exception:
                message = str(msg)
        stacklevel = logging_kwargs.pop("stacklevel", 1)
        self.logger.log(
            level,
            "",
            extra={
                "structured_event": event,
                "structured_message": message,
                "structured_fields": fields,
            },
            stacklevel=stacklevel + 1,
            **logging_kwargs,
        )


def configure_logging(settings: LoggingSettings, process_identity: Mapping[str, Any]) -> None:
    """Configure one process's standard logging handlers and stable identity."""
    global _INCLUDE_TRACEBACK, _PROCESS_IDENTITY
    with _CONFIG_LOCK:
        identity = dict(process_identity)
        identity.setdefault("service", "infer-nexus")
        identity.setdefault("service_version", "0.1.0")
        identity.setdefault("environment", os.getenv("INFER_NEXUS_ENVIRONMENT", "development"))
        identity.setdefault("process_role", "cli")
        identity.setdefault("pid", os.getpid())
        _PROCESS_IDENTITY = identity
        _INCLUDE_TRACEBACK = settings.include_traceback

        root = logging.getLogger()
        for handler in tuple(root.handlers):
            root.removeHandler(handler)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            _JsonFormatter() if settings.format == "json" else _ConsoleFormatter()
        )
        root.addHandler(handler)
        root.setLevel(getattr(logging, settings.level))

        for logger_name, level_name in settings.named_levels.items():
            logging.getLogger(logger_name).setLevel(getattr(logging, level_name))


def get_logger(module_name: str) -> logging.LoggerAdapter:
    """Return an application logger that accepts event fields as keyword arguments."""
    return _StructuredLogger(logging.getLogger(module_name), {})


@contextmanager
def bind_log_context(**fields: Any) -> Iterator[None]:
    """Bind request or operation fields for the current synchronous/async context."""
    merged = dict(_CONTEXT.get())
    merged.update(fields)
    token = _CONTEXT.set(merged)
    try:
        yield
    finally:
        _CONTEXT.reset(token)
