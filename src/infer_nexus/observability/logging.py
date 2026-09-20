"""Application logging interface and private serialization implementation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
import traceback
from typing import Any
from uuid import uuid4

from infer_nexus.core.config import LoggingSettings


_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("infer_nexus_log_context", default={})
_CONFIG_LOCK = threading.RLock()
_PROCESS_IDENTITY: dict[str, Any] = {}
_PROCESS_PID: int | None = None
_INCLUDE_TRACEBACK = True
_EVENT_WRITER_STATS = {"failures": 0}
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
    "physical_service",
    "node",
    "process_role",
    "process_instance",
    "pid",
    "source",
    "exception",
    "request_id",
    "schema_version",
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
            "schema_version": 1,
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
            "source": (
                "infer_nexus"
                if getattr(record, "infer_nexus_event", False)
                else _source_for_logger(record.name)
            ),
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
                "infer_nexus_event": True,
            },
            stacklevel=stacklevel + 1,
            **logging_kwargs,
        )


def _safe_filename_component(value: Any, default: str) -> str:
    """Return a bounded filename component without exposing path separators."""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return normalized or default


def _record_event_writer_failure(
    *,
    physical_service: str,
    process_role: str,
    path: str,
    error: BaseException,
) -> None:
    """Report a file-writer failure without sending it through logging again."""
    _EVENT_WRITER_STATS["failures"] += 1
    try:
        from infer_nexus.observability.metrics import GATEWAY_METRICS

        GATEWAY_METRICS.observe_event_writer_failure(
            physical_service=physical_service,
            process_role=process_role,
        )
    except Exception:
        # Metrics must never make the fallback stderr diagnostic disappear.
        pass
    try:
        sys.stderr.write(
            "infer-nexus event writer failed "
            f"for {path!r} ({physical_service}/{process_role}): "
            f"{type(error).__name__}\n"
        )
        sys.stderr.flush()
    except Exception:
        # A broken stderr stream is outside the logging module's recovery scope.
        pass


class _ApplicationEventFilter(logging.Filter):
    """Keep arbitrary framework records out of the authoritative event files."""

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "infer_nexus_event", False))


class _ProcessRotatingEventHandler(RotatingFileHandler):
    """A synchronous, process-owned JSONL writer with fork detection."""

    def __init__(
        self,
        directory: str | Path,
        *,
        max_bytes: int,
        backup_count: int,
        error_interval_seconds: float,
    ) -> None:
        self._directory = Path(directory)
        self._owner_pid = os.getpid()
        self._error_interval_seconds = error_interval_seconds
        self._last_error_at = 0.0
        self._infer_nexus_managed = True
        self._directory.mkdir(parents=True, exist_ok=True)
        super().__init__(
            self._path_for_current_process(),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )

    def _path_for_current_process(self) -> str:
        role = _safe_filename_component(_PROCESS_IDENTITY.get("process_role"), "process")
        instance = _safe_filename_component(
            _PROCESS_IDENTITY.get("process_instance"), "unknown"
        )
        return str(self._directory / f"events-{role}-{instance}.jsonl")

    def _switch_after_fork(self) -> None:
        """Close an inherited stream and give the child a new process identity."""
        global _PROCESS_IDENTITY, _PROCESS_PID
        try:
            super().close()
        except Exception:
            pass
        identity = dict(_PROCESS_IDENTITY)
        identity["pid"] = os.getpid()
        identity["process_instance"] = uuid4().hex
        identity["node"] = os.getenv("INFER_NEXUS_NODE_NAME", socket.gethostname())
        _PROCESS_IDENTITY = identity
        _PROCESS_PID = os.getpid()
        self._owner_pid = os.getpid()
        self.baseFilename = os.path.abspath(self._path_for_current_process())
        self._closed = False
        self._last_error_at = 0.0

    def _report_failure(self, error: BaseException) -> None:
        now = time.monotonic()
        if now - self._last_error_at < self._error_interval_seconds:
            return
        self._last_error_at = now
        _record_event_writer_failure(
            physical_service=str(_PROCESS_IDENTITY.get("physical_service", "unknown")),
            process_role=str(_PROCESS_IDENTITY.get("process_role", "unknown")),
            path=self.baseFilename,
            error=error,
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if os.getpid() != self._owner_pid:
                self._switch_after_fork()
            super().emit(record)
        except Exception as error:
            self._report_failure(error)


def event_writer_stats() -> dict[str, int]:
    """Return process-local writer diagnostics for tests and operator probes."""
    return dict(_EVENT_WRITER_STATS)


def configure_logging(settings: LoggingSettings, process_identity: Mapping[str, Any]) -> None:
    """Configure one process's standard logging handlers and stable identity."""
    global _INCLUDE_TRACEBACK, _PROCESS_IDENTITY, _PROCESS_PID
    with _CONFIG_LOCK:
        pid = os.getpid()
        previous_instance = (
            _PROCESS_IDENTITY.get("process_instance") if _PROCESS_PID == pid else None
        )
        identity = dict(process_identity)
        identity.setdefault("service", "infer-nexus")
        identity.setdefault("service_version", "0.1.0")
        identity.setdefault("environment", os.getenv("INFER_NEXUS_ENVIRONMENT", "development"))
        identity.setdefault("process_role", "cli")
        identity["pid"] = pid
        identity.setdefault(
            "physical_service",
            os.getenv("INFER_NEXUS_PHYSICAL_SERVICE", "local"),
        )
        identity.setdefault(
            "node",
            os.getenv("INFER_NEXUS_NODE_NAME", socket.gethostname()),
        )
        identity["process_instance"] = (
            previous_instance
            or identity.get("process_instance")
            or uuid4().hex
        )
        _PROCESS_IDENTITY = identity
        _PROCESS_PID = pid
        _INCLUDE_TRACEBACK = settings.include_traceback

        root = logging.getLogger()
        for handler in tuple(root.handlers):
            root.removeHandler(handler)
            if getattr(handler, "_infer_nexus_managed", False):
                handler.close()
        handler = logging.StreamHandler(sys.stdout)
        handler._infer_nexus_managed = True  # type: ignore[attr-defined]
        handler.setFormatter(
            _JsonFormatter() if settings.format == "json" else _ConsoleFormatter()
        )
        root.addHandler(handler)

        event_log_dir = settings.event_log_dir or os.getenv("INFER_NEXUS_EVENT_LOG_DIR")
        if event_log_dir:
            try:
                event_handler = _ProcessRotatingEventHandler(
                    event_log_dir,
                    max_bytes=settings.event_max_bytes,
                    backup_count=settings.event_backup_count,
                    error_interval_seconds=settings.event_writer_error_interval_seconds,
                )
            except Exception as error:
                _record_event_writer_failure(
                    physical_service=str(identity.get("physical_service", "unknown")),
                    process_role=str(identity.get("process_role", "unknown")),
                    path=str(event_log_dir),
                    error=error,
                )
            else:
                event_handler.addFilter(_ApplicationEventFilter())
                event_handler.setFormatter(_JsonFormatter())
                root.addHandler(event_handler)
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
