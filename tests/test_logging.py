"""Behavioral tests for the application logging interface."""

import asyncio
import json
import logging
from types import SimpleNamespace

import cloudpickle
import pytest

from infer_nexus.api.request_logging_middleware import RequestLoggingMiddleware
from infer_nexus.core.config import LoggingSettings
from infer_nexus.observability.logging import bind_log_context, configure_logging, get_logger


@pytest.fixture
def isolated_root_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    root = logging.getLogger()
    previous_handlers = root.handlers
    previous_level = root.level
    monkeypatch.setattr(root, "handlers", list(previous_handlers))
    monkeypatch.setattr(root, "level", previous_level)


def _configure(format: str = "json") -> None:
    configure_logging(
        LoggingSettings(format=format),
        {
            "service": "infer-nexus",
            "service_version": "test-version",
            "environment": "test",
            "process_role": "gateway",
            "pid": 123,
        },
    )


def test_json_logs_have_contract_fields_and_redact_sensitive_values(
    isolated_root_logger: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _configure()

    with bind_log_context(request_id="req-123", model="qwen3.5-9b"):
        get_logger("infer_nexus.test").info(
            "request.completed",
            request_id="wrong-id",
            outcome="success",
            status_code=200,
            authorization="Bearer secret",
            details={"prompt": "private input", "replica": "replica-1"},
        )

    line = capsys.readouterr().out.strip()
    record = json.loads(line)
    assert record["event"] == "request.completed"
    assert record["source"] == "infer_nexus"
    assert record["request_id"] == "req-123"
    assert record["model"] == "qwen3.5-9b"
    assert record["outcome"] == "success"
    assert record["authorization"] == "[REDACTED]"
    assert record["details"] == {"prompt": "[REDACTED]", "replica": "replica-1"}


def test_log_context_is_isolated_between_async_tasks(
    isolated_root_logger: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _configure()
    logger = get_logger("infer_nexus.test")

    async def emit(request_id: str) -> None:
        with bind_log_context(request_id=request_id):
            await asyncio.sleep(0)
            logger.info("request.completed", outcome="success")

    async def run_workers() -> None:
        await asyncio.gather(emit("req-a"), emit("req-b"))

    asyncio.run(run_workers())

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {record["request_id"] for record in records} == {"req-a", "req-b"}


def test_exception_is_serialized_as_one_json_line_without_exception_text(
    isolated_root_logger: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _configure()
    try:
        raise RuntimeError("backend failed")
    except RuntimeError:
        get_logger("infer_nexus.test").exception(
            "request.failed", error_code="backend_failure"
        )

    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    record = json.loads(output)
    assert record["exception"]["type"] == "RuntimeError"
    assert "test_exception_is_serialized_as_one_json_line_without_exception_text" in record[
        "exception"
    ]["stack_trace"]
    assert "backend failed" not in output


def test_console_profile_is_readable(
    isolated_root_logger: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _configure("console")

    get_logger("infer_nexus.test").info("model.replica.ready", model="qwen3.5-9b")

    output = capsys.readouterr().out
    assert "INFO" in output
    assert "model.replica.ready" in output
    assert "qwen3.5-9b" in output


def test_application_logger_uses_configured_global_level_as_fallback(
    isolated_root_logger: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = logging.getLogger("infer_nexus.level_fallback")
    monkeypatch.setattr(logger, "level", logging.NOTSET)

    configure_logging(
        LoggingSettings(level="WARNING"),
        {"process_role": "gateway"},
    )

    assert logger.getEffectiveLevel() == logging.WARNING


def test_standard_library_records_keep_their_message(
    isolated_root_logger: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Framework records without the application adapter retain their message."""
    _configure()

    logging.getLogger("ray.serve").warning("ordinary Serve warning")

    record = json.loads(capsys.readouterr().out)
    assert record["event"] == "log.record"
    assert record["message"] == "ordinary Serve warning"
    assert record["source"] == "ray_serve"


def test_request_logging_middleware_can_be_cloudpickled() -> None:
    """Ray must deserialize the wrapper before the wrapped ASGI app is restored."""
    middleware = RequestLoggingMiddleware(object())

    restored = cloudpickle.loads(cloudpickle.dumps(middleware))

    assert isinstance(restored, RequestLoggingMiddleware)


def test_ray_serve_proxy_owns_the_response_request_id_header(monkeypatch) -> None:
    """Serve adds the canonical ID outside ASGI; the Gateway adds only its legacy stream header."""
    request_id = "req-from-ray-proxy"

    class StreamingASGIApp:
        state = SimpleNamespace(
            settings=SimpleNamespace(
                observability=SimpleNamespace(logging=LoggingSettings())
            )
        )

        async def __call__(self, scope, receive, send) -> None:
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"text/event-stream"),
                        (b"x-request-id", b"inner-request-id"),
                    ],
                }
            )
            await send(
                {"type": "http.response.body", "body": b"data: done\n\n"}
            )

    monkeypatch.setattr(
        "infer_nexus.api.request_logging_middleware._serve_request_id",
        lambda: request_id,
    )
    messages = []
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
        "state": {},
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message) -> None:
        messages.append(message)

    asyncio.run(RequestLoggingMiddleware(StreamingASGIApp())(scope, receive, send))

    response_headers = dict(messages[0]["headers"])
    assert b"x-request-id" not in response_headers
    assert response_headers[b"x-infer-nexus-request-id"] == request_id.encode()
    assert scope["state"]["request_lifecycle"].context.request_id == request_id
