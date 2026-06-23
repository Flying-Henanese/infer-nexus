import asyncio
import json
from types import SimpleNamespace

import pytest

from infer_nexus.api.worker_admission_middleware import WorkerAdmissionMiddleware
from infer_nexus.control.worker_admission import WorkerAdmissionController



class FakeWorkerMetrics:
    def set_gateway_worker_inflight(self, *, worker: str, value: int) -> None:
        return None

    def set_gateway_worker_capacity(self, *, worker: str, value: int) -> None:
        return None

    def observe_gateway_worker_rejection(self, *, worker: str, reason: str) -> None:
        return None


def _scope(path: str, controller: WorkerAdmissionController) -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "app": SimpleNamespace(state=SimpleNamespace(worker_admission=controller)),
    }


async def _receive() -> dict:
    return {"type": "http.request", "body": b"", "more_body": False}


def test_middleware_returns_retryable_503_when_worker_is_full() -> None:
    async def run_case() -> None:
        called = False

        async def app(scope, receive, send) -> None:
            nonlocal called
            called = True

        controller = WorkerAdmissionController(
            max_inflight=1,
            retry_after_seconds=2,
            metrics=FakeWorkerMetrics(),
        )
        await controller.acquire()
        messages: list[dict] = []
        middleware = WorkerAdmissionMiddleware(app)

        async def send(message: dict) -> None:
            messages.append(message)

        await middleware(_scope("/v1/chat/completions", controller), _receive, send)

        assert called is False
        assert messages[0]["status"] == 503
        headers = dict(messages[0]["headers"])
        assert headers[b"retry-after"] == b"2"
        payload = json.loads(messages[1]["body"])
        assert payload["error"]["code"] == "gateway_worker_overloaded"

    asyncio.run(run_case())


def test_middleware_bypasses_health_endpoint() -> None:
    async def run_case() -> None:
        async def app(scope, receive, send) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        controller = WorkerAdmissionController(max_inflight=1, metrics=FakeWorkerMetrics())
        await controller.acquire()
        messages: list[dict] = []

        async def send(message: dict) -> None:
            messages.append(message)

        await WorkerAdmissionMiddleware(app)(
            _scope("/healthz", controller),
            _receive,
            send,
        )

        assert messages[0]["status"] == 200
        assert controller.snapshot.active_requests == 1

    asyncio.run(run_case())


def test_middleware_holds_slot_until_stream_finishes() -> None:
    async def run_case() -> None:
        first_chunk_sent = asyncio.Event()
        finish_stream = asyncio.Event()

        async def app(scope, receive, send) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"first", "more_body": True})
            first_chunk_sent.set()
            await finish_stream.wait()
            await send({"type": "http.response.body", "body": b"last", "more_body": False})

        controller = WorkerAdmissionController(max_inflight=1, metrics=FakeWorkerMetrics())
        middleware = WorkerAdmissionMiddleware(app)

        async def send(_message: dict) -> None:
            return None

        task = asyncio.create_task(
            middleware(_scope("/v1/chat/completions", controller), _receive, send)
        )

        await first_chunk_sent.wait()
        assert controller.snapshot.active_requests == 1
        finish_stream.set()
        await task
        assert controller.snapshot.active_requests == 0

    asyncio.run(run_case())


def test_middleware_releases_slot_when_application_raises() -> None:
    async def run_case() -> None:
        async def app(scope, receive, send) -> None:
            raise RuntimeError("boom")

        controller = WorkerAdmissionController(max_inflight=1, metrics=FakeWorkerMetrics())

        async def send(_message: dict) -> None:
            return None

        with pytest.raises(RuntimeError, match="boom"):
            await WorkerAdmissionMiddleware(app)(
                _scope("/v1/embeddings", controller),
                _receive,
                send,
            )

        assert controller.snapshot.active_requests == 0

    asyncio.run(run_case())
