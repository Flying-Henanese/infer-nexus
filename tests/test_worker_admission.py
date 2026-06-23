import asyncio
import os

import pytest

from infer_nexus.control.worker_admission import (
    WorkerAdmissionController,
    WorkerOverloadedError,
)


class FakeWorkerMetrics:
    def __init__(self) -> None:
        self.inflight: list[tuple[str, int]] = []
        self.capacity: list[tuple[str, int]] = []
        self.rejections: list[tuple[str, str]] = []

    def set_gateway_worker_inflight(self, *, worker: str, value: int) -> None:
        self.inflight.append((worker, value))

    def set_gateway_worker_capacity(self, *, worker: str, value: int) -> None:
        self.capacity.append((worker, value))

    def observe_gateway_worker_rejection(self, *, worker: str, reason: str) -> None:
        self.rejections.append((worker, reason))


def test_worker_admission_rejects_at_capacity_and_recovers_after_release() -> None:
    async def run_case() -> None:
        metrics = FakeWorkerMetrics()
        controller = WorkerAdmissionController(max_inflight=1, metrics=metrics)

        await controller.acquire()
        with pytest.raises(WorkerOverloadedError):
            await controller.acquire()

        assert controller.snapshot.active_requests == 1
        assert metrics.rejections == [(str(os.getpid()), "gateway_worker_overloaded")]

        await controller.release()
        await controller.acquire()
        assert controller.snapshot.active_requests == 1
        await controller.release()

    asyncio.run(run_case())


def test_worker_admission_zero_limit_is_unbounded() -> None:
    async def run_case() -> None:
        controller = WorkerAdmissionController(max_inflight=0, metrics=FakeWorkerMetrics())

        await asyncio.gather(*(controller.acquire() for _ in range(20)))
        assert controller.snapshot.active_requests == 20
        assert controller.snapshot.max_inflight == 0
        assert controller.snapshot.worker_pid == os.getpid()

        await asyncio.gather(*(controller.release() for _ in range(20)))
        assert controller.snapshot.active_requests == 0

    asyncio.run(run_case())


def test_worker_admission_exposes_retry_after_seconds() -> None:
    controller = WorkerAdmissionController(
        max_inflight=1,
        retry_after_seconds=3,
        metrics=FakeWorkerMetrics(),
    )

    assert controller.retry_after_seconds == 3


def test_worker_admission_concurrent_callers_never_exceed_limit() -> None:
    async def run_case() -> None:
        controller = WorkerAdmissionController(max_inflight=3, metrics=FakeWorkerMetrics())
        admitted = 0

        async def attempt() -> None:
            nonlocal admitted
            try:
                await controller.acquire()
            except WorkerOverloadedError:
                return
            admitted += 1

        await asyncio.gather(*(attempt() for _ in range(20)))

        assert admitted == 3
        assert controller.snapshot.active_requests == 3

    asyncio.run(run_case())
