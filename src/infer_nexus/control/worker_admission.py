"""Process-local admission control for one HTTP gateway worker."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from typing import Protocol

from infer_nexus.observability.metrics import GATEWAY_METRICS


class WorkerMetrics(Protocol):
    """Metrics operations used by worker admission."""

    def set_gateway_worker_inflight(self, *, worker: str, value: int) -> None: ...

    def set_gateway_worker_capacity(self, *, worker: str, value: int) -> None: ...

    def observe_gateway_worker_rejection(self, *, worker: str, reason: str) -> None: ...


class WorkerOverloadedError(RuntimeError):
    """Raised when a gateway worker has no immediately available capacity."""


@dataclass(frozen=True, slots=True)
class WorkerAdmissionSnapshot:
    """Immutable process-local worker admission state."""

    worker_pid: int
    max_inflight: int
    active_requests: int


class WorkerAdmissionController:
    """Apply a fail-fast active-request limit inside one gateway process."""

    def __init__(
        self,
        *,
        max_inflight: int,
        retry_after_seconds: int = 1,
        metrics: WorkerMetrics = GATEWAY_METRICS,
    ) -> None:
        if max_inflight < 0:
            raise ValueError("max_inflight must be greater than or equal to zero")
        if retry_after_seconds < 1:
            raise ValueError("retry_after_seconds must be greater than or equal to one")
        self.max_inflight = max_inflight
        self.retry_after_seconds = retry_after_seconds
        self.worker_pid = os.getpid()
        self._active_requests = 0
        self._lock = asyncio.Lock()
        self._metrics = metrics
        self._metrics.set_gateway_worker_capacity(
            worker=str(self.worker_pid),
            value=self.max_inflight,
        )
        self._sync_inflight_metric()

    @property
    def snapshot(self) -> WorkerAdmissionSnapshot:
        """Return the current process-local state."""
        return WorkerAdmissionSnapshot(
            worker_pid=self.worker_pid,
            max_inflight=self.max_inflight,
            active_requests=self._active_requests,
        )

    def _sync_inflight_metric(self) -> None:
        self._metrics.set_gateway_worker_inflight(
            worker=str(self.worker_pid),
            value=self._active_requests,
        )

    async def acquire(self) -> None:
        """Acquire capacity immediately or reject without adding a local queue."""
        async with self._lock:
            if self.max_inflight > 0 and self._active_requests >= self.max_inflight:
                self._metrics.observe_gateway_worker_rejection(
                    worker=str(self.worker_pid),
                    reason="gateway_worker_overloaded",
                )
                raise WorkerOverloadedError("Gateway worker is overloaded.")
            self._active_requests += 1
            self._sync_inflight_metric()

    async def release(self) -> None:
        """Release one active-request slot."""
        async with self._lock:
            if self._active_requests > 0:
                self._active_requests -= 1
            self._sync_inflight_metric()
