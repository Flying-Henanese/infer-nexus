"""Benchmark runner tests that do not require a live service."""

from __future__ import annotations

import asyncio

from infer_nexus.benchmark.config import BenchmarkConfig, WorkloadItem
from infer_nexus.benchmark.models import BenchmarkRequest, BenchmarkResult
from infer_nexus.benchmark.runner import BenchmarkRunner


class FakeBenchmarkClient:
    """Fake client for runner-only tests."""

    def __init__(self, *, fail_first_request: bool = False) -> None:
        self.fail_first_request = fail_first_request
        self.calls: list[str] = []
        self.max_active = 0
        self._active = 0

    async def complete(
        self,
        request: BenchmarkRequest,
        *,
        timeout_seconds: float,
    ) -> BenchmarkResult:
        self.calls.append(request.request_id)
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        try:
            await asyncio.sleep(0.01)
            if self.fail_first_request and request.request_id == "bench-000001":
                raise TimeoutError("request timed out")
            return BenchmarkResult(
                first_token_time=request.start_time + 0.01,
                end_time=request.start_time + 0.05,
                output_text="hello world",
                output_tokens=2,
                chunk_times=[request.start_time + 0.01, request.start_time + 0.03],
            )
        finally:
            self._active -= 1


def test_benchmark_runner_executes_fixed_concurrency_workload() -> None:
    """Fixed-concurrency mode should execute all measured requests via the client."""
    client = FakeBenchmarkClient()
    config = BenchmarkConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="qwen",
        num_requests=4,
        concurrency=2,
        warmup_requests=1,
        workload=[WorkloadItem(messages=[{"role": "user", "content": "hello"}])],
    )

    result = asyncio.run(BenchmarkRunner(config=config, client=client).run())

    assert len(result.samples) == 4
    assert result.summary.total_requests == 4
    assert result.summary.successful_requests == 4
    assert client.max_active == 2
    assert client.calls[0] == "warmup-000001"
    assert [sample.request_id for sample in result.samples] == [
        "bench-000001",
        "bench-000002",
        "bench-000003",
        "bench-000004",
    ]


def test_benchmark_runner_records_timeout_sample() -> None:
    """Client timeouts should become raw samples instead of aborting the run."""
    client = FakeBenchmarkClient(fail_first_request=True)
    config = BenchmarkConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="qwen",
        num_requests=2,
        concurrency=1,
        workload=[WorkloadItem(messages=[{"role": "user", "content": "hello"}])],
    )

    result = asyncio.run(BenchmarkRunner(config=config, client=client).run())

    assert [sample.status for sample in result.samples] == ["timeout", "ok"]
    assert result.samples[0].error_code == "timeout"
    assert result.summary.timeout_rate == 0.5
