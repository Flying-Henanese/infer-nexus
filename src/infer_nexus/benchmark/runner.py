"""Benchmark runner orchestration."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Protocol

from infer_nexus.benchmark.config import BenchmarkConfig, WorkloadItem
from infer_nexus.benchmark.metrics import summarize_samples
from infer_nexus.benchmark.models import (
    BenchmarkRequest,
    BenchmarkResult,
    BenchmarkRunResult,
    BenchmarkSample,
    BenchmarkStatus,
)
from infer_nexus.benchmark.tokenizer import EstimatedTokenizer


class BenchmarkClient(Protocol):
    """Protocol implemented by benchmark clients."""

    async def complete(
        self,
        request: BenchmarkRequest,
        *,
        timeout_seconds: float,
    ) -> BenchmarkResult:
        """Execute one benchmark request."""


class BenchmarkRunner:
    """Run a benchmark workload against a benchmark client."""

    def __init__(
        self,
        *,
        config: BenchmarkConfig,
        client: BenchmarkClient,
        tokenizer: EstimatedTokenizer | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.client = client
        self.tokenizer = tokenizer or EstimatedTokenizer()
        self.clock = clock

    async def run(self) -> BenchmarkRunResult:
        """Run warmup requests and measured requests."""
        await self._run_warmup()
        samples = await self._run_measured()
        return BenchmarkRunResult(
            samples=samples,
            summary=summarize_samples(
                samples,
                slo_ttft_ms=self.config.slo_ttft_ms,
                slo_e2e_ms=self.config.slo_e2e_ms,
            ),
        )

    async def _run_warmup(self) -> None:
        """Run warmup requests without recording samples."""
        if self.config.warmup_requests == 0:
            return
        await asyncio.gather(
            *[
                self._execute_request(
                    request_id=f"warmup-{index + 1:06d}",
                    workload_item=self._workload_item(index),
                )
                for index in range(self.config.warmup_requests)
            ]
        )

    async def _run_measured(self) -> list[BenchmarkSample]:
        """Run measured requests in the configured load mode."""
        if self.config.run_mode == "fixed_rate":
            return await self._run_fixed_rate()
        return await self._run_fixed_concurrency()

    async def _run_fixed_concurrency(self) -> list[BenchmarkSample]:
        """Run measured requests with a concurrency limit."""
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def guarded(index: int) -> BenchmarkSample:
            async with semaphore:
                return await self._execute_request(
                    request_id=f"bench-{index + 1:06d}",
                    workload_item=self._workload_item(index),
                )

        tasks = [guarded(index) for index in range(self.config.num_requests)]
        return list(await asyncio.gather(*tasks))

    async def _run_fixed_rate(self) -> list[BenchmarkSample]:
        """Run measured requests at a configured launch rate."""
        assert self.config.request_rate is not None
        interval = 1.0 / self.config.request_rate
        tasks: list[asyncio.Task[BenchmarkSample]] = []
        for index in range(self.config.num_requests):
            tasks.append(
                asyncio.create_task(
                    self._execute_request(
                        request_id=f"bench-{index + 1:06d}",
                        workload_item=self._workload_item(index),
                    )
                )
            )
            if index < self.config.num_requests - 1:
                await asyncio.sleep(interval)
        return list(await asyncio.gather(*tasks))

    def _workload_item(self, index: int) -> WorkloadItem:
        """Select workload item by weighted round-robin."""
        expanded: list[WorkloadItem] = []
        for item in self.config.workload:
            expanded.extend([item] * item.weight)
        return expanded[index % len(expanded)]

    async def _execute_request(
        self,
        *,
        request_id: str,
        workload_item: WorkloadItem,
    ) -> BenchmarkSample:
        """Execute one request and always return a raw sample."""
        start_time = self.clock()
        request = BenchmarkRequest(
            request_id=request_id,
            model=self.config.model,
            messages=workload_item.messages,
            stream=self.config.stream,
            start_time=start_time,
            max_tokens=workload_item.max_tokens,
        )
        input_tokens = self.tokenizer.count_messages(workload_item.messages)

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await self.client.complete(
                    request,
                    timeout_seconds=self.config.timeout_seconds,
                )
                output_tokens = result.output_tokens
                if output_tokens is None:
                    output_tokens = self.tokenizer.count_text(result.output_text)
                return BenchmarkSample(
                    request_id=request_id,
                    model=self.config.model,
                    status="ok",
                    start_time=start_time,
                    first_token_time=result.first_token_time,
                    end_time=result.end_time,
                    input_tokens=result.input_tokens or input_tokens,
                    output_tokens=output_tokens,
                    chunk_times=result.chunk_times,
                )
            except TimeoutError:
                if attempt < self.config.max_retries:
                    continue
                return self._error_sample(
                    request_id=request_id,
                    start_time=start_time,
                    input_tokens=input_tokens,
                    status="timeout",
                    error_code="timeout",
                )
            except Exception as exc:
                if attempt < self.config.max_retries:
                    continue
                return self._error_sample(
                    request_id=request_id,
                    start_time=start_time,
                    input_tokens=input_tokens,
                    status="error",
                    error_code=exc.__class__.__name__,
                )

    def _error_sample(
        self,
        *,
        request_id: str,
        start_time: float,
        input_tokens: int,
        status: BenchmarkStatus,
        error_code: str,
    ) -> BenchmarkSample:
        """Build an error sample using the runner clock."""
        return BenchmarkSample(
            request_id=request_id,
            model=self.config.model,
            status=status,
            start_time=start_time,
            first_token_time=None,
            end_time=self.clock(),
            input_tokens=input_tokens,
            output_tokens=0,
            error_code=error_code,
        )
