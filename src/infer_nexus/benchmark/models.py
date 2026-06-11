"""Benchmark data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

BenchmarkStatus = Literal["ok", "error", "timeout"]


@dataclass(slots=True)
class BenchmarkRequest:
    """One request sent by the benchmark runner."""

    request_id: str
    model: str
    messages: list[dict[str, Any]]
    stream: bool
    start_time: float
    max_tokens: int | None = None


@dataclass(slots=True)
class BenchmarkResult:
    """Client response timing returned to the runner."""

    first_token_time: float | None
    end_time: float
    output_text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    chunk_times: list[float] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkSample:
    """Raw per-request benchmark sample."""

    request_id: str
    model: str
    status: BenchmarkStatus
    start_time: float
    first_token_time: float | None
    end_time: float
    input_tokens: int
    output_tokens: int
    error_code: str | None = None
    chunk_times: list[float] = field(default_factory=list)

    @property
    def ttft_ms(self) -> float | None:
        """Return time to first token in milliseconds."""
        if self.first_token_time is None:
            return None
        return round((self.first_token_time - self.start_time) * 1000, 6)

    @property
    def e2e_latency_ms(self) -> float:
        """Return end-to-end latency in milliseconds."""
        return round((self.end_time - self.start_time) * 1000, 6)

    @property
    def tpot_ms(self) -> float | None:
        """Return per-output-token latency after first token in milliseconds."""
        if self.status != "ok" or self.first_token_time is None or self.output_tokens <= 0:
            return None
        return round((self.end_time - self.first_token_time) * 1000 / self.output_tokens, 6)

    def to_raw_dict(self) -> dict[str, Any]:
        """Serialize the raw sample with derived latency fields."""
        data = asdict(self)
        data["ttft_ms"] = self.ttft_ms
        data["tpot_ms"] = self.tpot_ms
        data["e2e_latency_ms"] = self.e2e_latency_ms
        return data


@dataclass(slots=True)
class BenchmarkSummary:
    """Aggregated benchmark result."""

    total_requests: int
    successful_requests: int
    error_rate: float
    timeout_rate: float
    duration_seconds: float
    requests_per_second: float
    output_tokens_per_second: float
    total_tokens_per_second: float
    goodput_requests_per_second: float
    ttft_ms: dict[str, float | None]
    tpot_ms: dict[str, float | None]
    e2e_latency_ms: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class BenchmarkRunResult:
    """Full benchmark run output."""

    samples: list[BenchmarkSample]
    summary: BenchmarkSummary
