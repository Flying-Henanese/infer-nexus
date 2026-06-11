"""Benchmark runner configuration models."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class WorkloadItem(BaseModel):
    """A single chat workload item."""

    messages: list[dict[str, Any]]
    weight: int = Field(default=1, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)


class BenchmarkConfig(BaseModel):
    """Configuration for one benchmark run."""

    base_url: str
    model: str
    api_key: str | None = None
    stream: bool = True
    run_mode: Literal["fixed_concurrency", "fixed_rate"] = "fixed_concurrency"
    num_requests: int = Field(default=10, ge=1)
    concurrency: int = Field(default=1, ge=1)
    request_rate: float | None = Field(default=None, gt=0)
    warmup_requests: int = Field(default=0, ge=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=0, ge=0)
    workload: list[WorkloadItem] = Field(default_factory=list)
    slo_ttft_ms: float | None = Field(default=None, gt=0)
    slo_e2e_ms: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_mode(self) -> "BenchmarkConfig":
        """Validate mode-specific fields and provide a default workload."""
        if self.run_mode == "fixed_rate" and self.request_rate is None:
            raise ValueError("request_rate is required for fixed_rate mode")
        if not self.workload:
            self.workload = [
                WorkloadItem(messages=[{"role": "user", "content": "hello"}]),
            ]
        return self
