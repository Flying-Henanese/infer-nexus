"""Benchmark metric aggregation tests."""

from infer_nexus.benchmark.metrics import summarize_samples
from infer_nexus.benchmark.models import BenchmarkSample


def test_summarize_samples_computes_latency_throughput_and_error_rates() -> None:
    """Summary metrics should be derived from raw benchmark samples."""
    samples = [
        BenchmarkSample(
            request_id="bench-000001",
            model="qwen",
            status="ok",
            start_time=10.0,
            first_token_time=10.1,
            end_time=10.5,
            input_tokens=10,
            output_tokens=20,
        ),
        BenchmarkSample(
            request_id="bench-000002",
            model="qwen",
            status="ok",
            start_time=10.5,
            first_token_time=10.8,
            end_time=11.5,
            input_tokens=12,
            output_tokens=40,
        ),
        BenchmarkSample(
            request_id="bench-000003",
            model="qwen",
            status="timeout",
            start_time=10.7,
            first_token_time=None,
            end_time=11.4,
            input_tokens=8,
            output_tokens=0,
            error_code="timeout",
        ),
    ]

    summary = summarize_samples(samples, slo_ttft_ms=250.0, slo_e2e_ms=700.0)

    assert summary.total_requests == 3
    assert summary.successful_requests == 2
    assert summary.error_rate == 1 / 3
    assert summary.timeout_rate == 1 / 3
    assert summary.ttft_ms["p50"] == 200.0
    assert summary.e2e_latency_ms["p95"] == 975.0
    assert summary.tpot_ms["p50"] == 18.75
    assert summary.output_tokens_per_second == 40.0
    assert summary.goodput_requests_per_second == 0.666667
