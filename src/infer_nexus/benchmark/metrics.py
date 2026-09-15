"""Benchmark metric aggregation helpers."""

from infer_nexus.benchmark.models import BenchmarkSample, BenchmarkSummary


def _percentile(values: list[float], percentile: float) -> float | None:
    """Compute a linear-interpolated percentile."""
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(sorted_values[0], 6)

    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    value = sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    return round(value, 6)


def _percentiles(values: list[float]) -> dict[str, float | None]:
    """Return standard latency percentiles."""
    return {
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _sample_window(samples: list[BenchmarkSample]) -> float:
    """Return wall-clock benchmark duration in seconds."""
    if not samples:
        return 0.0
    start = min(sample.start_time for sample in samples)
    end = max(sample.end_time for sample in samples)
    return max(end - start, 0.0)


def summarize_samples(
    samples: list[BenchmarkSample],
    *,
    slo_ttft_ms: float | None = None,
    slo_e2e_ms: float | None = None,
) -> BenchmarkSummary:
    """Aggregate raw benchmark samples."""
    total = len(samples)
    successful = [sample for sample in samples if sample.status == "ok"]
    timed_out = [sample for sample in samples if sample.status == "timeout"]
    duration = _sample_window(samples)

    ttft_values = [sample.ttft_ms for sample in successful if sample.ttft_ms is not None]
    tpot_values = [sample.tpot_ms for sample in successful if sample.tpot_ms is not None]
    e2e_values = [sample.e2e_latency_ms for sample in successful]

    output_tokens = sum(sample.output_tokens for sample in successful)
    input_tokens = sum(sample.input_tokens for sample in successful)
    goodput = [
        sample
        for sample in successful
        if (slo_ttft_ms is None or (sample.ttft_ms is not None and sample.ttft_ms <= slo_ttft_ms))
        and (slo_e2e_ms is None or sample.e2e_latency_ms <= slo_e2e_ms)
    ]

    total_token_count = input_tokens + output_tokens

    return BenchmarkSummary(
        total_requests=total,
        successful_requests=len(successful),
        error_rate=0.0 if total == 0 else (total - len(successful)) / total,
        timeout_rate=0.0 if total == 0 else len(timed_out) / total,
        duration_seconds=round(duration, 6),
        requests_per_second=0.0 if duration == 0 else round(total / duration, 6),
        output_tokens_per_second=0.0 if duration == 0 else round(output_tokens / duration, 6),
        total_tokens_per_second=0.0 if duration == 0 else round(total_token_count / duration, 6),
        goodput_requests_per_second=0.0 if duration == 0 else round(len(goodput) / duration, 6),
        ttft_ms=_percentiles(ttft_values),
        tpot_ms=_percentiles(tpot_values),
        e2e_latency_ms=_percentiles(e2e_values),
    )
