"""Benchmark report writer tests."""

import json

from infer_nexus.benchmark.metrics import summarize_samples
from infer_nexus.benchmark.models import BenchmarkSample, BenchmarkRunResult
from infer_nexus.benchmark.report import write_benchmark_report


def test_write_benchmark_report_outputs_jsonl_summary_and_markdown(tmp_path) -> None:
    """Report writer should persist raw samples and machine-readable summary."""
    samples = [
        BenchmarkSample(
            request_id="bench-000001",
            model="qwen",
            status="ok",
            start_time=1.0,
            first_token_time=1.2,
            end_time=1.5,
            input_tokens=4,
            output_tokens=8,
        )
    ]
    run_result = BenchmarkRunResult(samples=samples, summary=summarize_samples(samples))

    paths = write_benchmark_report(run_result, tmp_path)

    raw_lines = paths.raw_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 1
    assert json.loads(raw_lines[0])["ttft_ms"] == 200.0

    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert summary["total_requests"] == 1
    assert summary["successful_requests"] == 1

    markdown = paths.summary_markdown.read_text(encoding="utf-8")
    assert "# infer-nexus Benchmark Summary" in markdown
    assert "| Total requests | 1 |" in markdown
