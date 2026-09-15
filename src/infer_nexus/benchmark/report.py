"""Benchmark report writers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from infer_nexus.benchmark.models import BenchmarkRunResult


@dataclass(frozen=True, slots=True)
class BenchmarkReportPaths:
    """Paths written by the benchmark report writer."""

    raw_jsonl: Path
    summary_json: Path
    summary_markdown: Path


def write_benchmark_report(
    run_result: BenchmarkRunResult,
    output_dir: str | Path,
) -> BenchmarkReportPaths:
    """Write raw samples and summary reports."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    raw_jsonl = directory / "benchmark_samples.jsonl"
    summary_json = directory / "benchmark_summary.json"
    summary_markdown = directory / "benchmark_summary.md"

    with raw_jsonl.open("w", encoding="utf-8") as handle:
        for sample in run_result.samples:
            handle.write(json.dumps(sample.to_raw_dict(), ensure_ascii=False) + "\n")

    summary = run_result.summary.to_dict()
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_markdown.write_text(_format_summary_markdown(summary), encoding="utf-8")
    return BenchmarkReportPaths(
        raw_jsonl=raw_jsonl,
        summary_json=summary_json,
        summary_markdown=summary_markdown,
    )


def _format_summary_markdown(summary: dict[str, object]) -> str:
    """Format a concise Markdown summary."""
    rows = [
        ("Total requests", summary["total_requests"]),
        ("Successful requests", summary["successful_requests"]),
        ("Error rate", summary["error_rate"]),
        ("Timeout rate", summary["timeout_rate"]),
        ("Duration seconds", summary["duration_seconds"]),
        ("Requests per second", summary["requests_per_second"]),
        ("Output tokens per second", summary["output_tokens_per_second"]),
        ("Goodput requests per second", summary["goodput_requests_per_second"]),
    ]
    lines = [
        "# infer-nexus Benchmark Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    lines.extend(
        [
            "",
            "## Latency Percentiles",
            "",
            "```json",
            json.dumps(
                {
                    "ttft_ms": summary["ttft_ms"],
                    "tpot_ms": summary["tpot_ms"],
                    "e2e_latency_ms": summary["e2e_latency_ms"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
