"""Run a client-side OpenAI-compatible benchmark."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from infer_nexus.benchmark.client import OpenAICompatibleBenchmarkClient
from infer_nexus.benchmark.config import BenchmarkConfig
from infer_nexus.benchmark.report import write_benchmark_report
from infer_nexus.benchmark.runner import BenchmarkRunner
from infer_nexus.benchmark.workloads import get_builtin_workload


def parse_args() -> argparse.Namespace:
    """Parse benchmark CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--workload", default="short_chat")
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--non-stream", action="store_true")
    parser.add_argument("--slo-ttft-ms", type=float)
    parser.add_argument("--slo-e2e-ms", type=float)
    parser.add_argument("--output-dir", default="benchmark-results")
    return parser.parse_args()


async def run_from_args(args: argparse.Namespace) -> None:
    """Build benchmark components and run the benchmark."""
    run_mode = "fixed_rate" if args.request_rate is not None else "fixed_concurrency"
    config = BenchmarkConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        stream=not args.non_stream,
        run_mode=run_mode,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        request_rate=args.request_rate,
        warmup_requests=args.warmup_requests,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        workload=get_builtin_workload(args.workload),
        slo_ttft_ms=args.slo_ttft_ms,
        slo_e2e_ms=args.slo_e2e_ms,
    )
    client = OpenAICompatibleBenchmarkClient(
        base_url=config.base_url,
        api_key=config.api_key,
    )
    result = await BenchmarkRunner(config=config, client=client).run()
    paths = write_benchmark_report(result, Path(args.output_dir))
    print(f"raw samples: {paths.raw_jsonl}")
    print(f"summary json: {paths.summary_json}")
    print(f"summary markdown: {paths.summary_markdown}")


def main() -> None:
    """CLI entrypoint."""
    asyncio.run(run_from_args(parse_args()))


if __name__ == "__main__":
    main()
