"""Inspect Ray Serve hosted vLLM metrics exposed to Prometheus.

The script is intentionally dependency-free so it can run on a production host
without changing the Python environment. It can query either a Prometheus HTTP
API or a raw Prometheus text-format metrics endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable


RAY_SERVE_METRICS = (
    "ray_serve_deployment_queued_queries",
    "ray_serve_request_router_fulfillment_time_ms",
    "ray_serve_num_ongoing_requests_at_replicas",
    "ray_serve_replica_processing_queries",
    "ray_serve_deployment_processing_latency_ms",
)

VLLM_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:request_queue_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)


@dataclass(frozen=True)
class MetricCheck:
    """One metric availability check result."""

    name: str
    available: bool


def _read_url(url: str, *, timeout_seconds: float) -> str:
    request = urllib.request.Request(url, headers={"Accept": "application/json,text/plain"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to fetch {url}: {exc}") from exc


def _prometheus_query(base_url: str, query: str, *, timeout_seconds: float) -> bool:
    encoded = urllib.parse.urlencode({"query": query})
    url = f"{base_url.rstrip('/')}/api/v1/query?{encoded}"
    payload = json.loads(_read_url(url, timeout_seconds=timeout_seconds))
    if payload.get("status") != "success":
        return False
    data = payload.get("data") or {}
    result = data.get("result") or []
    return bool(result)


def _check_prometheus_api(
    base_url: str,
    metric_names: Iterable[str],
    *,
    timeout_seconds: float,
) -> list[MetricCheck]:
    return [
        MetricCheck(
            name=name,
            available=_prometheus_query(base_url, name, timeout_seconds=timeout_seconds),
        )
        for name in metric_names
    ]


def _metric_exists_in_text(metrics_text: str, metric_name: str) -> bool:
    prefixes = (
        f"# HELP {metric_name}",
        f"# TYPE {metric_name}",
        metric_name,
    )
    return any(line.startswith(prefixes) for line in metrics_text.splitlines())


def _check_metrics_text(metrics_url: str, metric_names: Iterable[str], *, timeout_seconds: float) -> list[MetricCheck]:
    body = _read_url(metrics_url, timeout_seconds=timeout_seconds)
    return [MetricCheck(name=name, available=_metric_exists_in_text(body, name)) for name in metric_names]


def _print_group(title: str, checks: list[MetricCheck]) -> None:
    print(title)
    for check in checks:
        status = "ok" if check.available else "missing"
        print(f"  [{status}] {check.name}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Ray Serve and embedded vLLM metrics for infer-nexus.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prometheus-url",
        help="Prometheus base URL, for example http://127.0.0.1:9090.",
    )
    source.add_argument(
        "--metrics-url",
        help="Raw Prometheus text metrics URL, for example a Ray metrics endpoint.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    all_metrics = (*RAY_SERVE_METRICS, *VLLM_METRICS)

    try:
        if args.prometheus_url:
            checks = _check_prometheus_api(
                args.prometheus_url,
                all_metrics,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            checks = _check_metrics_text(
                args.metrics_url,
                all_metrics,
                timeout_seconds=args.timeout_seconds,
            )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    ray_checks = checks[: len(RAY_SERVE_METRICS)]
    vllm_checks = checks[len(RAY_SERVE_METRICS) :]
    _print_group("Ray Serve metrics", ray_checks)
    _print_group("vLLM metrics", vllm_checks)

    ray_available = any(check.available for check in ray_checks)
    vllm_available = any(check.available for check in vllm_checks)
    print("Recommendation")
    if ray_available:
        print("  - Ray Serve metrics are visible; use them as the primary queue and replica signal.")
    else:
        print("  - Ray Serve metrics are not visible; verify Ray metrics scrape targets first.")
    if vllm_available:
        print("  - vLLM internal metrics are visible from this source; dashboard can include vLLM panels.")
    else:
        print("  - vLLM internal metrics are not visible; consider replica-level infer-nexus metrics after confirming the runtime shape.")

    return 0 if ray_available else 1


if __name__ == "__main__":
    raise SystemExit(main())
