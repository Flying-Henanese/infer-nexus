# Monitoring And Benchmark Baseline

Use this file when changing metrics, streaming behavior, benchmark workloads, or deployment validation.

## Current References

- `docs/PERFORMANCE_MONITORING_AND_BENCHMARK_PLAN.md`
- `docs/RAY_SERVE_VLLM_MONITORING.md`
- `src/infer_nexus/observability/metrics.py`
- `src/infer_nexus/benchmark/`
- `scripts/run_benchmark.py`
- `scripts/monitoring/inspect_ray_serve_vllm_metrics.py`

## Gateway Metrics

The gateway exposes Prometheus metrics from:

```text
/metrics
```

Important metric groups:

- request count and latency
- inflight requests
- error counters by stable code
- admission rejections
- gateway worker admission metrics
- stream TTFT and chunk interval approximation
- stream completion status
- token counters when response usage exists

## Ray Serve And vLLM Metrics

Current monitoring direction:

- Scrape infer-nexus gateway metrics through the public Serve HTTP proxy on
  `ray-head:8000/metrics`; there is no standalone gateway container in Compose
  mode.
- Scrape Ray Serve metrics from Ray's dynamic metrics targets.
- Do not make the gateway re-export Ray or vLLM metrics as the first step.
- Embedded vLLM internal metrics have not been observed through the current Ray metrics source.

## Benchmark Runner

The benchmark runner foundation lives under `src/infer_nexus/benchmark/` with CLI wrapper `scripts/run_benchmark.py`.

Current validation boundary:

- Unit and compile validation are available locally.
- CUDA Compose has received real A100 HTTP/SSE validation from the host through
  `127.0.0.1:8000` (mapped to `ray-head:8000`). Ascend still requires the
  equivalent real NPU runtime validation.

## Current Non-Goals

- Do not implement vLLM native metrics adapter unless explicitly requested.
- Do not move benchmark suites to a top-level `benchmarks/` directory unless the suite grows enough to justify it.
