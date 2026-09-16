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

### Admission Metric Contract

- `infer_nexus_requests_total{status="rejected"}` counts one HTTP request that
  the lifecycle classified as an explicit admission rejection.
- `infer_nexus_errors_total` records the same request's stable error code, such
  as `gateway_overloaded`.
- `infer_nexus_admission_rejections_total` is the cross-layer logical total and
  is incremented only by the outer request lifecycle, once per marked request.
- `infer_nexus_runtime_guard_rejections_total` is a layer-specific detail
  counter incremented by the per-model runtime guard before Serve handle
  routing. It can equal the generic admission total when that guard is the only
  rejection source, but the two metrics have different ownership and meaning.

## Ray Serve And vLLM Metrics

Current monitoring direction:

- Scrape infer-nexus gateway metrics through the public Serve HTTP proxy on
  `ray-head:8000/metrics`; there is no standalone gateway container in Compose
  mode.
- Scrape Ray Serve metrics from Ray's dynamic metrics targets.
- Do not make the gateway re-export Ray or vLLM metrics as the first step.
- Embedded vLLM internal metrics have not been observed through the current Ray metrics source.

## Structured Logs And Request IDs

- Application logs use readable console output in the local profile and JSONL
  in CUDA/Ascend Compose settings. Ray Core/Serve logging is configured in the
  selected format, and vLLM records propagate through the application handler.
- Request ID resolution is inbound validated `X-Request-ID`, then Ray Serve's
  request ID when available, then a generated UUID. The response always carries
  `X-Request-ID`; streaming responses also carry the legacy
  `X-Infer-Nexus-Request-ID`. The same ID follows internal Serve handle calls
  and model replica context; proxy propagation follows the model header policy.
- One Gateway request terminal event (`request.completed` or
  `request.failed`) is emitted per HTTP request. Streams also produce one
  `stream.*` terminal event. Admission rejection and model/process lifecycle
  events are emitted by the component that owns those state transitions.
- Local bootstrap logs are under `.infer-nexus/logs/`; actual Ray session logs
  are under `.infer-nexus/ray/session_latest/logs/` (or the printed external
  `/tmp/ray/session_latest/logs/` path). Compose exports separate session
  directories to `logs/<service>/ray/`; each sibling `container.log` contains
  that service command's stdout/stderr. Ray component files use 50 MiB × 3
  backups. `container.log` is append-only on the host and requires a host
  `logrotate` policy when retention is needed.
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
