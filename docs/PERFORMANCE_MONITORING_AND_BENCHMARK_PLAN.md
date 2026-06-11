# Performance Monitoring and Benchmark Plan

This document describes the planned monitoring, profiling, and benchmark work for
`infer-nexus`. The goal is to expose the metrics needed to optimize model
latency, throughput, and accelerator utilization across CUDA and Ascend
environments.

The plan intentionally separates always-on monitoring from short-window
profiling. Production monitoring should stay low overhead. Heavy profilers such
as Nsight, `torch.profiler`, `msprof`, and `torch_npu.profiler` should be used
only for focused investigations.

## 1. Goals

- Expose online serving indicators: request count, latency, inflight requests,
  errors, admission rejections, TTFT, TPOT, and token throughput.
- Preserve benchmark-grade client-side measurements for TTFT, TPOT, E2E
  latency, throughput, goodput, error rate, and timeout rate.
- Correlate application metrics with Ray Serve, vLLM, and hardware metrics.
- Support both CUDA and Huawei Ascend NPU environments without changing the
  public OpenAI-compatible API surface.
- Keep metric labels bounded and stable so Prometheus remains healthy.
- Provide a repeatable profiling workflow for CUDA and Ascend deep dives.

## 2. Metric Layers

Monitoring should be split into four layers.

### 2.1 Client Benchmark Metrics

Measured by a benchmark driver outside the service.

- request start time
- first SSE chunk/token arrival time
- per-chunk arrival time
- response end time
- input token count
- output token count
- error and timeout status
- benchmark workload metadata

Client-side measurement is the source of truth for user-visible TTFT, TPOT, and
E2E latency.

### 2.2 Gateway Application Metrics

Measured inside `infer-nexus` API and runtime dispatch paths.

- total requests
- per-model requests
- request latency
- inflight requests
- errors by stable code
- admission rejections
- streaming TTFT
- streaming TPOT
- input and output token counts when available

This layer explains how the gateway behaves before and after Ray Serve handle
dispatch.

### 2.3 Ray Serve and vLLM Runtime Metrics

Collected from Ray Serve and vLLM metrics endpoints or official metric APIs.

- Ray Serve replica state
- Ray Serve queue length and request latency
- vLLM running requests
- vLLM waiting requests
- KV cache usage
- prefix cache hit rate, when available
- prompt token throughput
- generation token throughput
- vLLM TTFT, TPOT, and E2E latency, when exposed by the installed version

Metric names may differ across Ray and vLLM versions. The implementation should
detect actual metric availability in the target runtime before relying on a
specific query.

### 2.4 Hardware Metrics

Collected by exporters or low-frequency sidecar samplers, never in the request
hot path.

CUDA:

- DCGM exporter
- `nvidia-smi` for ad hoc checks
- node exporter
- process exporter

Ascend:

- Ascend NPU exporter when available
- `npu-smi info` sidecar fallback when exporter support is limited
- node exporter
- process exporter

Hardware metrics should include memory/HBM usage, utilization, power,
temperature, health status, and process-level resource use.

## 3. Proposed Application Metrics

Use Prometheus-compatible metrics exposed from the FastAPI gateway.

Recommended metric names:

```text
infer_nexus_requests_total{model,task,endpoint,status}
infer_nexus_request_latency_seconds_bucket{model,task,endpoint}
infer_nexus_inflight_requests{model,task,endpoint}
infer_nexus_errors_total{model,task,code}
infer_nexus_admission_rejections_total{model,reason}
infer_nexus_stream_ttft_seconds_bucket{model}
infer_nexus_stream_tpot_seconds_bucket{model}
infer_nexus_input_tokens_total{model,task}
infer_nexus_output_tokens_total{model,task}
infer_nexus_request_queue_seconds_bucket{model}
```

Label rules:

- Use stable model identifiers, task names, endpoint names, and error codes.
- Do not use request IDs as metric labels.
- Do not use raw exception messages as labels.
- Do not use prompt text, user content, upstream URLs, or dynamic payload fields
  as labels.
- Keep task labels aligned with existing task types: `chat`, `embedding`, and
  `rerank`.

## 4. Instrumentation Points

Initial instrumentation should focus on these paths:

- `/v1/chat/completions`
- `/v1/embeddings`
- `/v1/rerank`
- `RuntimeExecutor.execute_chat`
- `RuntimeExecutor._execute_chat_stream`
- `RuntimeExecutor.execute_embedding`
- `RuntimeExecutor.execute_rerank`
- admission rejection handling
- runtime exception handling

Streaming chat requires special handling:

```text
request_start
-> backend or Serve handle dispatch
-> first non-empty SSE chunk
-> each subsequent chunk
-> final [DONE] or stream termination
```

TTFT should be measured as:

```text
first_chunk_time - request_start_time
```

TPOT should be measured from token or chunk timing. If exact output token counts
are unavailable in the server stream, the server can emit chunk-level timing and
the benchmark client should compute token-level TPOT using the model tokenizer.

## 5. Benchmark Runner Plan

Add a standalone benchmark runner instead of relying only on server-side
metrics.

Required features:

- OpenAI-compatible chat completions
- streaming and non-streaming modes
- fixed concurrency mode
- fixed request-rate mode
- configurable warmup and measurement windows
- timeout and retry controls
- JSONL raw sample output
- summary JSON or Markdown report
- tokenizer-based input and output token counting

Metrics to compute:

- TTFT P50/P90/P95/P99
- TPOT P50/P90/P95/P99
- E2E latency P50/P90/P95/P99
- requests per second
- output tokens per second
- total tokens per second
- error rate
- timeout rate
- goodput under SLO constraints
- cost per 1M input/output tokens, when pricing is provided

Suggested raw JSONL fields:

```json
{
  "request_id": "bench-000001",
  "model": "model-name",
  "status": "ok",
  "start_time": 0.0,
  "first_token_time": 0.0,
  "end_time": 0.0,
  "ttft_ms": 0.0,
  "tpot_ms": 0.0,
  "e2e_latency_ms": 0.0,
  "input_tokens": 0,
  "output_tokens": 0,
  "error_code": null
}
```

## 6. CUDA Monitoring Path

Prepared dependencies:

- Python extra: `cuda-monitoring`
- system check script: `scripts/check_cuda_perf_system_deps.sh`
- system install helper: `scripts/install_cuda_monitoring_system_deps.sh`

CUDA monitoring stack:

- DCGM exporter for GPU telemetry
- Prometheus for metrics storage
- Grafana for dashboards
- node exporter for host metrics
- process exporter for gateway, Ray, and vLLM process metrics
- `nvidia-smi` for manual checks and lightweight diagnostics

CUDA profiling tools:

- Nsight Systems (`nsys`) for CPU/GPU timeline
- Nsight Compute (`ncu`) for kernel-level analysis
- `torch.profiler` for PyTorch operator and CUDA kernel traces
- NVTX ranges for timeline annotation
- `py-spy` for Python CPU flamegraphs

Use profiling only in short windows. Compare benchmark results with profiler
disabled and enabled to estimate profiler overhead.

## 7. Ascend Monitoring Path

Prepared dependencies:

- Python check script: `scripts/check_perf_python_deps.py`
- system check script: `scripts/check_ascend_perf_system_deps.sh`

Ascend monitoring stack:

- Ascend NPU exporter when available
- `npu-smi info` sidecar sampler as a fallback
- Prometheus
- Grafana
- node exporter
- process exporter

Ascend profiling tools:

- `msprof` and `msopprof`
- `torch_npu.profiler`
- TensorBoard trace viewer
- `py-spy` for Python CPU flamegraphs

Ascend profiling must be validated in two shapes:

- direct single-process vLLM/vLLM-Ascend smoke profiling
- Ray Serve replica profiling, where the actual model work runs inside Ray
  worker processes

## 8. Grafana Dashboard Plan

Create dashboards in four sections.

Service overview:

- RPS by model
- error rate by model and code
- latency P50/P95/P99
- inflight requests
- admission rejections

LLM user experience:

- TTFT P50/P95/P99
- TPOT P50/P95/P99
- E2E latency P50/P95/P99
- output tokens per second
- goodput under configured SLO

Runtime bottlenecks:

- Ray Serve queue length
- replica count and health
- vLLM running requests
- vLLM waiting requests
- KV cache usage
- prefix cache hit rate, when available

Hardware utilization:

- CUDA GPU utilization or Ascend AI Core utilization
- GPU memory or Ascend HBM usage
- power draw
- temperature
- process CPU and RSS

## 9. Implementation Phases

### Phase 1: Metric Contract

- Finalize metric names and labels.
- Document histogram buckets for latency, TTFT, TPOT, and queue time.
- Define stable error codes used by metrics.
- Decide how to map model aliases to metric labels.

### Phase 2: Gateway Metrics

- Add `prometheus-client` integration.
- Add `/metrics` endpoint.
- Add request counters, latency histograms, inflight gauges, error counters, and
  admission rejection counters.
- Validate metrics under stub mode and serve mode.

### Phase 3: Streaming Metrics

- Instrument streaming response iterators.
- Measure first chunk time.
- Measure chunk timing and stream completion.
- Export TTFT and chunk-level TPOT approximations.
- Validate that passthrough SSE and mapped SSE both emit metrics.

### Phase 4: Benchmark Runner

- Add benchmark workload configuration.
- Support fixed concurrency and fixed request rate.
- Support streaming token timing.
- Output raw JSONL and summary reports.
- Add repeatable smoke workloads for short chat, RAG-style prompt, and long
  context prompt.

### Phase 5: Hardware Exporters

- Deploy DCGM exporter on CUDA.
- Deploy Ascend NPU exporter or `npu-smi` sampler on Ascend.
- Deploy node exporter and process exporter.
- Add Prometheus scrape examples.

### Phase 6: Ray and vLLM Metrics

- Confirm Ray Serve Prometheus endpoint and service discovery.
- Confirm installed vLLM metrics names in CUDA and Ascend environments.
- Add dashboard queries for queue, running/waiting requests, KV cache, and token
  throughput.
- Avoid private vLLM scheduler/cache introspection unless official metrics are
  unavailable.

### Phase 7: Profiling Workflow

- Add CUDA profiling runbooks for `nsys`, `ncu`, `torch.profiler`, and NVTX.
- Add Ascend profiling runbooks for `msprof`, `msopprof`, and
  `torch_npu.profiler`.
- Define profiling windows, warmup, and post-processing outputs.
- Record profiler overhead compared with profiler-disabled benchmark runs.

## 10. Validation Checklist

- `/metrics` endpoint returns Prometheus text format.
- A chat request increments request counters.
- A failed request increments error counters with a stable code.
- A streaming request records TTFT.
- A benchmark run produces raw JSONL and summary output.
- Prometheus scrapes application metrics.
- Prometheus scrapes hardware exporter metrics.
- Grafana shows service, runtime, and hardware panels.
- CUDA profiler can capture a short inference run.
- Ascend profiler can capture a short inference run.
- Metrics remain stable under repeated benchmark runs.

## 11. Open Decisions

- Exact histogram buckets for TTFT, TPOT, E2E latency, and queue time.
- Whether server-side TPOT should remain chunk-based when exact token counts are
  unavailable.
- Whether the benchmark runner should live under `scripts/` or a new
  `benchmarks/` directory.
- Whether dashboard JSON should be checked into the repository.
- How to normalize model names when clients use aliases but runtime uses served
  model names.
