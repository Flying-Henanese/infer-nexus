# vLLM Native Metrics Exposure Design

## 1. Background

`infer-nexus` currently runs local vLLM models inside Ray Serve replicas:

```text
infer-nexus gateway
  -> Ray Serve deployment handle
  -> Ray Serve replica
  -> VLLMBackend
  -> AsyncLLMEngine / LLM
```

Ray Serve metrics can describe the outer serving layer, such as deployment
health, request count, errors, and latency. They do not automatically expose the
internal state of the vLLM engine created inside a replica.

The missing production signals are vLLM engine-level metrics:

- scheduler queue depth
- running and waiting requests
- KV cache usage
- prefix cache hit rate
- preemptions caused by KV cache pressure
- queue time and engine-side latency

This document designs how to expose those vLLM-native metrics through the
existing Ray metrics scrape path.

## 2. Goals

- Expose vLLM internal scheduler, KV cache, and prefix cache metrics for each
  local vLLM model.
- Keep Prometheus scraping through Ray dashboard service discovery:

```yaml
scrape_configs:
  - job_name: ray
    http_sd_configs:
      - url: http://<ray-head-host>:8265/api/prometheus/sd
```

- Preserve vLLM metric semantics and names where possible.
- Support multiple models and multiple Ray Serve replicas.
- Avoid depending on private vLLM scheduler/cache objects unless no official
  metrics path is available.
- Fail open: metrics collection failures must not break inference requests.

## 3. Non-Goals

- Do not replace Ray Serve runtime metrics.
- Do not start a separate HTTP metrics server per replica unless Ray metrics
  integration cannot be made to work.
- Do not infer vLLM internal queue/cache state from gateway latency alone.
- Do not initially build a full Grafana dashboard; provide query foundations
  first.

## 4. Design Principle

The vLLM engine and its metrics source live in the Ray Serve replica process.
Therefore vLLM-native metrics must be enabled or collected in that same process.

The preferred path is:

```text
vLLM engine internal stats
  -> vLLM official metrics/stat logger API
  -> Prometheus registry in the Ray worker process
  -> Ray metrics endpoint
  -> Ray dashboard HTTP service discovery
  -> Prometheus
```

The design should prioritize official vLLM metrics mechanisms:

1. vLLM Prometheus/stat logger integration.
2. Public vLLM metrics APIs such as `LLM.get_metrics()` where available.
3. A version-gated adapter around semi-public APIs.
4. Private scheduler/cache object inspection only as a last resort.

Reference: vLLM metrics documentation:
https://docs.vllm.ai/en/stable/design/metrics.html

## 5. Target Metrics

The implementation should prefer native vLLM metric names. Depending on the
installed vLLM version, exact names may differ. The adapter must detect and
document the supported set at startup.

### Scheduler

```text
vllm:num_requests_running
vllm:num_requests_waiting
vllm:num_requests_swapped
vllm:time_in_queue_requests
```

### KV Cache

```text
vllm:gpu_cache_usage_perc
vllm:cpu_cache_usage_perc
vllm:kv_cache_usage_perc
vllm:num_preemptions_total
```

### Prefix Cache

```text
vllm:gpu_prefix_cache_hit_rate
vllm:cpu_prefix_cache_hit_rate
vllm:prefix_cache_queries
vllm:prefix_cache_hits
```

### Throughput And Latency Context

```text
vllm:prompt_tokens_total
vllm:generation_tokens_total
vllm:time_to_first_token_seconds
vllm:time_per_output_token_seconds
```

## 6. Required Labels

Metrics must be distinguishable by model and replica. At minimum:

```text
model_name
serve_application
serve_deployment
replica_id
engine_id
```

If vLLM emits only `model_name` or `engine` labels, the adapter should add
stable infer-nexus/Ray Serve context where supported by the metrics API. If
label injection is not supported for a metric family, the limitation must be
recorded in startup logs and documentation.

## 7. Component Design

### 7.1 `VLLMNativeMetricsAdapter`

Add a backend-local adapter responsible for vLLM metrics integration.

Suggested module:

```text
src/infer_nexus/observability/vllm_metrics.py
```

Responsibilities:

- Inspect the installed vLLM version.
- Detect available metrics/stat logger APIs.
- Configure vLLM metrics when constructing `AsyncLLMEngine` or `LLM`.
- Attach model and replica labels when supported.
- Expose an availability diagnostic.
- Keep all vLLM-version branching in one place.

Suggested public shape:

```python
class VLLMNativeMetricsAdapter:
    def build_engine_metrics_kwargs(
        self,
        *,
        model_name: str,
        serve_application: str,
        serve_deployment: str,
        replica_id: str,
    ) -> dict[str, object]:
        ...

    def mark_available(self) -> None:
        ...

    def mark_unavailable(self, reason: str) -> None:
        ...
```

The exact API should follow the installed vLLM version's constructor and
observability options.

### 7.2 Engine Construction Integration

`VLLMBackend.startup()` currently constructs either:

- `AsyncLLMEngine` for chat/generate mode.
- `LLM` for fallback, embedding, and scoring paths.

The integration point should be before engine construction:

```text
runtime_spec + Ray Serve replica context
  -> VLLMNativeMetricsAdapter
  -> vLLM metrics kwargs / logger configuration
  -> AsyncLLMEngine.from_engine_args(...) or LLM(...)
```

The adapter should not change request execution behavior.

### 7.3 Diagnostic Metrics

Because vLLM metrics APIs can vary by version, expose infer-nexus diagnostics
through Ray metrics:

```text
infer_nexus_vllm_native_metrics_available{model_name, reason} 0|1
infer_nexus_vllm_native_metrics_init_errors_total{model_name, error_type}
```

These diagnostics answer the difference between:

- the model is idle
- native metrics are unavailable
- native metrics failed to initialize

### 7.4 Startup Validation

At replica startup, log:

```text
vLLM version
engine kind: async | llm | stub
metrics integration mode
native metrics available: true | false
missing expected metric families, if known
```

Startup must not fail solely because metrics cannot be enabled unless an
explicit strict observability option is later added.

## 8. Configuration

Add runtime configuration flags after the first implementation proves the exact
vLLM API shape:

```yaml
runtime:
  vllm_metrics:
    enabled: true
    strict: false
    prefer_native_names: true
```

Initial defaults:

- `enabled: true` for real local vLLM runtime.
- `strict: false` so inference still starts when metrics integration is
  unsupported.
- `prefer_native_names: true` to keep compatibility with existing vLLM
  dashboards and queries.

## 9. Prometheus Queries

### Scheduler Queue

```promql
vllm:num_requests_waiting
```

```promql
vllm:num_requests_running
```

```promql
histogram_quantile(
  0.95,
  rate(vllm:time_in_queue_requests_bucket[5m])
)
```

### KV Cache Pressure

```promql
vllm:gpu_cache_usage_perc
```

```promql
rate(vllm:num_preemptions_total[5m])
```

### Prefix Cache Effectiveness

```promql
vllm:gpu_prefix_cache_hit_rate
```

If the installed vLLM version exposes hit/query counters instead of a hit-rate
gauge:

```promql
rate(vllm:prefix_cache_hits[5m])
/
rate(vllm:prefix_cache_queries[5m])
```

### Token Throughput

```promql
rate(vllm:prompt_tokens_total[1m])
```

```promql
rate(vllm:generation_tokens_total[1m])
```

### Metrics Availability

```promql
infer_nexus_vllm_native_metrics_available
```

## 10. Alerting Guidance

Suggested alert conditions:

- `vllm:num_requests_waiting > 0` for a sustained window while traffic is
  present.
- `vllm:gpu_cache_usage_perc > 0.90` for a sustained window.
- `rate(vllm:num_preemptions_total[5m]) > 0`.
- Prefix cache hit rate unexpectedly drops for workloads expected to share
  prefixes.
- `infer_nexus_vllm_native_metrics_available == 0` for any real vLLM model.

These alerts should be combined with Ray Serve request/error metrics so the
operator can distinguish engine pressure from ingress or routing failures.

## 11. Implementation Plan

### Phase 1: Version And API Discovery

- Print and record the installed vLLM version in runtime logs.
- Identify supported metrics/stat logger constructor options for the installed
  version.
- Confirm whether metrics can be registered in the Ray worker process without a
  separate vLLM HTTP server.

Acceptance:

- A local diagnostic script can report vLLM version and available metrics
  integration mode.

### Phase 2: Native Metrics Adapter

- Add `VLLMNativeMetricsAdapter`.
- Integrate it into `VLLMBackend` engine construction.
- Add availability/error diagnostic metrics.
- Keep metrics failures non-fatal.

Acceptance:

```bash
curl http://<ray-metrics-target>/metrics | grep 'vllm:num_requests_waiting'
curl http://<ray-metrics-target>/metrics | grep 'vllm:gpu_cache_usage_perc'
curl http://<ray-metrics-target>/metrics | grep -i 'prefix.*cache'
```

At least scheduler and KV cache metrics should appear for an active real vLLM
model.

### Phase 3: Label And Multi-Replica Validation

- Validate multiple local models.
- Validate multiple replicas for the same model.
- Confirm metric label cardinality and uniqueness.
- Add documentation for labels that cannot be injected due to vLLM API limits.

Acceptance:

- Prometheus can group metrics by `model_name`.
- Replica-level metrics can be distinguished when multiple replicas exist.

### Phase 4: Monitoring Pack

- Add Prometheus query examples to deployment docs.
- Add a minimal Grafana dashboard JSON or dashboard notes.
- Add alert recommendations for queue, KV pressure, preemption, and metrics
  availability.

## 12. Risks And Mitigations

### vLLM API Drift

Risk: vLLM metrics/stat logger APIs change across versions.

Mitigation:

- Keep all version branching in `VLLMNativeMetricsAdapter`.
- Log the selected integration mode.
- Prefer official APIs and native metric names.

### Duplicate Metric Registration

Risk: Multiple engines in one process may try to register the same metric
families.

Mitigation:

- Use vLLM's official multi-engine labeling if available.
- Add a deterministic `engine_id`.
- Validate multi-model deployments early.

### Label Cardinality

Risk: Per-request or unstable labels can overload Prometheus.

Mitigation:

- Allow only stable labels: model, application, deployment, replica, engine.
- Do not include request id, prompt hash, user id, or arbitrary route labels.

### Metrics Impacting Inference

Risk: Metrics collection failure or overhead affects request handling.

Mitigation:

- Initialize metrics once at startup.
- Avoid high-frequency polling of private engine objects.
- Treat metrics failures as diagnostics, not request failures.

## 13. Open Questions

- Which exact vLLM version will be the supported baseline for this branch?
- Does the supported version expose native metrics for embedded
  `AsyncLLMEngine`, or only for the OpenAI API server path?
- Can native vLLM metrics add Ray Serve replica labels without duplicating
  metric families?
- Are prefix cache counters enabled by default, or do they require additional
  vLLM flags such as prefix caching or KV cache metrics sampling?

## 14. Expected Operator Workflow

1. Start Ray and infer-nexus normally.
2. Confirm Ray metrics service discovery:

```bash
curl http://<ray-head-host>:8265/api/prometheus/sd
```

3. Pick one returned target and verify vLLM metrics:

```bash
curl http://<target-host>:<target-port>/metrics | grep 'vllm:'
```

4. Use Prometheus queries for scheduler queue, KV cache pressure, prefix cache
   effectiveness, and preemption rate.

If `vllm:` metrics are absent, check:

```promql
infer_nexus_vllm_native_metrics_available
```

Then inspect replica startup logs for the selected integration mode and any
vLLM metrics initialization error.
