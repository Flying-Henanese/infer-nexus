# Ray Serve Gateway Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve `infer-nexus` gateway resilience and latency when Ray Serve router/control-plane behavior slows down, especially queue-length probe storms triggered by `DeploymentHandle` routing.

**Architecture:** Keep the FastAPI gateway as the northbound OpenAI-compatible API, but reduce how much Ray Serve router/control-plane work can block the gateway request path. The immediate path is configuration hardening and better backpressure; the medium-term path is per-model runtime isolation and gateway-owned routing decisions based on local metric snapshots rather than synchronous Ray actor probes.

**Tech Stack:** FastAPI, Ray Serve `DeploymentHandle`, vLLM, Prometheus metrics, pytest, `uv run pytest`.

---

## 1. Problem Statement

Current request path:

```text
client
-> infer-nexus FastAPI gateway
-> RuntimeDispatcher
-> RuntimeExecutor
-> Ray Serve DeploymentHandle
-> Ray Serve router
-> ModelRuntimeReplica
-> VLLMBackend
-> vLLM
```

The observed bottleneck is not always vLLM inference. In recent remote runs, the gateway process became busy inside Ray Serve handle/router logic:

- `Failed to get queue length from Replica ... within 0.1s`
- `_fulfill_pending_requests`
- `_choose_replicas_with_backoff`
- `get_queue_len`
- `get_num_ongoing_requests.remote()`
- Ray actor RPC serialization/deserialization

This means the gateway can stall while Ray Serve is trying to select a replica or probe replica queue length. GPU utilization may stay low while client latency and timeout rate rise.

Important conclusion:

- Do not treat `PrefixCacheAffinityRouter` as production-safe on the current custom `handle.chat_completion.remote(request_payload=dict)` path.
- Do not put synchronous or high-frequency Ray actor queue probes in the gateway hot path.
- Prefer explicit gateway backpressure, per-model isolation, and locally cached routing state.

This plan supersedes the optimistic recommendation in `docs/PREFIX_CACHE_STICKY_DESIGN.md` that Ray Serve `PrefixCacheAffinityRouter` should be the primary routing direction for local vLLM chat models.

## 2. Current Code Touch Points

### Existing Gateway Request Path

- `src/infer_nexus/api/openai_routes.py`
  - HTTP route handlers.
  - Resolves model, checks model readiness/admission, calls `RuntimeDispatcher`.

- `src/infer_nexus/runtime/dispatcher.py`
  - Converts `ModelConfig` into `RuntimeTarget`.
  - Calls `RuntimeExecutor`.

- `src/infer_nexus/runtime/executor.py`
  - Core hot path for proxy, local stub, and Ray Serve execution.
  - `_invoke_handle()` calls `handle.<method>.remote(...)` for non-streaming Serve requests.
  - `_invoke_handle_stream()` calls streaming Serve handle methods.
  - Already has timeout, per-model inflight guard, and optional circuit breaker.

- `src/infer_nexus/runtime/handles.py`
  - Wraps `serve.get_deployment_handle(...)`.
  - Caches handles by `(app_name, deployment_name)`.

- `src/infer_nexus/runtime/deployments.py`
  - Builds `serve.deployment(...)` kwargs.
  - Already forwards `serve_deployment_kwargs`.
  - Already forwards `request_router_config`.

- `src/infer_nexus/core/config.py`
  - Defines runtime settings:
    - `serve_request_timeout_seconds`
    - `max_inflight_per_model`
    - `admission_acquire_timeout_seconds`
    - `circuit_breaker_enabled`
    - `circuit_breaker_failure_threshold`
    - `circuit_breaker_cooldown_seconds`

- `config/settings.yaml`
  - Current runtime defaults.

- `config/models.yaml`
  - Current model deployment config.
  - Some chat models still enable `PrefixCacheAffinityRouter`.

### Existing Tests

- `tests/test_dispatcher.py`
  - Serve handle invocation.
  - Handle caching.
  - Timeout behavior.
  - Per-model inflight guard.
  - Circuit breaker behavior.

- `tests/test_runtime.py`
  - Serve deployment config plumbing.
  - `request_router_config`.
  - `serve_deployment_kwargs`, including `max_queued_requests` and `max_ongoing_requests`.

- `tests/test_main.py`
  - App lifespan and runtime executor wiring.

## 3. Target Architecture

### Phase 1: Harden Current Gateway

Keep the existing architecture, but make overload behavior explicit:

```text
client
-> FastAPI route
-> model readiness/admission
-> RuntimeExecutor local per-model guard
-> bounded Ray Serve handle call
-> fast timeout / 429 / 504 / circuit open
```

Desired behavior:

- If a model runtime is saturated, reject quickly instead of adding more Ray Serve pending requests.
- If Ray Serve handle/router stops making progress, return a bounded error instead of holding gateway resources for minutes.
- If a model repeatedly times out or fails, open a circuit breaker for that model.
- Ray Serve deployment queues should be bounded so Serve itself applies backpressure.

### Phase 2: Isolate Ray Serve Handle Work

Move Ray Serve handle calls out of the FastAPI gateway process:

```text
FastAPI gateway
-> per-model runtime client
-> per-model worker process
-> Ray Serve DeploymentHandle
```

Desired behavior:

- Ray Serve router CPU or actor-RPC storms affect a per-model worker, not the main HTTP gateway.
- One bad model cannot consume the gateway event loop for every model.
- Workers can be restarted independently.

### Phase 3: Gateway-Owned Routing Snapshot

For future cache locality and load-aware routing, do not synchronously probe Ray actors during request dispatch.

Use:

```text
background sampler
-> local model/replica state snapshot
-> request router reads snapshot only
-> fallback or reject if snapshot is stale
```

Desired behavior:

- Request routing is a local memory read.
- Ray Serve/vLLM metrics are sampled asynchronously.
- Prefix/cache affinity is implemented only if it can degrade safely.

## 4. Implementation Tasks

### Task 1: Add A Safe Production Baseline Config

**Files:**
- Modify: `config/settings.yaml`
- Modify: `config/models.yaml`
- Test: `tests/test_runtime.py`

- [ ] **Step 1: Write a regression test for deployment backpressure kwargs**

Add or extend a test in `tests/test_runtime.py` that validates chat model deployment kwargs include Serve backpressure fields when configured.

Example assertion shape:

```python
assert deployment_kwargs["max_queued_requests"] == 16
assert deployment_kwargs["max_ongoing_requests"] == 1
```

Run:

```bash
uv run pytest tests/test_runtime.py -k "serve_deployment_kwargs" -v
```

Expected before config change:

```text
PASS for existing unit plumbing; no production config assertion yet
```

- [ ] **Step 2: Remove `PrefixCacheAffinityRouter` from production model config**

In `config/models.yaml`, remove or comment the `deployment_config.request_router_config` block for these models:

```yaml
Qwen3-32B
Qwen3.5-9B
```

Do not delete the schema support in code. Only stop enabling the router by default in production config.

- [ ] **Step 3: Add explicit Serve backpressure per chat model**

Add bounded queue settings through `deployment_config.serve_deployment_kwargs`.

Recommended starting point for large chat models:

```yaml
deployment_config:
  serve_deployment_kwargs:
    max_ongoing_requests: 1
    max_queued_requests: 16
```

Use `max_ongoing_requests: 1` for single vLLM engine replicas unless live tests show the engine handles higher concurrency reliably.

- [ ] **Step 4: Tighten gateway runtime defaults**

In `config/settings.yaml`, start with:

```yaml
runtime:
  serve_request_timeout_seconds: 60
  max_inflight_per_model: 4
  admission_acquire_timeout_seconds: 0
  circuit_breaker_enabled: true
  circuit_breaker_failure_threshold: 3
  circuit_breaker_cooldown_seconds: 60
```

If live SLA requires shorter failure windows, test `serve_request_timeout_seconds: 30`.

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/test_runtime.py tests/test_dispatcher.py -v
```

Expected:

```text
All selected tests pass.
```

- [ ] **Step 6: Commit**

```bash
git add config/settings.yaml config/models.yaml tests/test_runtime.py
git commit -m "chore: harden ray serve gateway defaults"
```

### Task 2: Make Circuit Breaker And Admission Errors Operationally Clear

**Files:**
- Modify: `src/infer_nexus/runtime/executor.py`
- Modify: `src/infer_nexus/api/openai_routes.py`
- Test: `tests/test_dispatcher.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Confirm current error mapping**

Read:

```bash
rg -n "RuntimeNotConnectedError|AdmissionRejectedError|_runtime_error_response|circuit" src/infer_nexus tests
```

Expected findings:

```text
RuntimeExecutor raises AdmissionRejectedError for local guard saturation.
RuntimeExecutor raises RuntimeNotConnectedError or RuntimeExecutionError for Serve failures.
openai_routes.py maps runtime exceptions into JSON error responses.
```

- [ ] **Step 2: Add tests for overloaded model response**

In `tests/test_api.py`, add a test that injects a runtime dispatcher raising `AdmissionRejectedError` and verifies the route returns a bounded overload response.

Expected response:

```text
HTTP 429
error code contains admission or overloaded semantics
```

- [ ] **Step 3: Add tests for circuit-open response**

In `tests/test_dispatcher.py`, extend the existing circuit breaker tests to assert:

```python
with pytest.raises(AdmissionRejectedError, match="Circuit breaker is open"):
    asyncio.run(dispatcher.dispatch_chat(registry.get("qwen3-chat"), request))
```

Then verify API mapping in `tests/test_api.py` returns a clear status and error body.

- [ ] **Step 4: Update error text if needed**

If current messages are ambiguous, update exception messages in `src/infer_nexus/runtime/executor.py` so operations can distinguish:

```text
gateway_local_admission_full
serve_request_timeout
serve_circuit_open
serve_handle_resolution_failed
```

Keep labels stable and bounded. Do not include raw prompt text, request body, or Ray replica IDs in public error codes.

- [ ] **Step 5: Run focused tests**

```bash
uv run pytest tests/test_api.py tests/test_dispatcher.py -k "admission or circuit or timeout or runtime" -v
```

Expected:

```text
All selected tests pass.
```

- [ ] **Step 6: Commit**

```bash
git add src/infer_nexus/runtime/executor.py src/infer_nexus/api/openai_routes.py tests/test_dispatcher.py tests/test_api.py
git commit -m "fix: clarify gateway overload and circuit errors"
```

### Task 3: Add Gateway Metrics For Serve Handle Health

**Files:**
- Modify: `src/infer_nexus/observability/metrics.py`
- Modify: `src/infer_nexus/runtime/executor.py`
- Test: `tests/test_proxy_streaming.py` or `tests/test_api.py`
- Test: `tests/test_dispatcher.py`

- [ ] **Step 1: Add metric definitions**

Add metrics with bounded labels:

```text
infer_nexus_serve_handle_calls_total{model,method,status}
infer_nexus_serve_handle_latency_seconds_bucket{model,method}
infer_nexus_serve_handle_timeouts_total{model,method}
infer_nexus_serve_circuit_state{model}
infer_nexus_runtime_guard_rejections_total{model,reason}
```

Avoid these labels:

```text
request_id
prompt
exception_message
ray_replica_id
raw_deployment_id
```

- [ ] **Step 2: Instrument non-streaming Serve calls**

In `RuntimeExecutor._invoke_handle()`, record:

```text
status=success
status=timeout
status=error
```

Start timing immediately before `remote_method.remote(...)` and stop after `_await_handle_response(...)` returns or raises.

- [ ] **Step 3: Instrument streaming Serve calls**

In `RuntimeExecutor._invoke_handle_stream()`, record:

```text
status=started
status=error
```

Stream completion status can continue to use existing stream metrics.

- [ ] **Step 4: Add tests for metrics rendering**

Use existing `render_prometheus_metrics()` patterns and assert metric names appear in rendered output after a fake Serve call.

Example assertion:

```python
body = render_prometheus_metrics().decode()
assert "infer_nexus_serve_handle_calls_total" in body
```

- [ ] **Step 5: Run focused tests**

```bash
uv run pytest tests/test_dispatcher.py tests/test_api.py tests/test_proxy_streaming.py -v
```

Expected:

```text
All selected tests pass.
```

- [ ] **Step 6: Commit**

```bash
git add src/infer_nexus/observability/metrics.py src/infer_nexus/runtime/executor.py tests/test_dispatcher.py tests/test_api.py tests/test_proxy_streaming.py
git commit -m "feat: expose serve handle health metrics"
```

### Task 4: Add A Runtime Worker Isolation Design Before Coding

**Files:**
- Create: `docs/RUNTIME_WORKER_ISOLATION_DESIGN.md`

- [ ] **Step 1: Document the isolation boundary**

Create `docs/RUNTIME_WORKER_ISOLATION_DESIGN.md` with this target flow:

```text
FastAPI gateway
-> RuntimeClient
-> per-model worker process
-> Ray Serve DeploymentHandle
```

The document must define:

- worker lifecycle
- request/response protocol
- stream forwarding behavior
- timeout ownership
- crash/restart behavior
- per-model worker pool sizing
- metrics exposed by gateway vs worker

- [ ] **Step 2: Include the non-goals**

The design must explicitly say:

- no custom prefix router in this phase
- no replacement of Ray Serve deployment management in this phase
- no distributed scheduler in this phase
- no new public API surface in this phase

- [ ] **Step 3: Add acceptance criteria**

Acceptance criteria:

```text
Ray Serve router CPU spike in one model worker does not block /healthz.
Ray Serve router CPU spike in one model worker does not block another model's request path.
Worker restart does not require restarting the FastAPI gateway.
Gateway returns bounded 503/504 when worker is unavailable or timed out.
```

- [ ] **Step 4: Review design against current files**

Map proposed components to existing files:

```text
src/infer_nexus/runtime/executor.py
src/infer_nexus/runtime/dispatcher.py
src/infer_nexus/runtime/handles.py
src/infer_nexus/main.py
config/settings.yaml
```

- [ ] **Step 5: Commit**

```bash
git add docs/RUNTIME_WORKER_ISOLATION_DESIGN.md
git commit -m "docs: design runtime worker isolation"
```

### Task 5: Implement Per-Model Runtime Worker Isolation

**Files:**
- Create: `src/infer_nexus/runtime/worker_client.py`
- Create: `src/infer_nexus/runtime/worker_process.py`
- Modify: `src/infer_nexus/runtime/executor.py`
- Modify: `src/infer_nexus/main.py`
- Modify: `src/infer_nexus/core/config.py`
- Test: `tests/test_runtime_worker.py`

- [ ] **Step 1: Add settings for worker mode**

Add runtime config fields:

```python
runtime_worker_enabled: bool = False
runtime_worker_request_timeout_seconds: int | float = 60
runtime_worker_start_timeout_seconds: int | float = 15
```

Keep default disabled so existing tests and local dev behavior remain stable.

- [ ] **Step 2: Write tests for disabled mode**

Add a test that confirms default app startup still wires `RuntimeExecutor` directly and does not start workers.

Run:

```bash
uv run pytest tests/test_main.py -v
```

Expected:

```text
Existing app startup tests still pass.
```

- [ ] **Step 3: Implement a minimal worker client interface**

Create `src/infer_nexus/runtime/worker_client.py` with an interface that can later use a process transport:

```python
from collections.abc import AsyncIterator

from starlette.responses import Response

from infer_nexus.core.schemas import (
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
)
from infer_nexus.runtime.types import RuntimeTarget


class RuntimeWorkerClient:
    async def chat_completion(
        self,
        *,
        target: RuntimeTarget,
        request: ChatCompletionsRequest,
    ) -> ChatCompletionsResponse | Response:
        raise NotImplementedError

    async def chat_completion_stream(
        self,
        *,
        target: RuntimeTarget,
        request: ChatCompletionsRequest,
    ) -> AsyncIterator[dict | bytes | str]:
        raise NotImplementedError

    async def embedding(
        self,
        *,
        target: RuntimeTarget,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse | Response:
        raise NotImplementedError

    async def rerank(
        self,
        *,
        target: RuntimeTarget,
        request: RerankRequest,
    ) -> RerankResponse | Response:
        raise NotImplementedError
```

The first implementation can be an in-process adapter used by tests.

- [ ] **Step 4: Add executor delegation path**

In `RuntimeExecutor`, add an optional worker client. If configured, delegate Serve-mode calls to the worker client instead of calling `_invoke_handle()` directly.

Keep these paths unchanged:

- `BackendType.VLLM_OPENAI_PROXY`
- local stub mode
- response normalization
- stream response wrapping

- [ ] **Step 5: Add process transport in a separate commit**

Implement the actual worker process only after the in-process interface and tests are green.

Transport options to evaluate in the design:

- local HTTP over loopback
- Unix domain socket
- multiprocessing queue

Prefer the simplest option that supports streaming without complex custom framing.

- [ ] **Step 6: Add tests**

Test cases:

```text
worker disabled: direct executor path is used
worker enabled: executor delegates to worker client
worker timeout: gateway returns bounded runtime error
worker error: error is mapped to existing runtime error response
streaming worker response: StreamingResponse still yields SSE chunks
```

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest tests/test_runtime_worker.py tests/test_dispatcher.py tests/test_api.py -v
```

Expected:

```text
All selected tests pass.
```

- [ ] **Step 8: Commit**

```bash
git add src/infer_nexus/runtime/worker_client.py src/infer_nexus/runtime/worker_process.py src/infer_nexus/runtime/executor.py src/infer_nexus/main.py src/infer_nexus/core/config.py tests/test_runtime_worker.py
git commit -m "feat: isolate ray serve handle calls in runtime workers"
```

### Task 6: Design A Local-Snapshot Router For Future Prefix Affinity

**Files:**
- Create: `docs/LOCAL_SNAPSHOT_ROUTER_DESIGN.md`

- [ ] **Step 1: Define snapshot inputs**

Document the metrics to sample asynchronously:

```text
per-model inflight requests
per-model rejection count
Ray Serve scheduling task count
Ray Serve scheduling backoff count
Ray Serve replica health
vLLM running requests
vLLM waiting requests
vLLM KV cache usage
vLLM prefix cache hit metrics if available
```

- [ ] **Step 2: Define request-time routing rules**

Request-time routing must read local state only:

```text
if snapshot is fresh and preferred replica has capacity: route preferred
if snapshot is fresh but preferred replica is full: route fallback
if snapshot is stale: route default or reject quickly
if model circuit is open: reject quickly
```

- [ ] **Step 3: Define prefix-affinity safety rules**

Prefix affinity must be optional and safe:

```text
cache locality never overrides hard capacity
stale prefix map is ignored
prefix map is per model
prefix key never stores raw prompt text
prefix key uses bounded hash metadata
```

- [ ] **Step 4: Define the first implementation boundary**

The first implementation should be a planning document only. Do not implement custom replica selection until Phase 1 and Phase 2 are stable in live tests.

- [ ] **Step 5: Commit**

```bash
git add docs/LOCAL_SNAPSHOT_ROUTER_DESIGN.md
git commit -m "docs: design local snapshot routing"
```

### Task 7: Live Validation Runbook

**Files:**
- Create: `docs/RAY_SERVE_GATEWAY_VALIDATION_RUNBOOK.md`

- [ ] **Step 1: Define A/B experiments**

Use the same model, prompt set, concurrency, and max tokens.

Compare:

```text
A: PrefixCacheAffinityRouter enabled
B: PrefixCacheAffinityRouter disabled
C: disabled + gateway backpressure + Serve backpressure + circuit breaker
```

- [ ] **Step 2: Define success metrics**

Record:

```text
ok count
timeout count
error count
p50 latency
p90 latency
p99 latency
TTFT
TPOT
gateway CPU
gateway GIL
Ray queue-length warning count
Ray Serve scheduling task metrics
GPU utilization
vLLM running/waiting requests
```

- [ ] **Step 3: Define stop conditions**

Stop a test run when any of these occur:

```text
gateway CPU stays near 100% for more than 60 seconds
Ray queue-length warning rate exceeds 1000/minute
timeout rate exceeds 20%
gateway log grows faster than 100MB/minute
GPU utilization is near zero while request latency rises
```

- [ ] **Step 4: Define commands**

Use existing benchmark tooling where possible:

```bash
uv run python scripts/run_benchmark.py --help
uv run python scripts/monitoring/inspect_ray_serve_vllm_metrics.py --help
```

Use `py-spy` only on target machines where it is already installed and approved.

- [ ] **Step 5: Commit**

```bash
git add docs/RAY_SERVE_GATEWAY_VALIDATION_RUNBOOK.md
git commit -m "docs: add ray serve gateway validation runbook"
```

## 5. Recommended Execution Order

1. Task 1: safe production baseline config.
2. Task 2: clearer overload/circuit errors.
3. Task 3: Serve handle health metrics.
4. Task 7: live validation runbook.
5. Run A/B validation on `Qwen3-32B` and `Qwen3.5-9B`.
6. Task 4: runtime worker isolation design.
7. Task 5: runtime worker isolation implementation.
8. Task 6: local-snapshot router design.

Do not implement custom prefix-affinity routing before Tasks 1-3 have been validated live. The first objective is to stop gateway stalls and control-plane amplification.

## 6. Tomorrow's First Work Session

Start here:

1. Open this file.
2. Open `config/settings.yaml`.
3. Open `config/models.yaml`.
4. Disable `PrefixCacheAffinityRouter` for `Qwen3-32B` and `Qwen3.5-9B`.
5. Add `serve_deployment_kwargs.max_ongoing_requests` and `serve_deployment_kwargs.max_queued_requests` for large chat models.
6. Enable gateway circuit breaker.
7. Run:

```bash
uv run pytest tests/test_runtime.py tests/test_dispatcher.py -v
```

8. Deploy to the remote test machine.
9. Run the same short-prompt, long-prompt, streaming, and concurrent tests used in the Notion investigation.
10. Compare queue-length warning count, gateway CPU, latency, timeout count, and GPU utilization.

## 7. Open Decisions

These decisions should be made after live validation, not before:

- Whether `serve_request_timeout_seconds` should be `30` or `60`.
- Whether `max_inflight_per_model` should be lower than `4` for very large models.
- Whether `max_queued_requests` should be `8`, `16`, or `32`.
- Whether runtime worker isolation should use local HTTP, Unix domain socket, or multiprocessing queues.
- Whether prefix/cache affinity should be implemented by a custom Ray-side router or a gateway-owned local-snapshot router.

## 8. Verification Checklist

Before considering this work successful:

- [ ] Gateway `/healthz` remains responsive during backend overload.
- [ ] One overloaded model does not block another model.
- [ ] Queue-length warning storms are reduced or bounded.
- [ ] Gateway logs do not grow uncontrollably.
- [ ] Gateway returns bounded 429/503/504 responses under overload.
- [ ] Stream requests still terminate cleanly on timeout or cancellation.
- [ ] Prometheus metrics expose Serve handle latency, timeout, and rejection signals.
- [ ] Live benchmark shows improved p90/p99 latency or reduced timeout rate under the same load.

## 9. Notes On Existing Documentation

`docs/PREFIX_CACHE_STICKY_DESIGN.md` currently says Ray Serve `PrefixCacheAffinityRouter` is the preferred direction. The live investigation changed that recommendation for the current custom handle-call path.

Treat the new position as:

- Ray Serve deployment management is still useful.
- Ray Serve handle calls are still the current execution mechanism.
- Ray Serve `PrefixCacheAffinityRouter` should not be enabled by default in production until proven stable for `handle.chat_completion.remote(request_payload=dict)`.
- Prefix/cache affinity should be revisited only after gateway backpressure, metrics, and isolation are reliable.
