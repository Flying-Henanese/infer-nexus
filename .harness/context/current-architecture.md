# Current Architecture Baseline

This file describes the architecture that is currently implemented in source code. Treat older design documents as background unless this baseline points to them.

## Request Path

Current OpenAI-compatible inference path:

```text
client
-> Ray Serve HTTP proxy
-> InferNexusGatewayIngress (FastAPI API surface)
-> api/openai_routes.py
-> api/deps.py
-> ModelRegistry
-> RuntimeDispatcher
-> RuntimeExecutor
-> local vllm / Ray Serve DeploymentHandle / vllm_openai_proxy
```

`runtime/worker_client.py` defines a future runtime-worker isolation boundary, but `main.py` does not construct or inject a runtime-worker client. It is not an active execution path.

Local `backend: vllm` models use Ray Serve deployments in serve mode:

```text
RuntimeExecutor
-> ServeDeploymentHandleResolver
-> Ray Serve deployment handle
-> ModelRuntimeReplica
-> VLLMBackend
-> replica-local vLLM runtime
```

Proxy models use `backend: vllm_openai_proxy` and are forwarded to an upstream OpenAI-compatible endpoint through `RuntimeExecutor`.

## Startup And State

- `src/infer_nexus/main.py` loads the selected settings file: `config/settings.yaml` by default, `config/settings.compose.yaml` in the root CUDA Compose flow, or `config/settings.ascend-compose.yaml` in the Ascend Compose flow.
- The catalog is loaded from one configured YAML file, currently `config/models.yaml`.
- `ModelRegistry`, `LocalModelStore`, `ServeApplicationBuilder`, `RuntimeExecutor`, `WorkerAdmissionController`, and `RuntimeDispatcher` are attached to `app.state`.
- Platform APIs are currently read-only catalog/status/load/capacity views.
- `control/reconciler.py` exists only as a skeleton.

## Containerized Runtime Shapes

The repository has two current Compose entrypoints with the same service topology:

- root `docker-compose.yml` + root `Dockerfile` + `config/settings.compose.yaml` for the CUDA path
- `ascend_deploy/docker-compose.yml` + `ascend_deploy/dockerfile` + `config/settings.ascend-compose.yaml` for the Ascend NPU path

Both use this runtime shape:

```text
client
-> ray-head Serve HTTP proxy
-> InferNexusGatewayIngress
-> Ray Serve deployment handle
-> Ray Serve replica in ray-worker
-> replica-local vLLM runtime
```

Compose owns `ray-head`, `ray-worker`, and the one-shot `serve-deployer`.
Ray Serve owns the CPU-only public Gateway ingress plus model deployment and
replica lifecycle after `serve-deployer` submits the applications.
`ray-head:8000` is the Compose public API port; there is no separate Compose
Uvicorn gateway service in Serve mode. Only `ray-head` and `ray-worker` remain
running after the deployment job exits successfully. The CUDA worker registers
GPU resources derived from `CUDA_VISIBLE_DEVICES`; the Ascend worker registers
custom `NPU` resources derived from `ASCEND_RT_VISIBLE_DEVICES`.
Platform-specific accelerator details stay in image, Compose, environment, and
settings files rather than request-path code.

## Verified Runtime Status

- The root CUDA Compose path has been validated end to end on A100 with the
  active Qwen3.5-9B catalog: host requests to `127.0.0.1:8000` (mapped to
  `ray-head:8000`) reached the Serve HTTP proxy, Gateway ingress, a vLLM model
  replica, and the HTTP/SSE response path. `/healthz`, `/readyz`, `/v1/models`,
  non-streaming chat, and streaming chat were checked.
- The Ascend Compose file has the same three-service topology and the same
  Gateway deployment path. Its Compose rendering and NPU resource mapping have
  been checked locally, but it has not yet received an end-to-end run on an
  Ascend host.

## Current Control Boundaries

- `infer-nexus` owns public HTTP ingress, model lookup, request validation, admission checks, proxying, and app-level metrics.
- Ray Serve owns local deployment lifecycle, replica placement, replica routing, and Serve-level metrics.
- vLLM owns replica-local model execution for local models.
- Cache affinity, when enabled, is configured through `config/models.yaml` deployment router settings such as `deployment_config.request_router_config.request_router_class`.
- Upstream OpenAI-compatible servers own protocol behavior for proxy models.

## Implemented Runtime Protections

- Process-local gateway worker admission is wired through `WorkerAdmissionMiddleware`.
- Runtime executor has per-model guards, bounded queue settings, handle timeouts, stream idle/lifetime timeouts, and optional circuit breaker settings.
- Serve deployment handles are cached by `(app_name, deployment_name)`.
- Streaming handles must support `options(stream=True)`; unavailable capability
  fails explicitly and never falls back to unary invocation.

## Known Checked-in Integration Gaps

- `config/settings.yaml` now defaults to CUDA and matches
  `scripts/start_minimal.sh`. `scripts/start_minimal_ascend.sh` still defaults
  to `config/settings.yaml`, so callers must pass
  `--settings config/settings.ascend-compose.yaml` for an Ascend launch.
- The active Qwen3.5-9B catalog entry uses a relative path that resolves below
  the Compose `/models` mount. Commented historical entries intentionally
  retain environment-specific absolute paths (for example `/app/models/...`
  and `/nas_data/...`). Before re-enabling one, verify that its path matches
  the selected runtime's model-store root and mounts; do not assume every
  environment should use the Compose `/models` path.
- Ascend starts the worker with
  `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1` by default for runtime
  compatibility. A real Ascend run must prove that Ray actor allocation and
  `ASCEND_RT_VISIBLE_DEVICES` agree before multi-replica autoscaling is relied
  upon.
- `AdmissionController.check_model_request()` is a no-op integration hook. Active protection currently comes from task/model validation, artifact checks, process-local worker admission, and `RuntimeExecutor` guards; readiness- and cluster-capacity-aware admission are not implemented.
- The full unit suite is not green against the current catalog. Many API, dispatcher, runtime, and script tests still expect the previous three-model fixture and old aliases. See `.harness/checklists/verification.md` before interpreting full-suite failures.

## Out Of Current Scope

Do not treat these as current architecture:

- dynamic model registration
- multi-node placement/control plane
- automatic heterogeneous hardware scheduling in application logic
- gateway-owned prefix-cache sticky routing
- active runtime-worker process isolation
