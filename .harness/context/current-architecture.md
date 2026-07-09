# Current Architecture Baseline

This file describes the architecture that is currently implemented in source code. Treat older design documents as background unless this baseline points to them.

## Request Path

Current OpenAI-compatible inference path:

```text
client
-> FastAPI gateway
-> api/openai_routes.py
-> api/deps.py
-> ModelRegistry
-> RuntimeDispatcher
-> RuntimeExecutor
-> local vllm / Ray Serve DeploymentHandle / vllm_openai_proxy
```

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

- `src/infer_nexus/main.py` loads the selected settings file, usually `config/settings.yaml` locally or `config/settings.compose.yaml` under the root Compose flow.
- The catalog is loaded from one configured YAML file, currently `config/models.yaml`.
- `ModelRegistry`, `LocalModelStore`, `ServeApplicationBuilder`, `RuntimeExecutor`, `WorkerAdmissionController`, and `RuntimeDispatcher` are attached to `app.state`.
- Platform APIs are currently read-only catalog/status/load/capacity views.
- `control/reconciler.py` exists only as a skeleton.

## Containerized Runtime Shape

The root `docker-compose.yml` is the current containerized runtime entrypoint:

```text
client
-> gateway container
-> Ray Serve deployment handle
-> Ray Serve replica in ray-worker
-> replica-local vLLM runtime
```

Compose owns container lifecycle for `ray-head`, `ray-worker`, one-shot `serve-deployer`, and long-running `gateway`. Ray Serve owns local deployment and replica lifecycle after `serve-deployer` submits the applications. Platform-specific accelerator details should stay in Docker image, Compose, environment, and settings files, not in request-path code.

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

## Out Of Current Scope

Do not treat these as current architecture:

- dynamic model registration
- multi-node placement/control plane
- automatic heterogeneous hardware scheduling in application logic
- gateway-owned prefix-cache sticky routing

