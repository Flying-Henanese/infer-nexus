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

- `src/infer_nexus/main.py` loads the selected settings file: `config/settings.yaml` by default, `config/settings.compose.yaml` in the root CUDA Compose flow, `config/settings.ascend-compose.yaml` in the Ascend Compose flow, or `config/settings.k8s.yaml` for the KubeRay smoke rollout.
- The catalog is loaded from one configured YAML file. The default and Compose flows select `config/models.yaml`; the current KubeRay rollout selects `config/models.k8s-smoke.yaml`.
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
Uvicorn gateway service in serve mode. The CUDA worker registers GPU resources
derived from `CUDA_VISIBLE_DEVICES`; the Ascend worker registers custom `NPU`
resources derived from `ASCEND_RT_VISIBLE_DEVICES`. Platform-specific
accelerator details stay in image, Compose, environment, and settings files
rather than request-path code.

The KubeRay shape uses the same in-Ray Gateway application. Its stable
`infer-nexus-gateway` NodePort Service selects the Ray head Serve HTTP proxy on
port 8000; it does not create a Uvicorn Deployment or a Ray Client connection.
`deploy/kuberay/serve-deployer.job.yaml` is one-shot and submits all
catalog-derived model applications before the Gateway application. The current
KubeRay settings select the one-model `config/models.k8s-smoke.yaml` catalog;
the five-model catalog requires a separate capacity-planned rollout.

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

- `AdmissionController.check_model_request()` is a no-op integration hook. Active protection currently comes from task/model validation, artifact checks, process-local worker admission, and `RuntimeExecutor` guards; readiness- and cluster-capacity-aware admission are not implemented.
- The full unit suite is not green against the current catalog. Many API, dispatcher, runtime, and script tests still expect the previous three-model fixture and old aliases. See `.harness/checklists/verification.md` before interpreting full-suite failures.

## Out Of Current Scope

Do not treat these as current architecture:

- dynamic model registration
- multi-node placement/control plane
- automatic heterogeneous hardware scheduling in application logic
- gateway-owned prefix-cache sticky routing
- active runtime-worker process isolation
