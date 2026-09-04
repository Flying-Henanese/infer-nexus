# Project Map

Use this file to choose the first source files to inspect. It is a navigation map, not a substitute for reading the code before changing behavior.

## Root Layout

- `src/infer_nexus/`: application package.
- `config/`: runtime settings and model catalog.
- `scripts/`: executable operational entry points and diagnostics.
- `tests/`: unit, API, dispatcher, runtime, benchmark, and smoke-test coverage.
- `docs/`: long-form architecture, operations, monitoring, and historical design notes.
- `monitoring/`: Prometheus example configuration.
- `Dockerfile`: root CUDA-oriented container runtime image definition.
- `docker-compose.yml`: root CUDA containerized runtime topology.
- `ascend_deploy/`: Ascend-specific image and Compose topology.
- `pyproject.ascend.toml`: application dependency set used when building the Ascend image without replacing its vendor runtime stack.
- `.harness/`: Codex collaboration context, rules, and checklists.

## Application Package

- `src/infer_nexus/main.py`
  - Reusable FastAPI app factory and host-specific lifespan wiring.
  - Loads settings, catalog, model store, admission, Serve builder, runtime executor, worker admission, and dispatcher into `app.state`.

- `src/infer_nexus/api/`
  - HTTP surface.
  - `openai_routes.py`: `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`, and `/rerank`.
  - `platform_routes.py`: read-only catalog, model status, cluster load, and capacity APIs.
  - `health_routes.py`: `/healthz` and `/readyz`.
  - `metrics_routes.py`: `/metrics`.
  - `deps.py`: FastAPI dependency accessors for `app.state`.
  - `worker_admission_middleware.py`: process-local gateway worker admission middleware.

- `src/infer_nexus/catalog/`
  - Model catalog schema, loading, and lookup.
  - `models.py`: Pydantic model configuration schema.
  - `loader.py`: loads one YAML catalog file.
  - `registry.py`: in-memory lookup by canonical name, alias, and served model name.

- `src/infer_nexus/runtime/`
  - Runtime dispatch and execution layer.
  - `dispatcher.py`: resolves `ModelConfig` into `RuntimeTarget` and calls the executor.
  - `executor.py`: executes stub, local Ray Serve handle, streaming, and proxy paths.
  - `serve_app.py`: builds Ray Serve application bindings and runtime context.
  - `deployments.py`: maps model configs to Ray Serve deployment specs.
  - `handles.py`: caches Ray Serve deployment handles.
  - `gateway_ingress.py`: CPU-only Serve ASGI ingress binding and static public route configuration.
  - `worker_client.py`: runtime-worker client scaffolding, not an active isolation path.

- `src/infer_nexus/backends/`
  - Local inference backend adapters.
  - `base.py`: backend interface.
  - `vllm.py`: main local vLLM backend facade.
  - `vllm_strict.py`: native vLLM OpenAI serving path.
  - `vllm_local_best_effort.py`: project-maintained local best-effort path.
  - `vllm_native/`: adapters around vLLM OpenAI serving internals.

- `src/infer_nexus/control/`
  - Control-plane helpers.
  - `admission.py`: no-op model-level admission integration hook; readiness, queue, and cluster-capacity decisions are not implemented there.
  - `worker_admission.py`: process-local gateway worker inflight guard.
  - `load_inspector.py`: current load snapshot.
  - `scaler.py` and `policies.py`: scaling policy scaffolding.
  - `reconciler.py`: skeleton only; dynamic reconciliation is not implemented.

- `src/infer_nexus/core/`
  - Shared schemas, config, enums, and error types.

- `src/infer_nexus/observability/`
  - Prometheus metrics and logging setup.

- `src/infer_nexus/benchmark/`
  - Benchmark runner, workloads, client, tokenizer, metrics, and report generation.

- `src/infer_nexus/model_store.py`
  - Local model artifact resolution and validation.

- `src/infer_nexus/auth/`
  - API key helper code. Authentication is not the primary current focus.

- `src/infer_nexus/artifacts/`
  - Model/artifact download helpers.

## Configuration

- `config/settings.yaml`
  - Default service, catalog path, scheduler, runtime mode, gateway guard, model store, and cluster settings.

- `config/settings.compose.yaml`
  - Root Compose settings entrypoint passed to `serve-deployer` and Gateway ingress replicas.

- `config/settings.ascend-compose.yaml`
  - Ascend Compose settings entrypoint; requests custom Ray `NPU` resources.

- `config/models.yaml`
  - Active model catalog.
  - The current validation catalog enables only the local Qwen3.5-9B
    `backend: vllm` model; the remaining catalog entries are commented out.
  - Ray Serve request-router options, including cache-affinity router class settings, belong under each model's `deployment_config.request_router_config`.

## Container Runtimes

- `docker-compose.yml`
  - Defines `ray-head`, `ray-worker`, and one-shot `serve-deployer`; `ray-head:8000` exposes the Serve Gateway ingress.
  - Compose owns container lifecycle; Ray Serve owns deployment and replica lifecycle after apps are submitted.

- `Dockerfile`
  - Builds the root CUDA-oriented shared runtime image.

- `ascend_deploy/docker-compose.yml`
  - Defines the same three Compose services as CUDA: `ray-head`, `ray-worker`,
    and one-shot `serve-deployer`.
  - Registers custom Ray `NPU` resources from `ASCEND_RT_VISIBLE_DEVICES`; the
    public Gateway remains a CPU-only Serve deployment behind `ray-head:8000`.

- `ascend_deploy/dockerfile`
  - Builds on an Ascend runtime image and installs application dependencies without intentionally replacing the image-provided torch/vLLM/vLLM-Ascend stack.

## Scripts

- `scripts/run_gateway.py`
  - Starts the FastAPI app only for local/stub debugging.

- `scripts/run_serve_runtime.py`
  - Builds and submits Ray Serve applications for configured local `backend: vllm` models.

- `scripts/run_benchmark.py`
  - CLI wrapper for the benchmark runner.

- `scripts/start_minimal.sh`
  - Local CUDA-oriented bootstrap that registers Ray GPU resources.

- `scripts/start_minimal_ascend.sh`
  - Local Ascend bootstrap that registers custom Ray `NPU` resources.

- `scripts/stop_minimal.sh`
  - Stops local runtime/Ray processes tracked by the minimal startup scripts.

- `scripts/monitoring/inspect_ray_serve_vllm_metrics.py`
  - Inspects Ray Serve and vLLM metric visibility from Prometheus or raw metrics endpoints.

- `scripts/environment_check/`
  - Accelerator dependency checks/install helpers for performance monitoring environments.

## Tests

- `tests/test_api.py`: OpenAI-compatible and platform API behavior.
- `tests/test_dispatcher.py`: runtime dispatch, executor, Serve handle, streaming, proxy behavior.
- `tests/test_runtime.py`: runtime config, Serve deployment, backend adapter behavior.
- `tests/test_main.py`: app lifespan and dependency wiring.
- `tests/test_proxy_streaming.py`: proxy streaming behavior.
- `tests/test_model_store.py`: local model artifact resolution.
- `tests/test_config.py`: settings loading and environment override behavior.
- `tests/test_scripts.py`: gateway and Serve launcher behavior.
- `tests/test_gateway_ingress.py`: reusable ASGI Gateway ingress component behavior.
- `tests/test_benchmark_*.py`: benchmark runner, report, metrics, and workloads.
- Smoke scripts under `tests/` require local model/runtime context and are not generic unit tests.
- See `.harness/checklists/verification.md` for the current full-suite baseline before interpreting failures.
