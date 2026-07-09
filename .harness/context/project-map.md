# Project Map

Use this file to choose the first source files to inspect. It is a navigation map, not a substitute for reading the code before changing behavior.

## Root Layout

- `src/infer_nexus/`: application package.
- `config/`: runtime settings and model catalog.
- `scripts/`: executable operational entry points and diagnostics.
- `tests/`: unit, API, dispatcher, runtime, benchmark, and smoke-test coverage.
- `docs/`: long-form architecture, operations, monitoring, and historical design notes.
- `monitoring/`: Prometheus example configuration.
- `dockerfile`: root container runtime image definition for the current Compose flow.
- `docker-compose.yml`: root containerized runtime topology.
- `.harness/`: Codex collaboration context, plans, checklists, and run summaries.

## Application Package

- `src/infer_nexus/main.py`
  - FastAPI app factory and lifespan wiring.
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
  - `admission.py`: model-level admission checks.
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
  - Root Compose settings entrypoint used by `serve-deployer` and `gateway`.

- `config/models.yaml`
  - Active model catalog.
  - Current enabled entries are local `backend: vllm` models unless changed in this file.
  - Ray Serve request-router options, including cache-affinity router class settings, belong under each model's `deployment_config.request_router_config`.

## Container Runtime

- `docker-compose.yml`
  - Defines `ray-head`, `ray-worker`, one-shot `serve-deployer`, and long-running `gateway`.
  - Compose owns container lifecycle; Ray Serve owns deployment and replica lifecycle after apps are submitted.

- `dockerfile`
  - Builds the shared runtime image. Keep platform-specific dependency choices in image/build settings rather than request-path code.

## Scripts

- `scripts/run_gateway.py`
  - Starts the FastAPI gateway.

- `scripts/run_serve_runtime.py`
  - Builds and submits Ray Serve applications for configured local `backend: vllm` models.

- `scripts/run_benchmark.py`
  - CLI wrapper for the benchmark runner.

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
- `tests/test_benchmark_*.py`: benchmark runner, report, metrics, and workloads.
- Smoke scripts under `tests/` require local model/runtime context and are not generic unit tests.

