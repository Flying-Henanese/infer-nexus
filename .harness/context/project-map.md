# Project Map

Use this file to find the first source and test files for a task. It is a
navigation map; verify behavior in code before changing it.

## Source And Tests

- Gateway app and HTTP APIs: `src/infer_nexus/main.py` builds the FastAPI app;
  `src/infer_nexus/gateway_runtime.py` constructs and attaches its dependencies;
  `src/infer_nexus/api/` owns OpenAI-compatible, platform, health, and metrics
  routes plus request and worker-admission middleware. Start with
  `tests/test_main.py`, `tests/test_api.py`, and `tests/test_gateway_ingress.py`.
- Catalog and model artifacts: `src/infer_nexus/catalog/` owns schema, YAML
  loading, and lookup; `src/infer_nexus/model_store.py` resolves local
  artifacts. Start with `tests/test_config.py` and `tests/test_model_store.py`.
  Read `.harness/context/model-config.md` for the active catalog and path rules.
- Dispatch and Serve runtime: `src/infer_nexus/runtime/` owns target dispatch,
  execution, deployment bindings, handle resolution, Gateway ingress, and
  proxy request-ID normalization. Start with `tests/test_dispatcher.py`,
  `tests/test_runtime.py`, `tests/test_proxy_streaming.py`, and
  `tests/test_request_id_proxy_middleware.py`.
- Local inference backends: `src/infer_nexus/backends/` isolates vLLM engine
  and native serving adapters. Read `docs/inference_backend_design.md` before
  changing compatibility behavior; start with `tests/test_runtime.py`.
- Admission and scaling: `src/infer_nexus/control/` owns worker admission,
  load inspection, and scaling/reconciliation scaffolding. Model-level
  `AdmissionController` and reconciliation are not fully implemented; check
  `.harness/context/current-architecture.md` before relying on them.
- Logs, metrics, and request identity: `src/infer_nexus/observability/`,
  `src/infer_nexus/core/request_context.py`, the API request middleware, and
  `src/infer_nexus/runtime/request_id_proxy_middleware.py`. Start with
  `tests/test_logging.py`, `tests/test_ray_logging.py`,
  `tests/test_request_id_proxy_middleware.py`, and
  `tests/test_compose_logging.py`.
- Structured event queries: `src/infer_nexus/observability/log_query.py` and
  `scripts/logs.py`. Start with `tests/test_log_query.py`; it covers safe
  service/path discovery, request correlation, Ray session ownership, malformed
  and partial JSONL, rotation/follow, and storage stats.
- Benchmarking: `src/infer_nexus/benchmark/` and
  `scripts/run_benchmark.py`; start with `tests/test_benchmark_*.py`.

`src/infer_nexus/core/` holds shared settings, schemas, enums, and errors;
`auth/` and `artifacts/` contain API-key helpers and download support.

## Configuration And Runtime Entry Points

- `config/settings.yaml` is the local default; `config/settings.compose.yaml`
  and `config/settings.ascend-compose.yaml` select the two Compose profiles.
  `config/models.yaml` is the single active catalog file.
- Root `Dockerfile` and `docker-compose.yml` define the CUDA runtime.
  `ascend_deploy/dockerfile` and `ascend_deploy/docker-compose.yml` define
  the Ascend runtime; `pyproject.ascend.toml` preserves its vendor stack.
- `scripts/run_serve_runtime.py` submits the model and Gateway Serve apps.
  `scripts/run_gateway.py` starts a standalone FastAPI app for local debugging.
  `scripts/start_minimal*.sh` and `scripts/stop_minimal.sh` manage local Ray
  startup and shutdown; see `.harness/context/request-flows.md` for mode
  differences and `tests/test_scripts.py` for launcher checks.
- `scripts/prepare_compose_logs.py` prepares host log directories.
  `scripts/monitoring/` and `scripts/environment_check/` contain monitoring
  diagnostics and accelerator dependency checks.

Use `.harness/checklists/verification.md` to select change-specific checks.
Smoke scripts under `tests/` need a local model/runtime and are not generic
unit tests.
