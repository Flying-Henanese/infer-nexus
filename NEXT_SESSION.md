# Next Session Handoff

## What Was Completed

### Documentation
- Created [ARCHITECTURE.md](./ARCHITECTURE.md) as the architecture source of truth
- Created [AGENT.md](./AGENT.md) as the implementation guardrail file for Codex
- Updated `ARCHITECTURE.md` to clarify platform-level resource pool boundaries vs per-deployment resource requests
- Migrated the repository expectations in both docs to a `src/` layout

### Repository Structure
- Migrated Python source code to `src/infer_nexus/`
- Added `tests/` and `scripts/` directories
- Added project metadata and dependency management in [pyproject.toml](./pyproject.toml)
- Configured `uv` to use the Tsinghua PyPI mirror for mainland China network conditions

### Configuration
- Added [config/settings.yaml](./config/settings.yaml)
- Added [config/models.yaml](./config/models.yaml)
- Current model catalog includes:
  - `qwen3-chat` -> `qwen3-32b-instruct`
  - `bge-embedding` -> `bge-large-zh-v1_5`
  - `bge-rerank` -> `bge-reranker-v2-m3`

### FastAPI Skeleton
- Added application entrypoint: [src/infer_nexus/main.py](./src/infer_nexus/main.py)
- Added platform and OpenAI route modules:
  - [src/infer_nexus/api/health_routes.py](./src/infer_nexus/api/health_routes.py)
  - [src/infer_nexus/api/platform_routes.py](./src/infer_nexus/api/platform_routes.py)
  - [src/infer_nexus/api/openai_routes.py](./src/infer_nexus/api/openai_routes.py)
- Added dependency wiring in [src/infer_nexus/api/deps.py](./src/infer_nexus/api/deps.py)

### Current API Surface
Implemented and tested:
- `GET /healthz`
- `GET /readyz`
- `GET /v1/models`
- `GET /api/catalog/models`
- `GET /api/catalog/models/{model_name}`
- `GET /api/models/{model_name}/status`
- `GET /api/cluster/load`
- `GET /api/cluster/capacity`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`

Important current behavior:
- `/v1/chat/completions` and `/v1/embeddings` are protocol-complete stubs
- They validate request shape, resolve model aliases, enforce task-type compatibility, run admission checks, and return OpenAI-style errors
- They currently return `501 runtime_not_connected` because the backend dispatch path is not wired yet

### Core Internal Modules
Added foundational modules for:
- config loading: [src/infer_nexus/core/config.py](./src/infer_nexus/core/config.py)
- shared enums/errors/schemas:
  - [src/infer_nexus/core/enums.py](./src/infer_nexus/core/enums.py)
  - [src/infer_nexus/core/errors.py](./src/infer_nexus/core/errors.py)
  - [src/infer_nexus/core/schemas.py](./src/infer_nexus/core/schemas.py)
- model catalog:
  - [src/infer_nexus/catalog/models.py](./src/infer_nexus/catalog/models.py)
  - [src/infer_nexus/catalog/loader.py](./src/infer_nexus/catalog/loader.py)
  - [src/infer_nexus/catalog/registry.py](./src/infer_nexus/catalog/registry.py)
- control-plane placeholders:
  - [src/infer_nexus/control/admission.py](./src/infer_nexus/control/admission.py)
  - [src/infer_nexus/control/load_inspector.py](./src/infer_nexus/control/load_inspector.py)
  - [src/infer_nexus/control/reconciler.py](./src/infer_nexus/control/reconciler.py)
  - [src/infer_nexus/control/scaler.py](./src/infer_nexus/control/scaler.py)
  - [src/infer_nexus/control/policies.py](./src/infer_nexus/control/policies.py)
- backend abstraction:
  - [src/infer_nexus/backends/base.py](./src/infer_nexus/backends/base.py)
  - [src/infer_nexus/backends/vllm.py](./src/infer_nexus/backends/vllm.py)
- observability placeholders:
  - [src/infer_nexus/observability/logging.py](./src/infer_nexus/observability/logging.py)
  - [src/infer_nexus/observability/metrics.py](./src/infer_nexus/observability/metrics.py)

### Ray Serve Skeleton
Implemented an optional `ray[serve]` integration skeleton without requiring Ray at import time.

Key files:
- [src/infer_nexus/runtime/deployments.py](./src/infer_nexus/runtime/deployments.py)
- [src/infer_nexus/runtime/serve_app.py](./src/infer_nexus/runtime/serve_app.py)

Current runtime layer supports:
- `DeploymentSpec` generation per model
- per-model deployment naming
- per-replica resource declarations (`num_cpus`, `num_gpus`)
- autoscaling bounds (`min_replicas`, `max_replicas`)
- backend runtime context generation
- conversion from deployment spec to Ray Serve deployment kwargs
- `build_serve_bindings()` that calls `serve.deployment(...).bind(...)`
- placeholder replica class `ModelRuntimeReplica`

Important current limitation:
- `ModelRuntimeReplica` is still a placeholder runtime and does not execute inference
- the app does not yet route `/v1/chat/completions` or `/v1/embeddings` into the runtime layer

## What Was Verified

### Dependency Installation
- `uv sync` works after adding the Tsinghua mirror
- `uv sync --extra dev` works

### Automated Tests
Command:
```bash
uv run pytest -q
```

Current result:
```text
13 passed in 0.33s
```

The tests cover:
- health endpoints
- catalog endpoints
- `/v1/models`
- OpenAI-style chat/embedding stub error semantics
- Serve plan generation
- fake Ray Serve binding generation
- behavior when `ray` is not installed

### Current Environment Boundaries
Verified on current macOS environment:
- Python package layout
- FastAPI app startup via `TestClient`
- config loading
- model registry behavior
- route behavior
- non-Ray runtime skeleton behavior

Not verified yet in current environment:
- real `ray[serve]` runtime startup
- GPU resource scheduling
- `vLLM` integration
- CUDA/NPU device visibility behavior
- actual inference request execution

## What Should Be Done Next

### Highest Priority
1. Wire OpenAI routes into a runtime dispatch layer
   - `/v1/chat/completions` should stop returning `501` once the runtime dispatch path exists
   - `/v1/embeddings` should do the same
   - Suggested starting point: introduce a runtime dispatcher module or extend `ModelRuntimeReplica` plus a lightweight gateway-side dispatch abstraction

2. Connect backend abstraction to request execution
   - Extend `InferenceBackend` beyond `build_runtime_spec()`
   - Add explicit methods for chat and embedding execution
   - Keep real vLLM execution behind the backend boundary

3. Decide the first practical integration shape for Ray Serve
   - Option A: gateway dispatches to Serve handles
   - Option B: Serve hosts the public HTTP ingress directly
   - Current code is closer to keeping FastAPI as the northbound gateway and using Serve internally
   - That direction should be preserved unless intentionally changed

### After That
4. Introduce request/response adapters between OpenAI schemas and backend calls
5. Flesh out admission control using real load signals
6. Add runtime-aware model status and cluster load aggregation
7. Add `ray[serve]` extra installation and local non-GPU smoke checks when desired
8. Prepare Ubuntu + CUDA validation plan for real integration

## Recommended Starting Commands For Next Session

Install dependencies:
```bash
uv sync --extra dev
```

Run tests:
```bash
uv run pytest -q
```

If a local app run is needed later:
```bash
uv run uvicorn infer_nexus.main:app --reload
```

## Notes For The Next Session
- The current codebase is intentionally protocol-first and runtime-light
- Do not skip the backend abstraction and route handlers directly into vLLM-specific code
- Keep `ARCHITECTURE.md` as the design source of truth
- Keep `AGENT.md` aligned if implementation rules change
- Do not introduce per-model static device pinning; the resource-pool model is already settled
