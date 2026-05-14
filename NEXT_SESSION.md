# Next Session Handoff

## Current State

`infer-nexus` is no longer just a skeleton. It now has:
- a working FastAPI northbound gateway
- a local model store abstraction
- offline model download support
- a runtime dispatcher/executor split
- per-model Ray Serve deployment assembly
- backend adapter wiring for `chat`, `embedding`, and `rerank`
- a minimal dual-process startup path for Ubuntu + CUDA validation

The current implementation still defaults to `stub` runtime behavior on this machine, but the code structure is already aligned to the intended real runtime shape.

## What Was Completed

### Documentation
- Maintained [ARCHITECTURE.md](./ARCHITECTURE.md) as the design source of truth
- Maintained [AGENT.md](./AGENT.md) as the implementation guardrail file
- Added [MINIMAL_STARTUP.md](./MINIMAL_STARTUP.md) for the first runnable Ubuntu + CUDA shape
- Added [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md) for first real-machine bring-up
- Updated [README.md](./README.md) to point to the startup and deployment docs
- Updated `ARCHITECTURE.md` to document future convergence toward a single operator-facing startup flow while preserving one external API entrypoint

### Repository Structure
- Python source is under `src/infer_nexus/`
- Tests are under `tests/`
- Operational scripts are under `scripts/`
- `uv` is configured to use the Tsinghua mirror in [pyproject.toml](./pyproject.toml)

### Configuration
- Settings file: [config/settings.yaml](./config/settings.yaml)
- Model catalog: [config/models.yaml](./config/models.yaml)
- Current registered models:
  - `qwen3-chat` -> `qwen3-32b-instruct`
  - `bge-embedding` -> `bge-large-zh-v1.5`
  - `bge-rerank` -> `bge-reranker-v2-m3`
- `model_path` now has local-path semantics for runtime loading
- `model_store.root_dir` is the local model artifact root
- `model_store.huggingface_endpoint` is `https://hf-mirror.com`

### Local Model Store and Offline Download
Added:
- [src/infer_nexus/model_store.py](./src/infer_nexus/model_store.py)
- [src/infer_nexus/artifacts/download.py](./src/infer_nexus/artifacts/download.py)
- [scripts/download_model.py](./scripts/download_model.py)

Current behavior:
- runtime only loads from local model paths
- runtime does not download model weights during startup or request handling
- the download script downloads into the local model store and prints a suggested registration snippet
- the download script does not modify `config/models.yaml`
- missing local model artifacts now surface as explicit errors

### FastAPI Gateway
Entrypoint:
- [src/infer_nexus/main.py](./src/infer_nexus/main.py)

Routes:
- [src/infer_nexus/api/health_routes.py](./src/infer_nexus/api/health_routes.py)
- [src/infer_nexus/api/platform_routes.py](./src/infer_nexus/api/platform_routes.py)
- [src/infer_nexus/api/openai_routes.py](./src/infer_nexus/api/openai_routes.py)
- [src/infer_nexus/api/deps.py](./src/infer_nexus/api/deps.py)

Current API surface:
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
- `POST /v1/rerank`
- `POST /rerank`

Important current behavior:
- unknown models return `404`
- task mismatch returns `400 unsupported_task_type`
- missing local artifacts return `503 model_artifact_missing`
- phase-1 unsupported request features return `400`, for example `stream=true`
- serve-mode handle failures return `501 runtime_not_connected`
- platform lookup/status endpoints now return `404` instead of leaking into `500`

### Runtime Layer
Implemented:
- [src/infer_nexus/runtime/types.py](./src/infer_nexus/runtime/types.py)
- [src/infer_nexus/runtime/dispatcher.py](./src/infer_nexus/runtime/dispatcher.py)
- [src/infer_nexus/runtime/executor.py](./src/infer_nexus/runtime/executor.py)
- [src/infer_nexus/runtime/handles.py](./src/infer_nexus/runtime/handles.py)
- [src/infer_nexus/runtime/deployments.py](./src/infer_nexus/runtime/deployments.py)
- [src/infer_nexus/runtime/serve_app.py](./src/infer_nexus/runtime/serve_app.py)

Current structure:
- route -> dispatcher -> executor -> replica -> backend adapter
- `RuntimeExecutor` supports:
  - `stub` mode: local replica invocation
  - `serve` mode: Serve deployment handle invocation
- per-model Serve deployment specs are generated from the catalog
- Serve application assembly can build one composed Serve app containing all model deployments plus a root deployment

### Backend Layer
Implemented:
- [src/infer_nexus/backends/base.py](./src/infer_nexus/backends/base.py)
- [src/infer_nexus/backends/vllm.py](./src/infer_nexus/backends/vllm.py)

Current behavior:
- `VLLMBackend` is stateful and has lifecycle hooks:
  - `startup()`
  - `shutdown()`
- `backend_init_mode=stub` keeps runtime safe on the current machine
- `backend_init_mode=real` is prepared for real vLLM engine initialization on Ubuntu + CUDA
- supported task mappings are explicitly enforced:
  - `chat -> generate`
  - `embedding -> embed`
  - `rerank -> score`
- config/runtime mismatches fail fast through `BackendConfigurationError`

Task-specific behavior:
- `chat` is designed around `LLM.chat()` semantics
- `embedding` is designed around `LLM.embed()` semantics
- `rerank` is designed around `LLM.score()` semantics

Current Phase 1 request limits:
- chat only supports text messages
- chat rejects `stream=true`
- multimodal message content is rejected
- `embedding` rejects empty input lists
- `rerank` rejects empty document lists

### Minimal Startup Path
Added:
- [scripts/run_gateway.py](./scripts/run_gateway.py)
- [scripts/run_serve_runtime.py](./scripts/run_serve_runtime.py)
- [MINIMAL_STARTUP.md](./MINIMAL_STARTUP.md)

Current minimal runtime shape:
1. Ray Serve runtime process owns model deployments
2. FastAPI gateway process exposes the northbound APIs

This is deliberate. It keeps runtime debugging and northbound API debugging separate for the first real deployment.

## What Was Verified

### Dependency Installation
- `uv sync` works with the configured mainland China mirror
- `uv sync --extra dev` works

### Automated Tests
Command:
```bash
uv run pytest -q
```

Current result:
```text
61 passed in 0.49s
```

Coverage now includes:
- health and readiness endpoints
- catalog and platform endpoints
- model status behavior
- OpenAI model discovery
- chat request success/error paths
- embedding request success/error paths
- rerank request success/error paths
- unknown-model behavior
- missing-artifact behavior
- admission rejection behavior
- serve-handle-not-connected behavior
- local model store behavior
- backend request validation behavior
- runtime dispatch in stub mode
- runtime dispatch in serve mode with fake handles
- Serve app/build-plan assembly
- startup script behavior for both gateway and serve runtime
- app lifespan wiring for `stub` and `serve` executor modes

### Current Environment Boundaries
Verified on current macOS environment:
- Python package layout
- config loading
- model registry behavior
- local model store behavior
- FastAPI app startup via `TestClient`
- route behavior
- runtime dispatch structure
- backend adapter wiring in stub mode
- startup script behavior through unit tests

Not verified yet in current environment:
- real `ray[serve]` runtime startup
- real `vLLM` installation on this machine
- GPU resource scheduling
- CUDA device visibility behavior
- actual inference execution with a real vLLM engine

Important note:
- Installing `vllm` on the current macOS arm64 machine did not work with the current dependency path because it resolved NVIDIA-specific packages without compatible wheels.
- Real vLLM validation should happen on the target `Ubuntu + CUDA` machine.

## Recommended Starting Documents For Next Session

Read these first:
1. [ARCHITECTURE.md](./ARCHITECTURE.md)
2. [AGENT.md](./AGENT.md)
3. [MINIMAL_STARTUP.md](./MINIMAL_STARTUP.md)
4. [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md)
5. [NEXT_SESSION.md](./NEXT_SESSION.md)

## What Should Be Done Next

### Highest Priority
1. Perform the first real Ubuntu + CUDA bring-up
   - Follow [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md)
   - Switch `config/settings.yaml` to:
     - `runtime.execution_mode: serve`
     - `runtime.backend_init_mode: real`
   - Start Ray with an explicit GPU pool boundary
   - Deploy the Serve runtime
   - Start the gateway
   - Verify chat, embedding, and rerank end-to-end

2. Validate real vLLM backend behavior in the target environment
   - Confirm `VLLMBackend.startup()` can initialize `vllm.LLM(...)`
   - Confirm `chat` works through the `LLM.chat()` path
   - Confirm `embedding` works through `LLM.embed()`
   - Confirm `rerank` works through `LLM.score()`
   - Inspect any model-specific startup needs such as `hf_overrides`, score templates, or task-specific runner settings

3. Fix real-environment issues found during first deployment
   - resource sizing mismatches
   - model artifact path mismatches
   - Serve handle connectivity issues
   - backend initialization failures
   - task-mode compatibility issues

### Next Architectural Step After First Real Bring-Up
4. Start converging toward a single operator-facing startup flow
   - Keep one external API entrypoint
   - Keep Ray Serve as the owner of model replicas
   - Move toward a Serve ingress deployment for the gateway instead of a permanently separate Uvicorn process

This does **not** mean “clients call per-model Serve ports”.
The intended direction is:
- one external API entrypoint
- gateway logic hosted as Serve ingress
- model deployments still separate behind it

### After That
5. Add real runtime-aware load and capacity inspection
   - replace stub cluster capacity
   - aggregate real runtime/deployment state

6. Flesh out admission control with real load signals
   - queue length
   - TTFT
   - latency-based rejection hooks

7. Revisit Phase 1 constraints after the first real deployment
   - streaming
   - richer rerank behavior
   - controlled config reload / reconcile

## Recommended Commands For Next Session

Install dependencies:
```bash
uv sync --extra dev
```

Run tests:
```bash
uv run pytest -q
```

If preparing the target runtime environment:
```bash
uv sync --extra serve --extra vllm
```

Start gateway locally:
```bash
uv run python scripts/run_gateway.py
```

Start serve runtime on target machine:
```bash
uv run python scripts/run_serve_runtime.py --ray-address auto
```

## Notes For The Next Session
- The project is now beyond protocol-only scaffolding. Avoid regressing it back into route-level ad hoc logic.
- Keep the backend abstraction intact. Do not bypass `dispatcher -> executor -> replica -> backend`.
- Keep `model_path` semantics local-path-oriented.
- Keep runtime model loading local-only; do not introduce online downloads into startup or request handling.
- Keep the shared resource-pool model; do not introduce per-model static device pinning.
- The current dual-process startup shape is temporary but intentional. It exists to simplify the first real deployment, not as the desired long-term UX.
