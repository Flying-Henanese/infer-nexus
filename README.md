# infer-nexus

Shared inference service factory for internal development and testing.

## Current Runtime Shape

- Runtime mode: `Gateway + per-model backend dispatch`
- Gateway: `FastAPI` (`/v1/*` OpenAI-compatible + `/api/*` platform APIs)
- Default gateway bind: `0.0.0.0:8000`
- Supported task APIs: `chat`, `embeddings`, `rerank`
- Backends:
  - `vllm`: local Ray Serve + `LLM.chat/embed/score` path
  - `vllm_openai_proxy`: upstream OpenAI-compatible proxy path (recommended for compatibility-sensitive models)

## Quick Start

For the current smallest runnable deployment shape:

- [MINIMAL_STARTUP.md](./MINIMAL_STARTUP.md)

For first-time Ubuntu + CUDA bring-up:

- [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md)

## Documentation Map

- [ARCHITECTURE.md](./ARCHITECTURE.md): system design, boundaries, and phase goals
- [OPENAI_PROXY_REFACTOR_PLAN.md](./OPENAI_PROXY_REFACTOR_PLAN.md): proxy-first refactor plan and rollout contract
- [AGENTS.md](./AGENTS.md): implementation rules and constraints

## Current Status

- Proxy backend (`vllm_openai_proxy`) code path is now wired for:
  - `POST /v1/chat/completions`
  - `POST /v1/embeddings`
  - `POST /v1/rerank`
- Request behavior for proxy models:
  - Keep payload as-is except `model` remap to configured upstream model name.
  - Return upstream payload/status as-is.
  - `stream=true` uses SSE passthrough.
  - The gateway targets one configured `upstream_base_url` per proxy model; if that endpoint is Ray Serve-backed, Ray Serve owns replica pooling, health, and autoscaling behind it.
- Validation in this environment is currently limited:
  - Local `pytest` execution is blocked by existing environment/lockfile issues.
  - Syntax-level validation was completed via `python3 -m compileall src tests`.

## Branch Delta (`dev` vs `master`)

Baseline:
- `master`: `c5195be` (`Preserve engine_kwargs for vLLM backend`)
- `dev`: `f423744`
- Diff scope: `16 files changed, 375 insertions(+), 29 deletions(-)`

What changed in the refactored branch:
- Model loading path was generalized from local-path-only to `model reference` mode:
  - `model_path` can be empty if `model_loading_config.model_id` is provided.
  - Added `require_local_artifacts` to support remote model IDs without local artifact checks.
  - Runtime now resolves a model reference string (local path or remote model ID) before backend init.
- Catalog schema and routing semantics were extended toward OpenAI/vLLM style metadata:
  - Added `model_loading_config`, `deployment_config`, `served_model_name`, `engine_kwargs` fields.
  - `served_model_name` is now routable in registry and used by `/v1/models` and response payload fallback.
- Ray Serve deployment config became override-friendly:
  - `deployment_config.autoscaling_config` and `deployment_config.ray_actor_options` can override defaults.
- vLLM backend compatibility and request fidelity were improved:
  - Supports passing loading `revision` into `LLM(...)`.
  - Chat sampling params now preserve more OpenAI-compatible fields (`presence_penalty`, `frequency_penalty`, `stop`, `seed`, `n`, `top_logprobs`, and selective `extra_body` passthrough such as `top_k`, `min_p`).
  - Multimodal image `data:` URLs are normalized before sending to vLLM (whitespace, URL-safe base64, padding).
  - Message serialization now carries `tool_call_id` when present.
- Runtime lifecycle scripts were hardened:
  - `start_minimal.sh` records Ray ownership/state in `.infer-nexus/ray_state.env`.
  - `stop_minimal.sh` reads ownership state, attempts `serve.shutdown()` first, then conditionally stops Ray and cleans stray Ray/vLLM processes.
- Default model config was tuned for a smaller/minimal deployment footprint:
  - `config/models.yaml` shifts `mineru` toward a local mounted path and lower GPU/replica defaults.

Validation coverage added by tests:
- Added/updated tests for remote model reference resolution, deployment override behavior,
  runtime spec metadata preservation, richer sampling params, and multimodal data URL normalization.

## Docker Compose Runtime Split

The first-stage Compose deployment keeps the current infer-nexus runtime architecture and only separates process lifecycles. The runtime image uses CUDA devel/runtime base images in a two-stage Dockerfile so dependency resolution stays in the builder stage while all services share the same CUDA-capable final image:

- `ray-head` runs the Ray control plane.
- `ray-worker` joins the Ray cluster and hosts Ray Serve replicas plus replica-local vLLM runtimes.
- `serve-deployer` runs `scripts/run_serve_runtime.py` once, submits Ray Serve applications, waits for readiness, and exits.
- `gateway` runs the FastAPI/Uvicorn API entrypoint and connects to Ray through `INFER_NEXUS_RAY_ADDRESS`.

Basic startup flow:

```bash
docker compose build
docker compose up
```

Useful deployment variables:

```bash
BUILDER_BASE_IMAGE=nvidia/cuda:12.2.0-devel-ubuntu22.04
RUNTIME_BASE_IMAGE=nvidia/cuda:12.2.0-runtime-ubuntu22.04
PYTHON_VERSION=3.11
APP_UID=10001
APP_GID=10001
COMPOSE_GPU_REQUEST=all
INFER_NEXUS_RAY_ADDRESS=ray-head:6379
MODEL_STORE_HOST_PATH=/data/models
GATEWAY_WORKERS=2
RAY_WORKER_NUM_GPUS=1
RAY_WORKER_RESOURCES={"NPU": 1}
ASCEND_RT_VISIBLE_DEVICES=0,1
```

Containers run as the `infer-nexus` non-root user created in the image. Make sure mounted host paths such as `MODEL_STORE_HOST_PATH` are readable, and writable if runtime artifact downloads are expected, by UID/GID `10001:10001` or the overridden `APP_UID`/`APP_GID`. CUDA Compose runs require NVIDIA Container Toolkit on the host so Docker can mount the driver into `ray-worker`. Set only the accelerator variables that match the target runtime. Compose defines the visible accelerator pool; per-model replica resource requirements still belong in `config/models.yaml`.

After startup, validate from the host:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models
```
