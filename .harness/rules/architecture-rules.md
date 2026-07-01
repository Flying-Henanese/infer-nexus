# Architecture Rules

Read this when a task changes model lifecycle, Ray Serve runtime boundaries, accelerator placement, admission, scaling, or backend abstractions.

## Non-Goals

Unless the user explicitly changes scope, do not add:
- Web UI
- dynamic model registration in Phase 1
- tenant-specific hardware isolation
- per-user or per-team scheduling partitions
- multi-backend coexistence
- cross-cluster scheduling
- static per-deployment device pinning

## Core Implementation Rules

### 1. Pre-registered models only
Phase 1 supports only pre-registered models loaded from configuration.

Implications:
- model inventory must live in config files, not hardcoded Python lists
- model registration happens at startup or through a controlled reconcile path
- do not add APIs that let users arbitrarily register models at runtime unless explicitly requested

### 2. One deployment per model
Use one Ray Serve deployment per model in Phase 1.

Implications:
- each model has a distinct deployment identity
- scaling, health, and metrics should remain model-specific
- avoid grouping unrelated models into a single deployment abstraction

### 3. Single shared accelerator pool
The platform uses one shared accelerator pool.

Implications:
- the pool boundary is defined at the Ray runtime boundary, not inside individual model configs
- model configs describe resource demand per replica, not fixed device IDs
- do not implement static `CUDA_VISIBLE_DEVICES` assignment per model or deployment

### 4. Platform-level pool boundary
For CUDA, treat `CUDA_VISIBLE_DEVICES` as a mechanism for defining the overall `infer-nexus` pool visible to Ray, not as a mechanism for manual model pinning.

For Ascend, follow the same principle with the relevant accelerator visibility controls.

Implementation rule:
- pool boundary belongs to deployment or ops startup configuration
- per-replica resource demand belongs to model configuration and Ray Serve deployment settings

### 5. OpenAI-compatible northbound APIs
Northbound APIs must prefer OpenAI-compatible request and response shapes for model inference.

Required first:
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`

Do not let OpenAI compatibility distort the internal architecture. Internal abstractions should still distinguish task types such as `chat`, `embedding`, `rerank`, and future `vlm`.

### 6. Native platform APIs are separate
Platform-native APIs must remain distinct from OpenAI-compatible inference APIs.

Keep separate routes and internal handlers for:
- model discovery
- model status
- cluster load
- cluster capacity
- liveness and readiness

### 7. Admission control is required
Do not implement a gateway that blindly accepts every request.

At minimum, request routing must be able to reject based on:
- model readiness
- model overload
- cluster capacity pressure
- invalid task or model combinations

### 8. Autoscaling should use inference-aware signals
Prefer scaling logic based on:
- queue length
- TTFT
- p95 latency
- inflight request count

Do not design scaling purely around raw CPU or GPU utilization.

### 9. Use Ray Serve as runtime, not as the full control plane
Ray Serve handles:
- deployment lifecycle
- replica routing
- autoscaling support
- runtime metrics exposure

`infer-nexus` itself must provide:
- model catalog
- request routing
- admission control
- platform-native APIs
- runtime state aggregation

### 10. Preserve backend substitution path
Even though Phase 1 uses `vLLM`, do not spread vLLM-specific assumptions across unrelated modules.

When changing src/infer_nexus/backends/, VLLMBackend, vllm_native/,
StrictNativeVLLMExecutor, LocalBestEffortVLLMExecutor, or compat_mode
execution semantics, read docs/inference_backend_design.md first.

Implementation rule:
- keep a thin backend abstraction layer
- isolate engine-specific startup and request adaptation logic
- keep the future path open for `vllm-ascend`
