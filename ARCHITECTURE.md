# infer-nexus Architecture

## 1. Overview

`infer-nexus` is a shared inference service factory for internal development and testing environments. It is designed to replace the current pattern where each user deploys and maintains their own model inference services on shared GPU servers.

The project uses:
- `Python` as the implementation language
- `uv` for dependency and environment management
- `Ray Serve` for deployment lifecycle, routing, and autoscaling support
- `vLLM` as the default inference backend
- OpenAI-compatible northbound APIs for client compatibility

Primary goals:
- Share GPU resources across teams and users
- Expose a single inference endpoint instead of many scattered endpoints
- Pre-register and centrally manage commonly used models
- Support elastic scaling for high-demand models
- Provide platform-native APIs for model discovery and cluster load inspection
- Keep the system simple enough for internal developer adoption

Out of scope for Phase 1:
- Web UI
- Fine-grained user or team resource isolation
- Fully dynamic model registration
- Multi-backend coexistence
- Cross-cluster scheduling

## 2. Design Principles

1. Pre-registered models only
   - All supported models are declared in configuration and managed by the platform.
   - This keeps resource usage predictable and operational behavior controlled.

2. Single northbound endpoint
   - Clients use a single service entrypoint.
   - Internally, requests are routed to model-specific Ray Serve deployments.

3. OpenAI-compatible first
   - Existing internal applications should require little or no code change.
   - Platform-specific capabilities are exposed through separate native APIs.

4. Shared GPU pool
   - All model services use the same GPU pool.
   - No user-level hardware reservation or hard partitioning in Phase 1.

5. Warm replicas for low-frequency large models
   - Low-frequency but expensive models keep a minimum replica count.
   - This avoids slow cold starts in developer workflows.

6. Clear separation of control plane and data plane responsibilities
   - Ray Serve is the serving runtime.
   - `infer-nexus` provides catalog, routing, admission control, and platform APIs.

## 3. High-Level Architecture

```text
Client
  -> infer-nexus API Gateway
      -> OpenAI-Compatible API Layer
      -> Native Platform API Layer
      -> Auth / Admission / Routing
          -> Model Catalog
          -> Load Inspector
          -> Scaling Policy
          -> Ray Serve Deployments
              -> vLLM Runtime per Model
                  -> LLM / Embedding / Rerank / VLM
```

### Responsibilities by layer

#### API Gateway
- Exposes one HTTP entrypoint
- Handles request authentication
- Separates OpenAI-compatible APIs from platform-native APIs
- Applies request admission checks before dispatch

#### Model Catalog
- Stores pre-registered model metadata
- Maps logical model names to runtime deployments
- Exposes discovery information to users and routing logic

#### Routing Layer
- Resolves the requested model
- Selects the correct task path based on model type
- Dispatches to the matching Ray Serve deployment

#### Load Inspector and Admission Control
- Aggregates runtime health and load indicators
- Decides whether requests should be admitted or rejected early
- Supports cluster and model load inspection APIs

#### Scaling Policy Layer
- Uses inference-related metrics such as queue length, TTFT, and latency
- Adjusts deployment replica counts within configured limits

#### Ray Serve Runtime
- Owns deployment lifecycle and replica management
- Handles internal routing to replicas
- Exposes runtime and deployment metrics through Prometheus-compatible endpoints

#### vLLM Backend
- Runs the actual model inference engines
- Provides OpenAI-like behavior where possible
- Can later be replaced by `vllm-ascend` without redesigning the entire platform

## 4. Core Decisions

### 4.1 Model registration mode
Phase 1 uses pre-registration only.

Implications:
- Models are defined in a configuration file
- Service startup loads model definitions into the catalog
- A reconciliation step ensures Ray Serve deployments match the declared state

Future-compatible extension:
- The architecture may later add controlled dynamic registration APIs
- This is intentionally not part of the initial implementation

### 4.2 Deployment granularity
Phase 1 uses one Ray Serve deployment per model.

Rationale:
- Simpler autoscaling boundaries
- Clear ownership of metrics and health
- Easier troubleshooting
- Easier mapping from `/v1/models` and catalog metadata to runtime state

### 4.3 GPU resource model
All models share one GPU pool.

Rationale:
- Matches the current internal environment
- Keeps scheduling and operations simple
- Maximizes hardware sharing in development and testing

Constraint:
- Resource admission must avoid accepting traffic the cluster cannot serve reasonably

### 4.4 Resource pool boundary vs deployment resource requests
The platform must distinguish between two separate concerns:

1. Resource pool boundary
   - Defines how much accelerator capacity is assigned to the entire `infer-nexus` platform
   - This is set at the Ray node or cluster process boundary
   - Example: make only 4 A100 GPUs visible to the Ray runtime used by `infer-nexus`

2. Deployment resource requests
   - Defines how much of that shared pool each deployment replica consumes
   - This is expressed through Ray or Ray Serve resource requirements such as GPU count per replica
   - This does not bind a deployment to specific device IDs ahead of time

Phase 1 should implement platform-level pooling, not static per-deployment device pinning.

#### Recommended CUDA pattern
For CUDA environments, the recommended pattern is:
- limit the GPUs visible to the Ray node that runs `infer-nexus`
- start Ray with a matching GPU resource count
- let Ray Serve schedule replicas inside that bounded pool

Conceptually:
- `CUDA_VISIBLE_DEVICES=0,1,2,3` defines the total GPU pool visible to `infer-nexus`
- Ray then treats that pool as 4 schedulable GPU resources
- a deployment with `num_gpus=1` consumes one unit from the pool
- a deployment with `num_gpus=4` consumes the full pool

This means `CUDA_VISIBLE_DEVICES` is still useful, but only to define the platform resource boundary. It should not be used as a per-model or per-deployment static binding mechanism.

#### Recommended Ascend pattern
For Ascend environments, the same principle applies:
- define the set of visible accelerator devices at the Ray runtime boundary
- let the serving layer schedule replicas within that bounded pool

If additional backend-specific environment handling is required for `vllm-ascend`, that logic should remain an implementation detail of the backend integration layer rather than a manual per-deployment operational step.

#### Design rule
The architecture should follow these rules:
- the total hardware budget for `infer-nexus` is defined at the platform or Ray node boundary
- deployments declare only resource demand, not concrete device IDs
- Ray Serve performs scheduling within the visible resource pool
- model configuration describes resource requirements per replica, not fixed device assignments

This separation is necessary to preserve shared-pool scheduling and avoid falling back to manual device partitioning.

### 4.5 Serving backend
Phase 1 assumes `vLLM` as the only backend.


Rationale:
- Good compatibility and current fit for LLM-style workloads
- Forward path exists to `vllm-ascend`

Implementation note:
- The codebase should still define a thin backend abstraction to avoid coupling all logic directly to vLLM internals

### 4.6 No Web UI
Phase 1 is API-only.

Rationale:
- Internal developer usage does not require a UI initially
- The platform APIs should provide all information needed by tools or scripts

## 5. Supported Model Types

Phase 1 should support these model task categories:
- `chat`
- `embedding`
- `rerank`
- `vlm` as a reserved extension point, even if not fully implemented first

These task categories should remain explicit in the internal model schema because:
- Request/response formats differ
- Scaling signals differ slightly by task type
- Validation and routing differ by task type

## 6. Northbound API Design

## 6.1 OpenAI-compatible APIs

Required in Phase 1:
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`

Recommended later:
- `POST /v1/responses`
- `POST /v1/rerank` or a clearly documented internal-compatible rerank endpoint

### Behavior expectations
- The `model` field is resolved through the catalog
- The client sees stable logical model names or aliases
- Errors should follow consistent JSON error semantics
- Streaming support for chat completions should be considered part of the design, even if implemented after the first non-streaming version

## 6.2 Native platform APIs

Required in Phase 1:
- `GET /api/catalog/models`
- `GET /api/catalog/models/{model_name}`
- `GET /api/models/{model_name}/status`
- `GET /api/cluster/load`
- `GET /api/cluster/capacity`
- `GET /healthz`
- `GET /readyz`

### API intent

#### `GET /api/catalog/models`
Returns the registered model list with metadata such as:
- logical name
- task type
- backend
- capabilities
- current status
- deployment name
- min/max replicas

#### `GET /api/catalog/models/{model_name}`
Returns a single model's configuration and current runtime view.

#### `GET /api/models/{model_name}/status`
Returns operational status such as:
- readiness
- current replicas
- inflight requests
- queue length
- recent TTFT and latency summaries
- admission state

#### `GET /api/cluster/load`
Returns a summarized runtime load view for the shared inference factory.

Example payload shape:
- total GPUs
- allocated GPUs
- free GPUs estimate
- active models
- pending scale actions
- rejection pressure state

#### `GET /api/cluster/capacity`
Returns a more static or planning-oriented view of cluster resources and configured model demands.

#### `GET /healthz`
Liveness probe for the API process.

#### `GET /readyz`
Readiness probe indicating the gateway, catalog, and runtime integrations are ready.

## 7. Request Lifecycle

### 7.1 Chat or embedding request flow

```text
Client request
  -> API validation
  -> authentication
  -> model resolution via catalog
  -> admission control check
  -> route to model-specific Ray Serve deployment
  -> vLLM inference
  -> response adaptation
  -> client response
```

### 7.2 Model discovery request flow

```text
Client request
  -> API layer
  -> catalog query
  -> optional runtime state enrichment
  -> response
```

### 7.3 Load inspection request flow

```text
Client request
  -> API layer
  -> metrics/state aggregation
  -> cluster summary computation
  -> response
```

## 8. Model Catalog Design

The catalog is the control-plane source of truth for supported models.

Each model entry should include at least:
- `name`: stable internal identifier
- `alias`: external logical name used by clients
- `task`: `chat`, `embedding`, `rerank`, or `vlm`
- `backend`: initially `vllm`
- `model_path`: Hugging Face ID or local model path
- `deployment_name`: Ray Serve deployment identifier
- `dtype`
- `tensor_parallel_size`
- `max_model_len`
- `gpu_per_replica`
- `min_replicas`
- `max_replicas`
- `status`
- `capabilities`: optional list such as `streaming`, `vision`, `tool_calling`
- `labels`: arbitrary internal tags

### Catalog responsibilities
- Load configuration at startup
- Validate schema correctness
- Resolve aliases to canonical model definitions
- Expose discovery metadata to `/v1/models` and native APIs
- Provide deployment mapping for routers and controllers

## 9. Configuration Model

Configuration should be declarative and file-based in Phase 1.

Recommended initial files:
- `config/settings.toml` or `config/settings.yaml`
- `config/models.yaml`

## 9.1 Local Model Store and Offline Registration

Phase 1 should use a local model store for all model weights. Runtime services should load only from local paths and should not download model weights as part of normal request handling or service startup.

### Design rules
- all model weights live under a configured local model store root
- runtime loading uses only locally available model files
- model download is an offline administrative operation, not part of the serving control plane
- model registration remains declarative and file-based
- the download workflow must not automatically mutate the main model registry configuration

### Operational workflow
The intended Phase 1 workflow is:
1. an administrator runs a standalone download script
2. the script downloads a model into the local model store
3. the script prints a suggested model configuration snippet
4. the administrator reviews that output and manually adds the model entry to `config/models.yaml`
5. the service starts or reloads using only models already present in configuration and on local disk

This keeps the serving path simple and avoids coupling model artifact acquisition to runtime lifecycle operations.

### Local model store responsibilities
The local model store concept in Phase 1 is intentionally narrow:
- define a root directory for model artifacts
- provide a stable location for runtime model loading
- support reuse of previously downloaded weights across restarts and replicas

The local model store is not a full artifact-management subsystem in Phase 1. It should not introduce dynamic registration, automatic cleanup policies, or online model acquisition during request handling.

### Download script behavior
The standalone download script should:
- download model weights into the configured local model store
- support Hugging Face downloads through `hf-mirror.com` under mainland China network conditions
- optionally support additional sources later, such as ModelScope
- perform lightweight validation that expected model files exist
- print a suggested configuration block for manual registration

The script should not directly edit `config/models.yaml`. Manual review and registration is preferred to keep configuration changes explicit and auditable.

### Configuration implications
Platform settings should include a local model store root, for example through `config/settings.yaml`. Model definitions in `config/models.yaml` should resolve to local model paths rather than relying on remote repository identifiers at runtime.

A relative path under the configured model store root is preferred over a machine-specific absolute path. This keeps configuration more portable across environments.

### Runtime implications
At runtime, `infer-nexus` should:
- load only models declared in `config/models.yaml`
- resolve each model to a local filesystem path
- fail clearly if a configured model is missing from local storage
- avoid implicit remote downloads when starting deployments or serving requests

Example model declaration:

```yaml
models:
  - name: qwen3-32b-instruct
    alias: qwen3-chat
    task: chat
    backend: vllm
    model_path: Qwen/Qwen3-32B-Instruct
    dtype: bfloat16
    tensor_parallel_size: 4
    max_model_len: 32768
    gpu_per_replica: 4
    min_replicas: 1
    max_replicas: 4
    capabilities:
      - streaming
    labels:
      - internal
      - llm

  - name: bge-large-zh-v15
    alias: bge-embedding
    task: embedding
    backend: vllm
    model_path: BAAI/bge-large-zh-v1.5
    dtype: float16
    tensor_parallel_size: 1
    max_model_len: 8192
    gpu_per_replica: 1
    min_replicas: 1
    max_replicas: 2
    labels:
      - internal
      - embedding
```

### Why configuration must stay out of code
- Easier model onboarding
- Easier review and change control
- Cleaner separation between platform logic and platform inventory
- Better compatibility with future reconciliation or reload workflows

## 10. Admission Control

Admission control is required even without user-level resource isolation.

Its purpose is to reject requests early when service quality would otherwise collapse.

### Admission inputs
Per-model signals:
- deployment readiness
- current replica count
- inflight requests
- request queue length
- recent TTFT
- recent p95 latency
- deployment at max replicas or not

Cluster-level signals:
- available GPU capacity estimate
- pending scale-up actions
- recent rejection pressure

### Admission outcomes
- admit request
- reject with `429 Too Many Requests`
- reject with `503 Service Unavailable`

### Error scenarios to encode clearly
- unknown model
- model not ready
- model overloaded
- cluster capacity exhausted
- invalid request for model task type

This is preferable to accepting every request and failing later with long timeouts.

## 11. Autoscaling Strategy

Phase 1 scaling should be driven by inference-oriented metrics, not just low-level infrastructure metrics.

### Primary scaling signals
- `queue_length`
- `ttft`
- `p95_latency`
- `inflight_requests`

### Secondary supporting signals
- `tokens_per_second`
- `gpu_utilization`
- `gpu_memory_utilization`

### Why these metrics
- `queue_length` directly reflects contention
- `ttft` is the most meaningful user-facing stress indicator for LLM workloads
- `p95_latency` protects overall request quality
- `inflight_requests` helps detect sustained pressure

### Suggested scale-up rules
Scale up when any of these conditions persist for a configured window:
- queue length above threshold
- TTFT above threshold
- p95 latency above threshold

Only scale up if:
- current replicas are below `max_replicas`
- cluster GPU capacity can satisfy the additional replica

### Suggested scale-down rules
Scale down when all of these conditions remain true for a configured cool-down window:
- queue length near zero
- TTFT within target
- low inflight request count
- replicas above `min_replicas`

### Task-type-specific tuning
The architecture should support different policies per task type.

Examples:
- chat: TTFT and latency weighted more heavily
- embedding: queue length and throughput weighted more heavily
- rerank: queue length and latency weighted more heavily

## 12. Runtime State and Reconciliation

The system should maintain a clear difference between:
- desired state from configuration
- observed runtime state from Ray Serve and metrics

A reconciliation loop should:
- ensure configured models exist in the catalog
- ensure deployments exist for registered models
- ensure deployment settings match current desired configuration
- surface mismatches and degraded status to platform APIs

This does not require full dynamic hot registration in Phase 1. It only requires that the platform can compare declared state with actual state.

## 13. Observability

Ray Serve's built-in Prometheus support should be used directly for runtime metrics. `infer-nexus` should add business-level and control-plane metrics on top.

### Recommended custom metrics
- `infer_nexus_requests_total`
- `infer_nexus_request_latency_seconds`
- `infer_nexus_ttft_seconds`
- `infer_nexus_model_requests_total`
- `infer_nexus_model_inflight_requests`
- `infer_nexus_model_queue_length`
- `infer_nexus_admission_rejections_total`
- `infer_nexus_model_status`

### Monitoring views the platform should support
- per-model request volume
- per-model TTFT and latency
- per-model scaling behavior
- per-model rejection counts
- cluster GPU allocation summary
- degraded model states

### Logging recommendations
Structured logs should include:
- request id
- model name
- task type
- admission decision
- deployment name
- latency summary
- failure reason when applicable

## 14. Error Model

The platform should define a consistent error format across OpenAI-compatible and native APIs where practical.

Recommended classes of failure:
- model not found
- model unavailable
- model overloaded
- cluster capacity exhausted
- invalid parameter
- backend runtime failure

OpenAI-compatible APIs should preserve expected HTTP semantics and JSON structure as closely as practical, while native APIs can expose more explicit platform-specific fields.

## 15. Security and Access Control

Phase 1 does not require complex tenant isolation, but basic access control is still necessary.

Minimum recommendations:
- API key authentication
- request identity in logs and metrics
- optional per-key rate limiting later

Rationale:
- Prevent accidental misuse
- Preserve auditability
- Keep a path open for future quota controls without redesigning the gateway

## 16. Proposed Codebase Layout

```text
src/
  infer_nexus/
    api/
      openai_routes.py
      platform_routes.py
      health_routes.py
    auth/
      api_keys.py
    backends/
      base.py
      vllm.py
    catalog/
      models.py
      registry.py
      loader.py
    control/
      admission.py
      load_inspector.py
      reconciler.py
      scaler.py
      policies.py
    runtime/
      deployments.py
      serve_app.py
    observability/
      metrics.py
      logging.py
    core/
      config.py
      enums.py
      errors.py
      schemas.py
    main.py
config/
  settings.yaml
  models.yaml
tests/
scripts/
```

### Layout rationale
- `api/` defines northbound HTTP behavior
- `catalog/` owns model metadata and discovery
- `control/` owns admission, scaling, and reconciliation logic
- `runtime/` owns Ray Serve integration points
- `backends/` isolates inference engine details
- `observability/` centralizes metrics and logging concerns

## 17. Phase 1 Implementation Scope

Phase 1 should implement only the minimum platform needed to replace ad hoc personal deployments.

### Must-have
- file-based pre-registered model catalog
- one deployment per model
- OpenAI-compatible chat and embeddings APIs
- native model discovery APIs
- native load inspection APIs
- basic admission control
- metric-driven autoscaling hooks
- Prometheus metrics integration
- API key authentication

### Nice-to-have but not required for Phase 1
- streaming chat completions if non-streaming lands first
- rerank endpoint if chat and embeddings are already stable
- config reload and deployment reconcile without full process restart

## 18. Future Extensions

The following should remain possible without architectural rework:
- controlled dynamic registration
- `vllm-ascend` backend substitution
- richer rerank and multimodal support
- stronger quota and rate-limit controls
- admin APIs for model lifecycle operations
- a thin internal UI built on top of native APIs
- convergence from the current two-process startup shape toward a single operator-facing startup flow while preserving one external API entrypoint

## 19. Recommended Next Step

The next implementation artifact should be a repository scaffold that matches this architecture but keeps Phase 1 scope disciplined.

Recommended immediate deliverables:
1. project skeleton with `uv`
2. config schema and model catalog loader
3. FastAPI gateway with placeholder OpenAI-compatible routes
4. Ray Serve integration skeleton
5. admission and metrics interfaces

This keeps the first coding step aligned with the intended runtime architecture instead of drifting into ad hoc service assembly.
