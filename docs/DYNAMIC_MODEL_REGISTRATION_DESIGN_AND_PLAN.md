# Dynamic Model Registration Design And Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development or superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Allow `infer-nexus` to add, validate, deploy, and serve a new model
without stopping the gateway process.

**Architecture:** Add a control-plane path that persists a new model definition,
refreshes the in-memory catalog safely, and lets a reconciler deploy the model
into Ray Serve asynchronously. Keep the public FastAPI gateway and existing
Serve handle execution path unchanged.

**Tech Stack:** FastAPI, Pydantic, YAML catalog files, Ray Serve, vLLM,
pytest.

---

## 1. Current Code Baseline

The current code already has the right runtime shape, but the catalog is loaded
only at startup:

- `src/infer_nexus/main.py`
  - Loads `config/models.yaml` once in `lifespan`.
  - Builds `ModelRegistry`, `LocalModelStore`, `ServeApplicationBuilder`,
    `RuntimeExecutor`, and `RuntimeDispatcher`.
  - Stores those objects in `app.state`.
- `src/infer_nexus/catalog/loader.py`
  - Loads one YAML catalog file and validates it as `ModelCatalogFile`.
- `src/infer_nexus/catalog/registry.py`
  - Builds an in-memory name and alias index.
  - Does not currently support add, update, delete, or atomic refresh.
- `src/infer_nexus/runtime/serve_app.py`
  - Converts all registered local `vllm` models into Ray Serve deployment
    bindings.
  - Can build runtime context for a model.
- `src/infer_nexus/runtime/dispatcher.py`
  - Resolves a request model into a runtime target using the registry and
    `ServeApplicationBuilder`.
- `src/infer_nexus/control/reconciler.py`
  - Exists, but is currently a skeleton.
- `scripts/run_serve_runtime.py`
  - Deploys all configured models into Ray Serve at startup.
  - Uses `serve.run(..., route_prefix=None, blocking=False)` once per model app.
  - Waits for Serve apps to become ready.

Important existing direction:

```text
Client
  -> FastAPI gateway
  -> RuntimeDispatcher / RuntimeExecutor
  -> Ray Serve deployment handle
  -> ModelRuntimeReplica
  -> backend
```

Dynamic registration must preserve this path. It should not introduce internal
HTTP calls between the gateway and Ray Serve.

## 2. Design Target

Adding a model should become an asynchronous control-plane operation:

```text
POST /api/catalog/models
  -> validate request body as ModelConfig
  -> validate artifact and runtime config
  -> persist model definition
  -> atomically refresh catalog snapshot
  -> create deployment operation
  -> optionally trigger Ray Serve reconciliation
  -> return operation id
```

The new model should move through explicit states:

```text
registered
  -> validating
  -> deploying
  -> warming_up
  -> serving
  -> failed
```

For the first implementation, use a smaller state machine:

```text
registered
  -> deploying
  -> serving
  -> failed
```

The API should not block until a large model finishes loading. It should return
an operation id and expose status endpoints.

Example response:

```json
{
  "operation_id": "deploy-Qwen3-32B-20260607T103000",
  "model": "Qwen3-32B",
  "status": "deploying"
}
```

## 3. Persistence Design

Do not write directly back into the checked-in `config/models.yaml` from the
API. Keep the static catalog and runtime-added models separate.

Recommended first format:

```text
config/models.yaml          static, reviewed catalog
config/models.d/*.yaml      runtime-added or operator-added model definitions
```

Rules:

- `config/models.yaml` remains the baseline catalog.
- Each runtime-added model is written as one file under `config/models.d/`.
- The file name should be derived from the canonical model name, for example
  `config/models.d/qwen3-32b.yaml`.
- Loading order is deterministic: base file first, then sorted `models.d/*.yaml`.
- Duplicate canonical names are rejected.
- Duplicate alias or `served_model_name` values are rejected.
- A failed deployment does not remove the persisted model definition; it marks
  deployment state as `failed` with an error message.

This mirrors OpenLLM's useful idea that runnable models should be described by
external metadata, while keeping `infer-nexus` independent from Bento.

## 4. Runtime State Design

Current `app.state.registry` and `app.state.runtime_dispatcher` are built once.
Dynamic registration needs one of these changes:

1. Make `ModelRegistry` mutable.
2. Replace the whole registry and dispatcher atomically.
3. Add a `RuntimeStateManager` that owns current snapshots.

Recommended choice: **add `RuntimeStateManager`**.

Reason:

- It avoids mutating indexes while requests are reading them.
- It keeps refresh logic out of FastAPI route handlers.
- It gives future update/delete/reconcile operations a single coordination
  point.

Proposed object:

```python
class RuntimeStateManager:
    def __init__(
        self,
        registry: ModelRegistry,
        serve_builder: ServeApplicationBuilder,
        executor: RuntimeExecutor,
    ) -> None: ...

    def get_registry(self) -> ModelRegistry: ...

    def get_dispatcher(self) -> RuntimeDispatcher: ...

    def replace_registry(self, registry: ModelRegistry) -> None: ...
```

`replace_registry(...)` rebuilds the dispatcher against the new registry. A
simple lock is enough for Phase 1 because replacement is infrequent and fast.

FastAPI dependencies should eventually read from `RuntimeStateManager` instead
of directly from `app.state.registry`.

## 5. Deployment Reconciliation Design

The API layer should not directly call `serve.run(...)`.

Add a real reconciler:

```text
Reconciler
  -> reads desired model state from ModelRegistry
  -> reads observed deployment state from DeploymentRegistry / Ray Serve status
  -> builds one model binding with ServeApplicationBuilder
  -> calls serve.run(..., name=app_name, route_prefix=None, blocking=False)
  -> waits for app/deployment healthy
  -> records serving or failed state
```

The first reconciliation mode should be explicit:

```text
POST /api/reconcile
POST /api/catalog/models/{name}/deploy
```

Do not make every registration automatically deploy in the first pass. Manual
reconcile is easier to test and safer for large model loading failures.

Automatic deployment can be added after the manual path is stable.

## 6. API Design

### 6.1 Register Model

```http
POST /api/catalog/models
```

Request body should match the existing model catalog shape for one model:

```json
{
  "name": "qwen3-32b",
  "alias": "qwen3-chat",
  "task": "chat",
  "backend": "vllm",
  "compat_mode": "vllm_native",
  "model_path": "/nas_data/models/Qwen/Qwen3-32B",
  "dtype": "auto",
  "tensor_parallel_size": 1,
  "max_model_len": 30000,
  "cpu_per_replica": 4,
  "gpu_per_replica": 0.9,
  "gpu_memory_utilization": 0.9,
  "min_replicas": 1,
  "max_replicas": 1,
  "vllm": {
    "openai_serving": {
      "enabled": true
    }
  }
}
```

Response:

```json
{
  "operation_id": "register-qwen3-32b-20260607T103000",
  "model": "qwen3-32b",
  "status": "registered"
}
```

### 6.2 Deploy Model

```http
POST /api/catalog/models/{model_name}/deploy
```

Response:

```json
{
  "operation_id": "deploy-qwen3-32b-20260607T103010",
  "model": "qwen3-32b",
  "status": "deploying"
}
```

### 6.3 Operation Status

```http
GET /api/operations/{operation_id}
```

Response:

```json
{
  "operation_id": "deploy-qwen3-32b-20260607T103010",
  "kind": "deploy_model",
  "model": "qwen3-32b",
  "status": "serving",
  "message": "Serve application infer-nexus-model-qwen3-32b is running"
}
```

### 6.4 Model Status

Existing:

```http
GET /api/models/{model_name}/status
```

Extend it to include deployment status once `DeploymentRegistry` exists:

```json
{
  "name": "qwen3-32b",
  "status": "serving",
  "message": "local artifact present and Serve deployment healthy"
}
```

## 7. Error Handling

Registration errors should be synchronous and return 4xx:

- Invalid Pydantic config: `invalid_model_config`
- Duplicate canonical name: `model_already_exists`
- Duplicate alias: `model_alias_conflict`
- Missing local model artifact: `model_artifact_missing`
- Unsupported backend/task/compat mode: reuse current validation message

Deployment errors should update operation state:

- Ray Serve unavailable: `runtime_not_connected`
- Deployment failed or unhealthy: `deployment_failed`
- Timed out waiting for healthy deployment: `deployment_timeout`
- Backend config error during replica startup: `backend_misconfigured`

Inference requests for registered-but-not-serving models should return:

```json
{
  "error": {
    "message": "Model 'qwen3-32b' is registered but not ready for inference.",
    "type": "invalid_request_error",
    "code": "model_not_ready"
  }
}
```

## 8. Implementation Plan

### Task 1: Add Catalog Loading From `models.d`

**Files:**

- Modify: `src/infer_nexus/catalog/loader.py`
- Test: `tests/test_model_store.py` or new `tests/test_catalog_loader.py`

- [ ] Add `load_model_catalog_sources(base_path, dynamic_dir)` that loads the
      base catalog and sorted dynamic YAML files.
- [ ] Preserve `load_model_catalog(path)` as the existing single-file loader.
- [ ] Reject duplicate names and aliases across all loaded sources.
- [ ] Test that base plus two dynamic model files produces one combined
      `ModelCatalogFile`.
- [ ] Test that duplicate aliases raise a clear error.

Example implementation shape:

```python
def load_model_catalog_sources(
    base_path: str | Path,
    dynamic_dir: str | Path | None = None,
) -> ModelCatalogFile:
    base = load_model_catalog(base_path)
    models = list(base.models)
    if dynamic_dir is not None:
        for path in sorted(Path(dynamic_dir).glob("*.yaml")):
            loaded = load_model_catalog(path)
            models.extend(loaded.models)
    return ModelCatalogFile(models=models)
```

Expected test command:

```bash
uv run pytest tests/test_catalog_loader.py -q
```

### Task 2: Add Dynamic Catalog Persistence

**Files:**

- Create: `src/infer_nexus/catalog/store.py`
- Test: `tests/test_catalog_store.py`

- [ ] Add `DynamicCatalogStore` with `write_model(model)` and
      `delete_model_file(model_name)`.
- [ ] Serialize one model per YAML file under `config/models.d/`.
- [ ] Use safe file names derived from canonical model names.
- [ ] Write through a temporary file and atomic rename.
- [ ] Test writing a model and loading it back through
      `load_model_catalog_sources(...)`.

Example interface:

```python
class DynamicCatalogStore:
    def __init__(self, dynamic_dir: str | Path) -> None:
        self.dynamic_dir = Path(dynamic_dir)

    def model_path(self, model_name: str) -> Path:
        safe_name = model_name.replace("/", "-").replace(" ", "-")
        return self.dynamic_dir / f"{safe_name}.yaml"

    def write_model(self, model: ModelConfig) -> Path:
        self.dynamic_dir.mkdir(parents=True, exist_ok=True)
        target = self.model_path(model.name)
        tmp = target.with_suffix(".yaml.tmp")
        payload = {"models": [model.model_dump(mode="json")]}
        with tmp.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        tmp.replace(target)
        return target
```

Expected test command:

```bash
uv run pytest tests/test_catalog_store.py -q
```

### Task 3: Add Runtime State Manager

**Files:**

- Create: `src/infer_nexus/runtime/state.py`
- Modify: `src/infer_nexus/api/deps.py`
- Modify: `src/infer_nexus/main.py`
- Test: `tests/test_runtime_state.py`

- [ ] Add `RuntimeStateManager`.
- [ ] Store current `ModelRegistry`.
- [ ] Rebuild `RuntimeDispatcher` when the registry is replaced.
- [ ] Update FastAPI dependencies to read registry and dispatcher from the
      state manager.
- [ ] Keep existing dependency names so route code does not need broad changes.

Example interface:

```python
class RuntimeStateManager:
    def __init__(
        self,
        registry: ModelRegistry,
        serve_builder: ServeApplicationBuilder,
        executor: RuntimeExecutor,
    ) -> None:
        self._lock = threading.RLock()
        self._serve_builder = serve_builder
        self._executor = executor
        self._registry = registry
        self._dispatcher = RuntimeDispatcher(registry, serve_builder, executor)

    def get_registry(self) -> ModelRegistry:
        with self._lock:
            return self._registry

    def get_dispatcher(self) -> RuntimeDispatcher:
        with self._lock:
            return self._dispatcher

    def replace_registry(self, registry: ModelRegistry) -> None:
        with self._lock:
            self._registry = registry
            self._dispatcher = RuntimeDispatcher(
                registry=registry,
                serve_builder=self._serve_builder,
                executor=self._executor,
            )
```

Expected test command:

```bash
uv run pytest tests/test_runtime_state.py tests/test_api.py -q
```

### Task 4: Add Operation And Deployment Registries

**Files:**

- Create: `src/infer_nexus/control/operations.py`
- Create: `src/infer_nexus/control/deployments.py`
- Modify: `src/infer_nexus/core/schemas.py`
- Test: `tests/test_operations.py`

- [ ] Add `OperationRecord` with `operation_id`, `kind`, `model`, `status`,
      `message`, and timestamps.
- [ ] Add in-memory `OperationRegistry`.
- [ ] Add in-memory `DeploymentRegistry` keyed by model name.
- [ ] Add response schemas for operation status.
- [ ] Keep persistence out of Phase 1; the model definition itself is persisted
      by `DynamicCatalogStore`.

Statuses:

```python
OperationStatus = Literal["registered", "deploying", "serving", "failed"]
```

Expected test command:

```bash
uv run pytest tests/test_operations.py -q
```

### Task 5: Add Register Model API

**Files:**

- Modify: `src/infer_nexus/api/platform_routes.py`
- Modify: `src/infer_nexus/api/deps.py`
- Modify: `src/infer_nexus/main.py`
- Test: `tests/test_api.py`

- [ ] Add dependency accessors for `DynamicCatalogStore`,
      `RuntimeStateManager`, and `OperationRegistry`.
- [ ] Add `POST /api/catalog/models`.
- [ ] Validate request body as `ModelConfig`.
- [ ] Validate local artifacts using `LocalModelStore.require_model_path(...)`
      when `require_local_artifacts` is true.
- [ ] Write the model to `models.d`.
- [ ] Reload base plus dynamic sources.
- [ ] Replace runtime state manager registry.
- [ ] Return operation status `registered`.
- [ ] Test registration success.
- [ ] Test duplicate alias rejection.
- [ ] Test missing artifact rejection.

Expected test command:

```bash
uv run pytest tests/test_api.py -q
```

### Task 6: Implement Single-model Serve Reconciliation

**Files:**

- Modify: `src/infer_nexus/control/reconciler.py`
- Modify: `src/infer_nexus/runtime/serve_app.py`
- Test: `tests/test_runtime.py`
- Test: `tests/test_reconciler.py`

- [ ] Add `ServeApplicationBuilder.build_serve_binding_for_model(...)`.
- [ ] Add `Reconciler.deploy_model(model_name)`.
- [ ] Use `serve.run(binding, name=app_name, route_prefix=None, blocking=False)`.
- [ ] Reuse the readiness logic shape from `scripts/run_serve_runtime.py`.
- [ ] Update operation and deployment status to `serving` or `failed`.
- [ ] Test with fake Serve that a single app is deployed.
- [ ] Test failed Serve status updates operation to `failed`.

Expected test command:

```bash
uv run pytest tests/test_reconciler.py tests/test_runtime.py -q
```

### Task 7: Add Deploy Model API

**Files:**

- Modify: `src/infer_nexus/api/platform_routes.py`
- Modify: `src/infer_nexus/api/deps.py`
- Modify: `src/infer_nexus/main.py`
- Test: `tests/test_api.py`

- [ ] Add `POST /api/catalog/models/{model_name}/deploy`.
- [ ] Resolve model from current runtime state manager registry.
- [ ] Create operation with status `deploying`.
- [ ] Trigger `Reconciler.deploy_model(...)`.
- [ ] For Phase 1, execute reconciliation synchronously in tests and return the
      resulting operation.
- [ ] In serve mode, keep route implementation ready for FastAPI background
      tasks if deploy latency becomes too high.

Expected test command:

```bash
uv run pytest tests/test_api.py tests/test_reconciler.py -q
```

### Task 8: Gate Inference On Deployment Status

**Files:**

- Modify: `src/infer_nexus/runtime/dispatcher.py`
- Modify: `src/infer_nexus/core/errors.py`
- Modify: `src/infer_nexus/api/openai_routes.py`
- Test: `tests/test_dispatcher.py`
- Test: `tests/test_api.py`

- [ ] Add a `model_not_ready` error type.
- [ ] Let dispatcher or admission check deployment status before invoking a
      local Serve-backed model.
- [ ] Allow proxy-backed models to use proxy readiness rules.
- [ ] Return OpenAI-style error for `/v1/chat/completions` when a registered
      model is not yet serving.
- [ ] Test that registered-but-not-deployed model returns `model_not_ready`.

Expected test command:

```bash
uv run pytest tests/test_dispatcher.py tests/test_api.py -q
```

### Task 9: Document Operator Workflow

**Files:**

- Modify: `docs/DEPLOYMENT_CHECKLIST.md`
- Modify: `docs/MODELS_YAML_CONFIGURATION_MANUAL.md`

- [ ] Add a section explaining static catalog vs dynamic `models.d` catalog.
- [ ] Add curl examples for register, deploy, status, and inference.
- [ ] Add failure recovery instructions:
      - edit or delete dynamic model YAML
      - reload catalog
      - redeploy model
      - inspect operation status
- [ ] State that update/delete/rollback are future follow-ups unless already
      implemented in this phase.

Expected verification:

```bash
rg -n "models.d|POST /api/catalog/models|deploy" docs
```

## 9. Recommended Delivery Phases

### Phase A: Online Catalog Registration

Deliver:

- `models.d` persistence.
- Dynamic catalog loader.
- Runtime state manager.
- `POST /api/catalog/models`.

Value:

- Operators can add a model definition without editing the base catalog.
- The gateway can see the new model without restart.

Limitation:

- The model is not loaded into Ray Serve until deploy/reconcile is triggered.

### Phase B: Manual Online Deployment

Deliver:

- Deployment operation registry.
- `POST /api/catalog/models/{name}/deploy`.
- Reconciler deploys one model into Ray Serve.
- Status endpoint shows `serving` or `failed`.

Value:

- Operators can load a new model without stopping the gateway.
- Failures are isolated to the new model.

### Phase C: Automatic Async Deployment

Deliver:

- Registration optionally triggers background deployment.
- Operation status is updated asynchronously.
- Gateway returns `model_not_ready` until deployment becomes healthy.

Value:

- One API call can register and deploy a model.

### Phase D: Update, Delete, Rollback

Deliver:

- `PATCH /api/catalog/models/{name}`.
- `DELETE /api/catalog/models/{name}`.
- Ray Serve undeploy/drain.
- Deployment revision history.

Value:

- The system becomes a real online model lifecycle control plane.

## 10. First Implementation Cut

The first implementation should stop after Phase B.

Reason:

- It proves online registration and online loading.
- It avoids background-task complexity while the reconciler is new.
- It makes Ray Serve failures easier to debug.
- It preserves existing request routing and Serve handle semantics.

Concrete first milestone:

```text
1. Start gateway and Serve runtime.
2. POST a new model config to /api/catalog/models.
3. Confirm /api/catalog/models lists the new model.
4. POST /api/catalog/models/{name}/deploy.
5. Confirm operation status becomes serving.
6. Send /v1/chat/completions or /v1/embeddings to the new model.
7. Existing models continue serving throughout the flow.
```

## 11. Non-goals For The First Cut

- No automatic placement.
- No multi-node scheduling changes.
- No Kubernetes or KubeRay integration.
- No database requirement.
- No automatic remote model download.
- No per-request hardware routing.
- No silent fallback to another model.
- No update/delete/rollback until register/deploy is stable.

## 12. Verification Matrix

Use these commands while implementing:

```bash
uv run pytest tests/test_catalog_loader.py tests/test_catalog_store.py -q
uv run pytest tests/test_runtime_state.py tests/test_operations.py -q
uv run pytest tests/test_reconciler.py tests/test_runtime.py -q
uv run pytest tests/test_api.py tests/test_dispatcher.py -q
```

Before considering the phase complete, run:

```bash
uv run pytest tests/test_dispatcher.py tests/test_runtime.py tests/test_api.py tests/test_proxy_streaming.py -q
```

Manual serve-mode validation should cover:

```text
register model
deploy model
query operation status
query model status
call OpenAI-compatible endpoint
verify an existing model still works
verify failed model load does not break existing models
```
