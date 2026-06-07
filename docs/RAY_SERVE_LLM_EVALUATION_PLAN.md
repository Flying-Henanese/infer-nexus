# Ray Serve LLM Evaluation Plan

## 1. Background

`infer-nexus` currently uses a custom gateway plus custom Ray Serve deployments for local vLLM models:

```text
Client
  -> infer-nexus FastAPI gateway
  -> catalog / auth / admission / backend dispatch
  -> Ray Serve deployment handle
  -> ModelRuntimeReplica
  -> VLLMBackend
  -> vLLM local-best-effort path or replica-local vLLM OpenAI serving adapter
```

For OpenAI-compatible local vLLM behavior, the project currently contains hand-written adapter logic under:

- `src/infer_nexus/backends/vllm.py`
- `src/infer_nexus/backends/vllm_strict.py`
- `src/infer_nexus/backends/vllm_native/`
- `src/infer_nexus/runtime/deployments.py`
- `src/infer_nexus/runtime/executor.py`

Ray Serve LLM provides a higher-level path through:

```python
from ray.serve.llm import LLMConfig, build_openai_app
```

If this path works in the target runtime environments, it may allow `infer-nexus` to delegate more vLLM/OpenAI protocol behavior to Ray Serve LLM instead of maintaining replica-local vLLM OpenAI serving adapters directly.

This document records the next evaluation and migration work. It is not a decision to replace the current implementation immediately.

## 2. Current Known Runtime Constraints

Target production environments discussed so far:

- CUDA environment: vLLM `0.18.1`
- Huawei Ascend environment: vLLM `0.18.1`
- Huawei Ascend Ray version: Ray `2.48.0`
- Huawei Ascend image: vendor-provided image with Ray, Ray Serve, vLLM, and vLLM-Ascend already installed

Important constraint:

- Do not casually `pip install` or upgrade Ray/vLLM inside the Ascend image before validation. The vendor image may contain a version combination that was tested together with CANN, `torch-npu`, `vllm`, `vllm-ascend`, and Ray.

## 3. What Ray Serve LLM Could Replace

The current `vllm_native` path exists to call vLLM's OpenAI serving internals inside a Ray Serve replica.

The current hand-written responsibilities include:

- dynamically importing vLLM OpenAI serving classes
- constructing vLLM OpenAI chat and embedding request objects
- constructing or adapting vLLM serving objects
- adapting engine-client method signatures
- normalizing full chat completion payloads
- passing streaming chat chunks through without semantic mutation
- normalizing embedding payloads
- surfacing strict adapter initialization and invocation failures

If `build_openai_app` is usable, Ray Serve LLM could take over much of this OpenAI-compatible serving surface:

- chat completions
- streaming chat completions
- embeddings
- score-style text-pair scoring, if it can be safely mapped to `infer-nexus` rerank semantics
- Ray Serve LLM request routing, including `PrefixCacheAffinityRouter`
- vLLM engine/deployment configuration encapsulation

This does not mean `infer-nexus` should remove its gateway. Ray Serve LLM should be evaluated as an internal backend implementation behind the existing gateway.

The long-term target is the internal Serve-handle path, not an internal HTTP proxy:

```text
Client
  -> infer-nexus FastAPI gateway
  -> catalog / auth / admission / alias rewrite
  -> Ray Serve application or deployment handle
  -> Ray Serve LLM app built by build_openai_app
  -> Ray Serve LLM router / server
  -> vLLM / vLLM-Ascend engine
```

An internal HTTP route may still be useful for smoke tests and debugging, but the preferred production direction is to keep the existing efficient gateway-to-Serve-handle topology and replace the custom replica implementation behind that boundary.

## 4. Validation Task 1: Verify Ray Serve LLM On Huawei Ascend

Goal:

- Confirm that Ray `2.48.0` in the Huawei Ascend image can import Ray Serve LLM APIs and use `build_openai_app` to start a vLLM-backed Serve application.

### 4.1 Import-Level Validation

Run inside the Ascend runtime image:

```bash
python -c "import ray; print(ray.__version__); from ray import serve; print('serve ok'); from ray.serve.llm import LLMConfig, LLMServingArgs, build_openai_app; print('serve llm ok')"
```

Expected result:

- Ray version prints `2.48.0`
- `ray.serve.llm` imports successfully
- `LLMConfig`, `LLMServingArgs`, and `build_openai_app` are available

Failure handling:

- If `ray.serve.llm` is missing, record the exact import error and installed Ray package metadata.
- If extra dependencies are missing, do not immediately mutate the vendor image. First determine whether the missing package can be installed safely in a separate test image.

### 4.2 Minimal `build_openai_app` Deployment

Create a minimal test script outside the production service first. The script should:

- start or connect to Ray
- construct one `LLMConfig`
- call `build_openai_app`
- deploy with `serve.run`
- confirm the returned app handle or `serve.get_app_handle(...)` can invoke the generated app without an HTTP hop
- issue a local OpenAI-compatible chat completion request through the handle path
- issue a streaming chat completion request through the handle path
- optionally issue the same requests through HTTP as a reference/debug comparison
- stop the Serve app cleanly

The test should use a known-good local model path from the Ascend environment.

Validation points:

- the model loads through vLLM-Ascend
- Ray resources map correctly to Ascend NPU resources
- no CUDA-specific assumptions leak into the path
- `trust_remote_code`, `tensor_parallel_size`, `max_model_len`, dtype, tool parser, reasoning parser, and other engine kwargs are accepted or mapped
- handle invocation accepts a stable request shape that can be constructed from the gateway payload
- streaming over the handle path returns a stable async result or generator shape
- HTTP and handle paths have equivalent OpenAI semantics where both are available
- Ray Serve metrics and health status remain usable

### 4.3 Embedding Validation

Ray Serve LLM exposes embeddings APIs in recent documentation. Validate on Ascend:

- one embedding model, if available
- request and response shape
- batching behavior
- compatibility with current `EmbeddingRequest` and `EmbeddingResponse`

Decision point:

- If Ray Serve LLM embeddings are stable, `vllm_native` embedding adapter logic can be a migration candidate.
- If not, keep the current local-best-effort or proxy embedding path.

### 4.4 Rerank / Score Validation

Ray Serve LLM has score-style text-pair scoring in its API surface, but this should not be assumed to be identical to the current `infer-nexus` rerank API.

Validate:

- whether `build_openai_app` exposes scoring through the handle path in Ray `2.48.0`
- whether the HTTP route exists and can be used as a reference path
- whether the request schema accepts query/document pairs or only raw text pairs
- whether the response schema can be mapped to current `RerankResponse`
- whether Qwen3-Reranker or the target reranker model behaves correctly under vLLM `0.18.1` and vLLM-Ascend

Decision point:

- If score output is stable, implement a thin gateway mapping:

```text
query + documents[]
  -> text_pairs[]
  -> scores[]
  -> sorted rerank results
```

- If not, keep the current rerank implementation or proxy rerank path.

### 4.5 Prefix Cache Affinity Validation

Current `infer-nexus` already supports passing Ray Serve LLM's `PrefixCacheAffinityRouter` through `deployment_config.request_router_config`.

For the `build_openai_app` path, validate:

- whether Ray Serve LLM uses prefix-aware routing automatically or through explicit config
- whether the router works with the generated OpenAI ingress
- whether the router still works when the gateway calls the generated app through a Serve handle
- whether it improves TTFT, prefill latency, p95/p99 latency, and timeout rate for repeated chat workloads

## 5. Validation Task 2: Migration And Code Changes

The recommended implementation strategy is additive:

1. Add a new experimental backend or compatibility mode.
2. Keep the current `vllm` and `vllm_native` paths intact.
3. Route selected models through Ray Serve LLM behind the existing gateway.
4. Promote only after CUDA and Ascend validation both pass.

Possible naming options:

```yaml
backend: ray_serve_llm
```

or:

```yaml
backend: vllm
compat_mode: ray_serve_llm
```

The second option keeps Ray Serve LLM as a vLLM serving mode. The first option makes it a distinct backend. The cleaner choice depends on whether Ray Serve LLM will also own deployment topology and not just request compatibility.

### 5.1 Configuration Schema

Likely files:

- `src/infer_nexus/catalog/models.py`
- `docs/MODELS_YAML_CONFIGURATION_MANUAL.md`
- `config/models.yaml`

Add or formalize fields for Ray Serve LLM:

- Ray Serve LLM enablement flag or backend type
- `model_loading_config`
- `deployment_config`
- `engine_kwargs`
- `served_model_name`
- route/app naming policy
- optional OpenAI app route prefix
- optional Ray Serve LLM router config

Keep existing gateway-facing fields:

- `name`
- `alias`
- `task`
- `served_model_name`
- `capabilities`
- admission and policy metadata

### 5.2 Runtime App Builder

Likely files:

- `src/infer_nexus/runtime/serve_app.py`
- `src/infer_nexus/runtime/deployments.py`
- `scripts/run_serve_runtime.py`
- `tests/test_runtime.py`
- `tests/test_scripts.py`

Current local vLLM path builds:

```text
serve.deployment(ModelRuntimeReplica).bind(runtime_context)
```

Ray Serve LLM path would build:

```text
LLMConfig(...)
build_openai_app(...)
serve.run(...)
```

Required changes:

- introduce a builder for Ray Serve LLM application bindings
- decide one model per Serve app vs one multi-model Ray Serve LLM app
- preserve stable app names for gateway handle lookup
- record the generated app's ingress handle contract in tests
- preserve resource configuration and autoscaling config
- pass through `PrefixCacheAffinityRouter` where supported
- expose deployment readiness in the existing startup wait loop

### 5.2.1 Single App With Multiple Models

One important design question is whether local Ray Serve LLM models should be deployed as:

```text
one Ray Serve LLM app per model
```

or:

```text
one Ray Serve LLM app
  -> multiple LLMConfig entries
  -> multiple model deployments
```

The second shape may simplify `infer-nexus` routing substantially. The gateway would keep owning external model lookup and alias rewriting, then send all local Ray Serve LLM requests to one app handle:

```text
Client
  -> infer-nexus gateway
  -> auth / catalog / alias rewrite / admission
  -> one Ray Serve LLM app handle
  -> Ray Serve LLM routes by request.model
  -> model-specific Serve deployment
  -> replica router / PrefixCacheAffinityRouter
  -> vLLM / vLLM-Ascend
```

Important clarification:

- This does not mean putting all models into one `LLMConfig`.
- It means passing multiple model-specific `LLMConfig` objects into one `build_openai_app(...)` call, if the target Ray version supports that shape.

Potential benefits:

- fewer Serve apps to create and monitor
- simpler gateway handle lookup
- less per-model app naming and route-prefix management
- Ray Serve LLM can own model routing inside the generated OpenAI-compatible app
- prefix-cache affinity remains a per-model-deployment replica-routing concern

Risks and validation points:

- Ray `2.48.0` may have different multi-model `build_openai_app` behavior than newer Ray versions.
- Per-model health and readiness must remain visible even when one app contains multiple deployments.
- A failure in one model deployment must not make the whole local LLM app operationally ambiguous.
- Per-model resource settings, autoscaling config, NPU resources, and engine kwargs must remain independently configurable.
- Chat, embedding, and score/rerank models may not all fit cleanly in one app; a task-grouped app split may be needed.
- Gateway alias rewriting must map client-facing model names to the exact served model names recognized by Ray Serve LLM.

Recommended validation order:

1. Start with one app containing one chat model.
2. Add a second chat model to the same app.
3. Verify request routing by `model`.
4. Verify each model has independent replicas, autoscaling, and health state.
5. Verify `PrefixCacheAffinityRouter` is applied within each model deployment, not across different models.
6. Only then test embedding and score/rerank models in the same app.

Possible final shapes:

```text
Option 1:
  one Ray Serve LLM app for all local OpenAI-compatible models

Option 2:
  one Ray Serve LLM app for chat models
  one Ray Serve LLM app for embedding models
  keep rerank on current backend until score mapping is proven

Option 3:
  one Ray Serve LLM app per model, only if multi-model routing or health isolation is weak in Ray 2.48.0
```

### 5.3 Gateway Dispatch

Likely files:

- `src/infer_nexus/api/openai_routes.py`
- `src/infer_nexus/runtime/dispatcher.py`
- `src/infer_nexus/runtime/executor.py`
- `src/infer_nexus/runtime/handles.py`
- `tests/test_api.py`
- `tests/test_dispatcher.py`

The gateway should continue to own:

- auth
- model lookup
- alias resolution
- admission
- request ID handling
- platform errors
- `/api/*` endpoints

For Ray Serve LLM-backed models, the gateway should use the Serve handle path as the target production topology:

```text
gateway
  -> serve.get_app_handle(app_name) or serve.run(...) returned handle
  -> Ray Serve LLM generated app
  -> vLLM / vLLM-Ascend
```

Required behavior:

- avoid an extra internal HTTP hop
- preserve OpenAI-compatible request semantics
- rewrite only the client-facing `model` field to the internal served model name when needed
- preserve streaming behavior through the handle response shape
- map Ray Serve LLM errors into the existing `infer-nexus` gateway error taxonomy
- keep request ID and observability metadata at the gateway boundary

The main validation risk is that `build_openai_app` produces an OpenAI ingress app, so the generated app's handle invocation contract may be less obvious than the HTTP contract. The migration should explicitly test:

- whether `serve.run(...)` returns a usable ingress handle
- whether `serve.get_app_handle(app_name)` returns an equivalent live handle
- whether the handle accepts dict payloads, FastAPI/Starlette request-like objects, or another stable input type
- whether non-streaming responses resolve to dict/Pydantic/Starlette response objects
- whether streaming responses can be consumed as async generators without going through HTTP SSE parsing

Internal HTTP should be retained only as:

- a smoke-test path
- a debugging reference against OpenAI-compatible HTTP behavior
- a fallback option if handle invocation is structurally unusable in Ray `2.48.0`

### 5.4 vLLM Native Adapter Reduction

Likely files:

- `src/infer_nexus/backends/vllm_native/chat.py`
- `src/infer_nexus/backends/vllm_native/embedding.py`
- `src/infer_nexus/backends/vllm_native/common.py`
- `src/infer_nexus/backends/vllm_strict.py`
- `src/infer_nexus/backends/vllm.py`
- `tests/test_runtime.py`

If Ray Serve LLM is adopted for selected models:

- keep current `vllm_native` initially as fallback
- add tests proving Ray Serve LLM path returns OpenAI-compatible chat and streaming responses
- stop adding new vLLM internal import workarounds to `vllm_native` unless needed for fallback
- after enough validation, consider deleting or shrinking:
  - dynamic vLLM OpenAI serving imports
  - engine-client compatibility proxy
  - chat serving adapter wrapper
  - embedding serving adapter wrapper
  - strict native adapter initialization branches

Do not remove local-best-effort immediately. It is still useful for development stubs, lightweight tests, and fallback behavior.

### 5.5 Proxy Backend Interaction

Likely files:

- `src/infer_nexus/backends/vllm.py`
- `src/infer_nexus/runtime/dispatcher.py`
- `docs/OPENAI_PROXY_REFACTOR_PLAN.md`

Ray Serve LLM should be treated as an internally managed Serve app, not as an external OpenAI-compatible upstream.

Keep clear separation:

- `vllm_openai_proxy`: external upstream configured by URL
- `ray_serve_llm`: internally managed Ray Serve LLM app invoked primarily through Serve handles
- `vllm`: current custom local Ray Serve replica path

This avoids confusing "proxy to external service" with "internally managed Ray Serve LLM runtime".

### 5.6 Tests And Validation Coverage

Add unit tests for:

- config parsing
- Ray Serve LLM runtime plan generation
- app naming
- model alias and served model rewrite
- handle request invocation
- handle streaming consumption
- error propagation
- embeddings response mapping
- score-to-rerank mapping, if implemented

Add integration tests for:

- CUDA + vLLM `0.18.1`
- Ascend + Ray `2.48.0` + vLLM `0.18.1`
- chat non-streaming
- chat streaming
- handle vs HTTP semantic parity, if HTTP is available as a reference path
- tools/tool choice
- reasoning parser
- multimodal payloads, if target model supports them
- embeddings
- rerank/score, if target model supports it
- prefix cache affinity behavior

## 6. Validation Task 3: Advantages And Risks

### 6.1 Advantages Compared With Current Implementation

Less hand-written protocol compatibility code:

- `build_openai_app` delegates OpenAI-compatible request handling to Ray Serve LLM.
- This may reduce local request/response reconstruction logic and vLLM internal import coupling.

Closer to Ray/vLLM upstream behavior:

- Ray Serve LLM is designed to wrap vLLM serving behavior in Ray Serve.
- It should track upstream OpenAI-compatible serving changes better than project-local adapters.

Cleaner streaming behavior:

- Streaming chunks can be produced by the Ray Serve LLM/vLLM stack rather than reconstructed by `infer-nexus`.

Preserves the efficient internal-call topology:

- The target design keeps the current gateway-to-Ray-Serve-handle shape instead of adding an internal HTTP hop.
- This should preserve most of the current dispatch efficiency while reducing local OpenAI protocol adapter code.

Better alignment with Ray Serve LLM routing:

- `PrefixCacheAffinityRouter` and related Ray Serve LLM routing features may work more naturally with the generated OpenAI app than with custom method-handle payloads.

Simpler local vLLM chat and embedding path:

- The custom `vllm_native` adapter layer may become smaller or unnecessary for models that use Ray Serve LLM.

More standard deployment vocabulary:

- `LLMConfig`, `model_loading_config`, `deployment_config`, and `engine_kwargs` align with Ray Serve LLM documentation and examples.

### 6.2 Risks Compared With Current Implementation

Ray Serve LLM API maturity:

- The API has been treated as beta-stage surface area in current planning.
- Ray `2.48.0` is an early version for this stack, especially on Ascend.

Version sensitivity:

- CUDA may use newer Ray versions, while Ascend currently uses Ray `2.48.0`.
- vLLM `0.18.1` is newer than the vLLM versions discussed in some Ray release notes around Ray `2.48.0`.
- Behavior must be validated in the actual vendor image.

Ascend compatibility risk:

- Ray Serve LLM may assume CUDA-oriented defaults in some paths.
- Resource mapping must work with custom NPU resources and vLLM-Ascend.
- Engine kwargs may differ between CUDA vLLM and vLLM-Ascend.

Gateway responsibility overlap:

- `build_openai_app` generates an OpenAI-compatible app.
- `infer-nexus` already owns the northbound OpenAI-compatible gateway.
- Directly exposing Ray Serve LLM would bypass auth, catalog lookup, admission control, platform APIs, and alias policy.

Operational topology complexity:

- Running a generated OpenAI app behind the existing gateway introduces a second Serve ingress shape.
- The target handle path avoids an internal HTTP hop, but only if the generated app's handle invocation contract is stable enough.
- Startup, readiness, shutdown, and route-prefix ownership need to be made explicit.

Handle invocation uncertainty:

- `build_openai_app` is primarily documented as an OpenAI-compatible Serve app.
- The HTTP contract is more obvious than the Python handle contract.
- Streaming, error responses, and response object normalization must be tested through handles, not inferred from HTTP behavior.

Embedding and rerank parity risk:

- Embeddings appear to be a natural fit, but still need schema and runtime validation.
- Rerank should not be assumed to be native. Ray Serve LLM score APIs may need gateway-side schema conversion and sorting.

Loss of current fallback behavior:

- The current implementation has local-best-effort paths and explicit fallback/stub behavior.
- A Ray Serve LLM path should be additive until it proves equivalent or better.

Debuggability:

- More behavior moves into Ray Serve LLM internals.
- This can reduce local code but may make version-specific failures harder to diagnose.

## 7. Recommended Evaluation Sequence

1. Run import-level validation on Ascend Ray `2.48.0`.
2. Build a minimal external `build_openai_app` smoke script for one chat model.
3. Validate `serve.run(...)` returned handle and `serve.get_app_handle(...)` invocation.
4. Validate chat non-streaming and streaming through the handle path on Ascend.
5. Validate optional HTTP behavior only as a reference path.
6. Validate the same handle script on CUDA with vLLM `0.18.1`.
7. Validate embeddings if a target embedding model is available.
8. Validate score/rerank mapping only after chat and embeddings are stable.
9. Add an experimental `ray_serve_llm` backend or `ray_serve_llm` compat mode.
10. Route one non-critical chat model through the new handle path behind the existing gateway.
11. Compare Ray Serve LLM handle responses against current `vllm_native` responses.
12. Measure latency, TTFT, error rate, streaming behavior, and prefix-cache affinity behavior.
13. Decide whether to migrate more models or keep Ray Serve LLM as an optional backend.

## 8. Decision Criteria

Adopt Ray Serve LLM for local chat models if:

- `build_openai_app` works on Ascend Ray `2.48.0` without unsafe image mutation
- the generated app can be invoked efficiently through a Serve handle
- non-streaming and streaming handle contracts are stable enough to test and support
- it supports required vLLM-Ascend engine kwargs
- chat and streaming behavior match or improve over current `vllm_native`
- tool calling, reasoning, and multimodal payloads work for target models
- readiness and shutdown can be integrated cleanly
- prefix-cache affinity remains available and measurable
- failure modes can be mapped cleanly through the existing gateway

Adopt Ray Serve LLM for embeddings if:

- embedding endpoint behavior is stable on both CUDA and Ascend
- response schema matches current API expectations or requires only thin mapping
- performance is acceptable

Adopt Ray Serve LLM for rerank only if:

- score APIs are available in the deployed Ray version
- target reranker models produce correct scores
- gateway-side mapping to `query + documents[] -> ranked results[]` is simple and tested

Do not adopt Ray Serve LLM as the public entrypoint unless the project intentionally redesigns gateway ownership. The safer target remains:

```text
Client
  -> infer-nexus gateway
  -> Ray Serve handle
  -> Ray Serve LLM internal app
  -> vLLM / vLLM-Ascend
```
