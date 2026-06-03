# Infer-Nexus Compatibility Refactor Plan

## Goal

Reduce protocol drift between `infer-nexus` and upstream `vLLM` / OpenAI-compatible behavior.

The core direction is:

- keep the gateway focused on routing, policy, and observability
- minimize request/response reinterpretation in local gateway code
- use native `vLLM` OpenAI serving inside each Ray replica as the primary local chat compatibility path
- treat `vllm_openai_proxy` as an explicit HTTP proxy to an external OpenAI-compatible upstream, not as access to replica-local `vLLM`
- keep local `LLM.chat(...)` behavior as a limited fallback, not the compatibility baseline

The desired endpoint shape is:

```text
Client OpenAI SDK
  -> infer-nexus /v1/chat/completions
  -> RuntimeDispatcher model routing
  -> Ray Serve replica
  -> VLLMBackend strict OpenAI path
  -> vLLM OpenAIServingChat.create_chat_completion(...)
  -> native OpenAI-compatible response / stream chunks
```

The Ray replica does not need to start a separate HTTP server. The HTTP endpoint
is still provided by `infer-nexus`; the replica-local responsibility is to pass
the reconstructed OpenAI request payload into vLLM's native OpenAI serving layer.


## Problem Summary

The current chat path still contains gateway-local protocol interpretation in
[src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py).

This creates repeated compatibility failures when upstream `vLLM` behavior changes or when callers use modern OpenAI-style payloads:

- text content blocks vs plain string content
- multimodal message content
- tool-calling request fields
- reasoning-related request fields
- streaming chunk shape
- native OpenAI serving adapter availability and fallback behavior

The specific class of bugs we just hit is a symptom of this: the gateway interpreted message content locally and rejected payloads that should have been normalized or passed through.


## Terminology

- `backend: vllm`
  - A local Ray Serve replica owns the vLLM engine lifecycle.
  - In strict mode, chat requests must be handled through vLLM's native OpenAI serving adapter inside that replica.
  - In best-effort mode, chat requests may use `LLM.chat(...)` / `AsyncLLMEngine.generate(...)` and gateway-local adaptation.

- `backend: vllm_openai_proxy`
  - The gateway forwards HTTP requests to `proxy_config.upstream_base_url`.
  - The upstream is an already-running OpenAI-compatible service, typically `vllm serve`.
  - This backend is not a mechanism for calling the vLLM engine inside a Ray replica.

- OpenAI-compatible HTTP endpoint
  - The externally visible endpoint is `infer-nexus`.
  - A local Ray replica can participate in that endpoint by using vLLM's OpenAI serving adapter, but the replica itself is not an HTTP server.


## Target Architecture

### Desired State

For chat requests:

1. **Strict compatibility models**
   - for `backend: vllm`, must use replica-local native `vLLM` OpenAI serving
   - for `backend: vllm_openai_proxy`, must proxy to an external OpenAI-compatible HTTP upstream
   - must not silently fall back to local `LLM.chat(...)`
   - may only rewrite the `model` field to the internal served model name and merge explicit pass-through extras

2. **Best-effort local models**
   - may use local `LLM.chat(...)`
   - do not claim full OpenAI compatibility
   - support only the subset explicitly documented by this project

### Gateway Responsibilities

The gateway should primarily own:

- auth
- routing
- model selection
- admission / rate limiting
- request ID propagation
- logging / metrics
- minimal model name remapping

The gateway should avoid owning:

- detailed chat message semantics
- multimodal content interpretation for strict models beyond basic schema acceptance and payload preservation
- tool-calling protocol behavior replication
- reasoning field semantics
- stream chunk protocol recreation for strict models

## Current Status

Status as of `2026-06-01`:

- Phase 1 is functionally complete in the current codebase.
- Phase 2 core chat-path goals are functionally complete for local `backend: vllm` strict models.
- Phase 3 is partially complete: high-signal tests exist in the current suite, but the plan's full CI/contract-test closure is not finished yet.
- For the currently deployed model set and the current compatibility scope, the refactor can be considered operationally closed.

What has been verified in code and against a remote instance:

- the local `backend: vllm` implementation has now been structurally split into:
  - `src/infer_nexus/backends/vllm.py` as the facade / lifecycle / dispatch layer
  - `src/infer_nexus/backends/vllm_local_best_effort.py` for the legacy
    `local_best_effort` chat path
  - `src/infer_nexus/backends/vllm_strict.py` for replica-local native
    `vllm_native` / strict serving
- after that split, `VLLMBackend` still remains the single external backend
  entrypoint, but strict and local-best-effort execution now dispatch through
  dedicated executors instead of one monolithic implementation file
- `compat_mode` is implemented in model config and runtime spec propagation.
- strict local `backend: vllm` chat models require `vllm.openai_serving.enabled=true`.
- strict local chat requests do not silently fall back to local `LLM.chat(...)` when the native serving adapter is unavailable.
- strict local chat payloads are passed into vLLM OpenAI serving with minimal mutation: `model` rewrite plus `extra_body` merge.
- strict local streaming passes through native OpenAI-style SSE chunks.
- adapter initialization failures are recorded and surfaced as backend configuration errors.
- a remote `qwen3.5-9b` strict instance was validated with:
  - non-stream chat
  - stream chat
  - reasoning responses
  - tool calling with `tool_call_parser=qwen3_coder`
  - multimodal image input
  - `2026-06-02` follow-up validation of thinking-enabled requests confirmed that
    final-answer truncation tracks request `max_tokens` budget, not startup
    `max_model_len`
- a remote `qwen3-32b` instance was validated with:
  - non-stream chat
  - tool calling
- a remote `qwen3-vl-chat-8b-instruct` instance was validated with:
  - multimodal image input
- a remote `qwen3-embedding-8b` instance was validated with:
  - `/v1/embeddings`
  - non-empty vector output
  - `2026-06-02` follow-up validation confirmed that the model runs correctly in
    `compat_mode: vllm_native` after completing the replica-local native
    embeddings serving adapter path
- a remote `bge-reranker` instance was validated with:
  - `/v1/rerank`
  - correct ranking for a simple factual pair

Operational guidance discovered during validation:

- `Qwen3.5-9B` tool calling is stable with:
  - `compat_mode: strict_openai`
  - `vllm.openai_serving.enabled=true`
  - `engine_kwargs.enable_auto_tool_choice=true`
  - `engine_kwargs.tool_call_parser=qwen3_coder`
  - `openai_serving.reasoning_parser=qwen3`
- `Qwen3.5-9B` should not default to thinking-enabled request behavior for general assistant traffic.
- The recommended pattern is:
  - default `chat_template_kwargs.enable_thinking=false`
  - explicitly enable thinking per request only for reasoning-heavy tasks
- Request-level override using `extra_body.chat_template_kwargs.enable_thinking=false` was verified to restore concise final-answer behavior for `qwen3.5-9b`.
- Additional `2026-06-02` runtime finding for `Qwen3.5-9B`:
  - with `extra_body.chat_template_kwargs.enable_thinking=true`, the model can
    spend most of the completion budget in the `reasoning` field before emitting
    final `content`
  - a short prompt (`prompt_tokens=31`) still hit `finish_reason="length"` when
    `max_tokens=192`, which rules out startup `max_model_len` as the cause for
    that failure mode
  - the same prompt completed successfully with final `content` once
    `max_tokens` was raised to `256` (`completion_tokens=232`)
  - operationally, reasoning-enabled requests should budget completion tokens
    explicitly; otherwise the model may terminate inside reasoning without ever
    emitting the final answer
- Additional `2026-06-02` runtime finding for `qwen3-embedding-8b`:
  - switching the model to `compat_mode: vllm_native` exposed multiple missing
    pieces in the local native embeddings serving integration rather than a
    problem in vLLM itself
  - the integration was fixed incrementally by:
    - adding explicit initialization of the replica-local native embeddings
      serving adapter during backend startup
    - passing required constructor kwargs such as `request_logger=None` to
      vLLM's `ServingEmbedding(...)`
    - constructing concrete `EmbeddingCompletionRequest` objects instead of
      trying to instantiate the `EmbeddingRequest` union type alias
    - extending the shared engine-client compat proxy to adapt `encode(...)`
      calls, including dropping unsupported kwargs like `trace_headers`
    - defaulting `pooling_task="embed"` when the native serving stack reaches a
      generic local `LLM.encode(...)`
  - after those fixes, remote `/v1/embeddings` requests succeeded for both a
    single string input and a list input with `encoding_format="float"`
- Additional `2026-06-02` refactor validation finding:
  - after splitting `vllm.py` into facade + `vllm_local_best_effort.py` +
    `vllm_strict.py`, the remote smoke suite still passed for the current
    deployed model set
  - verified paths after the split included:
    - strict/native non-stream chat
    - strict/native stream chat
    - strict/native tool calling
    - strict/native multimodal chat
    - native embeddings
    - local-best-effort rerank
  - this gives high confidence that the structural extraction itself did not
    regress the primary runtime paths

What remains before this plan can be called fully closed:

- explicit failure-path validation in a target runtime for adapter initialization/startup failure scenarios
- completion of the Phase 3 contract-test story, especially:
  - dedicated end-to-end compatibility contract coverage
  - CI confidence that future `vLLM` upgrades cannot silently drift request/response behavior
- broader verification of the external `vllm_openai_proxy` strict path beyond the current unit-test coverage

## Remote Validation Record

Remote validation target:

- host: `192.168.0.194:8000`
- date: `2026-06-01`
- follow-up date: `2026-06-02`

Remote model set observed during validation:

- `qwen3-vl-chat-8b-instruct`
- `qwen3-32b`
- `mineru`
- `qwen3-embedding-8b`
- `bge-reranker`
- `qwen3.5-9b`

Validation summary:

- `qwen3.5-9b`
  - strict OpenAI path verified
  - native tool calling verified
  - native reasoning field behavior verified
  - native SSE stream shape verified
  - multimodal request acceptance verified
  - `2026-06-02` follow-up:
    - non-stream request with `enable_thinking=false` returned concise final
      `content` (`"OK"`) as expected
    - streaming request returned native OpenAI-style
      `chat.completion.chunk` SSE frames ending with `data: [DONE]`
    - thinking-enabled requests with a short prompt and `max_tokens=192`
      terminated with `finish_reason="length"` and reasoning-only output
    - the same thinking-enabled prompt completed with final `content` at
      `max_tokens=256`, `384`, and `512`
    - this behavior indicates completion-budget exhaustion inside reasoning, not
      a remote `max_model_len` bottleneck
- `qwen3-32b`
  - local-best-effort chat verified
  - tool calling behavior verified
  - `2026-06-02` post-refactor follow-up:
    - non-stream strict/native request returned `"OK"`
    - streaming request returned native OpenAI-style
      `chat.completion.chunk` SSE frames ending with `data: [DONE]`
- `qwen3-vl-chat-8b-instruct`
  - multimodal image prompt verified
  - `2026-06-02` post-refactor follow-up:
    - multimodal request with an externally hosted `https` image URL succeeded
      and returned the expected color description (`"white"`)
    - multimodal request with a real local JPEG carried as
      `data:image/jpeg;base64,...` also succeeded and produced a correct
      one-sentence scene description
    - an earlier failing request using a tiny inline PNG data URL appears to be
      a sample-specific `vLLM + Pillow` decode edge case rather than a general
      strict/native multimodal regression
- `qwen3-embedding-8b`
  - embeddings endpoint verified
  - `2026-06-02` native-serving follow-up:
    - startup initially failed because `ServingEmbedding.__init__()` required
      `request_logger`
    - request handling then failed because the integration tried to instantiate
      the `EmbeddingRequest` union alias instead of a concrete
      `EmbeddingCompletionRequest`
    - request handling then failed again because the native serving stack passed
      `trace_headers` into a local `LLM.encode(...)` that did not accept that
      kwarg
    - request handling then failed once more because generic `LLM.encode(...)`
        required `pooling_task="embed"`
    - after fixing those four compatibility gaps, remote `/v1/embeddings`
      succeeded for:
      - `{"model":"qwen3-embedding-8b","input":"hello world"}`
      - `{"model":"qwen3-embedding-8b","input":["hello","world"],"encoding_format":"float"}`
- `bge-reranker`
  - rerank endpoint verified
  - `2026-06-02` post-refactor follow-up:
    - rerank behavior remained correct after the executor split
- `mineru`
  - OpenAI-style multimodal smoke test returned a non-informative payload during this run, but separate validation outside this document confirmed the model itself is operational
  - `2026-06-02` post-refactor follow-up:
    - simple OpenAI-style chat smoke request still returned successfully,
      indicating that the facade / executor split did not break request routing
- `qwen3.5-9b`
  - `2026-06-02` post-refactor multimodal follow-up:
    - multimodal request with an externally hosted `https` image URL succeeded
      and returned `"white"`
    - multimodal request with the local JPEG carried as
      `data:image/jpeg;base64,...` also succeeded and produced a correct
      one-sentence scene description
    - this confirms that strict/native multimodal handling is still working
      after the extraction to `vllm_strict.py`

Closure decision for this plan:

- Closed for the current deployment scope and the current primary compatibility objective:
  - strict local `backend: vllm` chat compatibility for `qwen3.5-9b`
  - baseline chat availability for the remaining chat models
  - embedding and rerank endpoint availability for the newly added models
- Not treated as a blanket guarantee for every future `vLLM` upgrade or every possible proxy/native topology; those remain covered by the open test/CI follow-up items above.


## Implementation Plan

## Phase 1: Stop Silent Drift

### Objective

Make compatibility failures explicit instead of silently degrading into a different execution path.

### Changes

1. **Fail fast when native serving is required but unavailable**
   - File:
     - [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
   - Change:
     - when `compat_mode=strict_openai` and `backend: vllm`, require `vllm.openai_serving.enabled=true`
     - adapter init failure must surface as a startup or request error
     - remove silent fallback to `LLM.chat(...)` / `AsyncLLMEngine.generate(...)` for strict models
   - Reason:
     - silent fallback hides protocol drift and makes debugging much harder

2. **Introduce explicit compatibility mode in model config**
   - Files:
     - [src/infer_nexus/catalog/models.py](/F:/GitHub/infer-nexus/src/infer_nexus/catalog/models.py)
     - [src/infer_nexus/runtime/dispatcher.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/dispatcher.py)
   - Proposed config:
     - `compat_mode: strict_openai | local_best_effort`
   - Meaning:
     - `strict_openai`: replica-local native OpenAI serving for `backend: vllm`, or external HTTP proxy for `backend: vllm_openai_proxy`
     - `local_best_effort`: local backend path is allowed

3. **Improve adapter initialization visibility**
   - Files:
     - [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
     - optionally [src/infer_nexus/api/openai_routes.py](/F:/GitHub/infer-nexus/src/infer_nexus/api/openai_routes.py)
   - Change:
     - log adapter init failure with concrete reason
     - make it visible in request failure logs

### Acceptance Criteria

- strict models do not silently fall back to local `LLM.chat(...)`
- `backend: vllm` strict models fail clearly when replica-local native OpenAI serving cannot be constructed
- adapter init failures are visible in logs
- model config clearly states compatibility expectations


## Phase 2: Make Ray Replicas Behave as OpenAI-Compatible Endpoints

### Objective

Make `infer-nexus` + replica-local vLLM OpenAI serving behave like an OpenAI-compatible HTTP endpoint while preserving Ray replica ownership of the local model runtime.

### Changes

1. **Use replica-local native OpenAI serving as the strict local chat path**
   - Files:
     - [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
     - [src/infer_nexus/runtime/deployments.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/deployments.py)
     - [src/infer_nexus/runtime/serve_app.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/serve_app.py)
   - Change:
     - for strict chat models, require native serving adapter readiness during startup
     - fail deployment startup if the adapter cannot be constructed
     - do not create an additional HTTP server inside the replica

2. **Pass OpenAI payloads into vLLM serving with minimal mutation**
   - File:
     - [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
   - Change:
     - build the vLLM serving request from the original OpenAI request model
     - rewrite only the `model` field to `served_model_name`
     - merge `extra_body` into the top-level payload
     - preserve `messages`, `tools`, `tool_choice`, `response_format`, `stream`, `stream_options`, reasoning fields, multimodal content blocks, and vLLM extension fields
   - Reason:
     - vLLM's OpenAI serving layer, not the gateway, should own protocol semantics

3. **Pass through strict streaming chunks**
   - Files:
     - [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
     - [src/infer_nexus/runtime/executor.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/executor.py)
     - optionally [src/infer_nexus/api/openai_routes.py](/F:/GitHub/infer-nexus/src/infer_nexus/api/openai_routes.py)
   - Change:
     - if vLLM OpenAI serving yields bytes or strings, pass them through as SSE chunks
     - if it yields dict chunks, frame them as SSE without changing the chunk body
     - do not use local delta reconstruction for strict models

4. **Downgrade local `LLM.chat(...)` to a documented fallback path**
   - File:
     - [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
   - Change:
     - keep it for simple local/dev cases
     - do not continue expanding it as the main OpenAI compatibility layer
     - restrict gateway-local content/tool/reasoning interpretation to `local_best_effort`

5. **Keep `vllm_openai_proxy` as an external-upstream option**
   - Files:
     - [src/infer_nexus/runtime/executor.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/executor.py)
     - [src/infer_nexus/catalog/models.py](/F:/GitHub/infer-nexus/src/infer_nexus/catalog/models.py)
   - Change:
     - document and preserve `vllm_openai_proxy` as HTTP forwarding to `proxy_config.upstream_base_url`
     - do not describe it as a path to the Ray replica's internal vLLM engine
   - Use when:
     - the model should be served by an independently managed `vllm serve` process
     - HTTP-level passthrough to an external OpenAI-compatible upstream is operationally preferred

### Acceptance Criteria

- strict `backend: vllm` chat models use replica-local native vLLM OpenAI serving only
- strict `backend: vllm_openai_proxy` chat models proxy to an external OpenAI-compatible HTTP endpoint only
- local `LLM.chat(...)` is no longer the implicit default compatibility path
- strict request bodies and streaming chunks are not reinterpreted by gateway-local compatibility code


## Phase 3: Add Contract Tests

### Objective

Detect protocol drift in CI before it reaches test or production environments.

### Changes

1. **Add chat compatibility contract tests**
   - Files:
     - [tests/test_runtime.py](/F:/GitHub/infer-nexus/tests/test_runtime.py)
     - [tests/test_dispatcher.py](/F:/GitHub/infer-nexus/tests/test_dispatcher.py)
     - [tests/test_api.py](/F:/GitHub/infer-nexus/tests/test_api.py)
     - new file recommended:
       - `tests/test_openai_compat_contract.py`

2. **Cover request-shape compatibility cases**
   - string content
   - text-only content blocks
   - image content blocks
   - tool-calling fields
   - `tool_choice`
   - `parallel_tool_calls`
   - reasoning fields
   - stream and non-stream behavior
   - error code and error body shape

3. **Cover strict-vs-best-effort routing behavior**
   - strict `backend: vllm`:
     - uses native OpenAI serving adapter when available
     - fails when the adapter cannot initialize
     - fails when adapter invocation fails
     - never falls back to local `LLM.chat(...)`
   - `local_best_effort`:
     - may fall back to local `LLM.chat(...)`
     - keeps documented subset behavior
   - `vllm_openai_proxy`:
     - forwards to external `proxy_config.upstream_base_url`
     - is not coupled to Ray replica-local vLLM state

4. **Add comparison tests against upstream behavior**
   - compare gateway behavior to:
     - replica-local native `vLLM` OpenAI serving adapter
     - or a controlled external upstream proxy target
   - focus on:
     - payload preservation
     - response shape
     - SSE chunk shape
     - error classification

### Acceptance Criteria

- compatibility-sensitive request forms are covered by tests
- `vLLM` upgrades cannot pass CI if request/response behavior drifts
- stream and non-stream behavior are validated separately


## Files Most Likely to Change

### Core runtime path

- [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
- [src/infer_nexus/runtime/executor.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/executor.py)
- [src/infer_nexus/runtime/dispatcher.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/dispatcher.py)
- [src/infer_nexus/runtime/deployments.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/deployments.py)
- [src/infer_nexus/runtime/serve_app.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/serve_app.py)

### Configuration and schema

- [src/infer_nexus/catalog/models.py](/F:/GitHub/infer-nexus/src/infer_nexus/catalog/models.py)
- [src/infer_nexus/core/enums.py](/F:/GitHub/infer-nexus/src/infer_nexus/core/enums.py)
- optionally [src/infer_nexus/core/schemas.py](/F:/GitHub/infer-nexus/src/infer_nexus/core/schemas.py), if `compat_mode` should be exposed in public model metadata

### API surface and error mapping

- [src/infer_nexus/api/openai_routes.py](/F:/GitHub/infer-nexus/src/infer_nexus/api/openai_routes.py)

### Tests

- [tests/test_runtime.py](/F:/GitHub/infer-nexus/tests/test_runtime.py)
- [tests/test_dispatcher.py](/F:/GitHub/infer-nexus/tests/test_dispatcher.py)
- [tests/test_api.py](/F:/GitHub/infer-nexus/tests/test_api.py)
- [tests/test_multimodal_contract.py](/F:/GitHub/infer-nexus/tests/test_multimodal_contract.py)


## Recommended Short-Term Execution Order

1. add `compat_mode` to model config
2. pass `compat_mode` into `runtime_spec` / runtime context
3. require `openai_serving.enabled=true` for strict local `backend: vllm` chat models
4. disable silent fallback for strict models
5. improve adapter init logging and startup validation
6. ensure strict chat payloads go through native serving with minimal mutation
7. ensure strict streaming chunks are passed through without local delta reconstruction
8. add high-signal contract tests


## Implementation Checklist

### Configuration Model

- [x] [src/infer_nexus/core/enums.py](/F:/GitHub/infer-nexus/src/infer_nexus/core/enums.py)
  - add `CompatibilityMode`
  - values:
    - `STRICT_OPENAI = "strict_openai"`
    - `LOCAL_BEST_EFFORT = "local_best_effort"`

- [x] [src/infer_nexus/catalog/models.py](/F:/GitHub/infer-nexus/src/infer_nexus/catalog/models.py)
  - import `CompatibilityMode`
  - add `ModelConfig.compat_mode`
  - default to `CompatibilityMode.LOCAL_BEST_EFFORT` to preserve existing behavior
  - validate that local strict chat models use:
    - `backend: vllm`
    - `task: chat`
    - `vllm.openai_serving.enabled: true`
  - keep `vllm_openai_proxy` validation tied to `proxy_config`

### Runtime Context Propagation

- [x] [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
  - update `VLLMBackend.build_runtime_spec(...)`
  - include `compat_mode` in the returned `runtime_spec`

- [x] [src/infer_nexus/runtime/dispatcher.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/dispatcher.py)
  - include `compat_mode` in the `vllm_openai_proxy` runtime context
  - keep proxy behavior as external HTTP forwarding to `proxy_config.upstream_base_url`

- [x] [src/infer_nexus/runtime/serve_app.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/serve_app.py)
  - confirm `build_runtime_context(...)` carries the updated `runtime_spec`
  - no separate HTTP server should be started inside the Ray replica

### Strict Mode Helpers

- [x] [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
  - add `_compat_mode(runtime_spec: dict[str, Any] | None = None) -> str`
  - add `_is_strict_openai(runtime_spec: dict[str, Any] | None = None) -> bool`
  - add `_requires_openai_serving_adapter(runtime_spec: dict[str, Any] | None = None) -> bool`
  - add `_raise_openai_serving_unavailable(reason: str | None = None) -> None`
  - ensure helper behavior is based on `runtime_spec["compat_mode"]`, not only `openai_serving.enabled`

### Startup Behavior

- [x] [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
  - update `startup()`
  - after async engine startup, if strict local chat requires OpenAI serving and adapter is missing, raise `BackendConfigurationError`
  - after sync fallback engine startup, apply the same strict adapter readiness check
  - preserve existing non-strict warning behavior for `local_best_effort`
  - preserve stub mode behavior for local tests unless a later decision explicitly disallows strict stubs

- [x] [src/infer_nexus/runtime/deployments.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/deployments.py)
  - confirm `ModelRuntimeReplica.__init__(...)` lets backend startup failures surface
  - do not catch strict adapter startup failures and convert them into ready-but-degraded replicas

### Strict Chat Request Path

- [x] [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
  - update `_build_openai_serving_request_payload(...)`
  - preserve OpenAI request fields from `ChatCompletionsRequest.model_dump(...)`
  - rewrite only `model` to `served_model_name`
  - merge `request.extra_body` into the top-level payload
  - avoid collapsing text-only content blocks in strict mode
  - avoid rejecting multimodal content in strict mode when the schema accepted it
  - leave tools, `tool_choice`, reasoning fields, response format, stream options, and vLLM extension fields for vLLM serving to interpret

- [x] [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
  - update `chat_completion(...)`
  - if strict and adapter exists, call `_call_openai_serving_chat_completion(...)`
  - if strict and adapter is missing, raise an explicit backend configuration/request error
  - if strict adapter invocation fails, raise the error and do not clear the adapter for fallback
  - preserve current fallback path for `local_best_effort`

### Strict Streaming Path

- [x] [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
  - update `chat_completion_stream(...)`
  - if strict and adapter exists, yield `_iter_openai_serving_stream(...)`
  - if strict and adapter is missing, raise an explicit backend configuration/request error
  - if strict adapter streaming fails, raise the error and do not fallback to `_iter_chat_deltas(...)`
  - preserve `_iter_chat_deltas(...)` and `_iter_chat_completion_deltas(...)` for `local_best_effort`

- [x] [src/infer_nexus/runtime/executor.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/executor.py)
  - confirm bytes/string chunks from strict native serving are passed through as SSE without body mutation
  - confirm dict chunks are framed as SSE without changing chunk fields
  - avoid local delta reconstruction for strict chunks

### Error Visibility

- [x] [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
  - log adapter initialization failures with model name, served model name, engine kind, and concrete exception message
  - store `openai_serving_adapter_init_error` for request-time diagnostics
  - include enough detail in strict adapter unavailable errors to distinguish:
    - import layout unsupported
    - no compatible engine client
    - adapter constructor failure
    - adapter invocation failure

- [ ] [src/infer_nexus/api/openai_routes.py](/F:/GitHub/infer-nexus/src/infer_nexus/api/openai_routes.py)
  - only update if current error mapping hides strict adapter failure details too aggressively
  - keep OpenAI-style error response shape

### Tests

- [x] [tests/test_runtime.py](/F:/GitHub/infer-nexus/tests/test_runtime.py)
  - model config accepts default `local_best_effort`
  - model config accepts explicit `strict_openai`
  - strict local `backend: vllm` chat model requires `vllm.openai_serving.enabled=true`
  - `build_runtime_spec(...)` includes `compat_mode`
  - strict startup fails when OpenAI serving adapter init fails
  - strict `chat_completion(...)` fails when adapter is missing
  - strict `chat_completion(...)` does not fallback when adapter invocation fails
  - strict `chat_completion_stream(...)` does not fallback when adapter streaming fails
  - `local_best_effort` keeps current fallback behavior
  - strict OpenAI serving payload preserves text content blocks, image blocks, tools, reasoning fields, stream options, and extra body fields

- [x] [tests/test_dispatcher.py](/F:/GitHub/infer-nexus/tests/test_dispatcher.py)
  - proxy runtime context includes `compat_mode`
  - proxy dispatch still uses external `proxy_config.upstream_base_url`
  - strict/local metadata does not change backend selection semantics

- [x] [tests/test_api.py](/F:/GitHub/infer-nexus/tests/test_api.py)
  - add only if API-level error response behavior changes
  - preserve OpenAI-style error body for strict adapter failures

- [x] [tests/test_multimodal_contract.py](/F:/GitHub/infer-nexus/tests/test_multimodal_contract.py)
  - adjust expectations so strict mode payload preservation is tested separately from local best-effort multimodal validation

- [ ] `tests/test_openai_compat_contract.py`
  - dedicated end-to-end compatibility contract coverage is still missing as a separate test module

- [ ] CI/upstream comparison coverage
  - comparison tests against replica-local native serving or a controlled external upstream are not fully closed yet


## First Deliverable Scope

The smallest useful first delivery is:

1. `compat_mode` in model config
2. `compat_mode` propagated into `VLLMBackend.build_runtime_spec(...)`
3. fail-fast behavior for strict local `backend: vllm` models when native serving adapter init fails
4. no silent fallback for strict chat models when adapter calls fail
5. strict request payload construction that preserves OpenAI fields and rewrites only `model`
6. 6-8 contract tests covering:
   - string content
   - text-only block content
   - image block content
   - tool-calling
   - reasoning
   - streaming shape
   - adapter unavailable behavior
   - local best-effort fallback behavior

This does not fully finish the architecture shift, but it removes the most dangerous class of hidden compatibility drift.


## Non-Goals

These items should not be mixed into the first refactor pass:

- broad unrelated cleanup in runtime modules
- changing embedding or rerank paths unless compatibility issues require it
- redesigning all configuration structures at once
- rewriting every local fallback path before strict-mode guardrails are in place
- making Ray replicas host separate HTTP servers
- treating `vllm_openai_proxy` as a path to the replica-local vLLM engine


## Final Success Criteria

This refactor is successful when:

- chat compatibility behavior no longer depends primarily on gateway-local protocol recreation
- strict local `backend: vllm` models behave through replica-local vLLM OpenAI serving or fail clearly
- `vllm_openai_proxy` remains clearly scoped to external OpenAI-compatible HTTP upstreams
- SDK clients can use modern OpenAI request shapes without unexpected gateway-specific breakage
- `vLLM` version upgrades become test-driven instead of incident-driven

Current close-out assessment:

- The primary refactor objective has been achieved for the currently deployed model set.
- The strict local `qwen3.5-9b` path now behaves as an OpenAI-compatible endpoint for the key scenarios validated in this plan.
- Remaining work is test-hardening and future-proofing, not a blocker for considering the current compatibility refactor deployment complete.
