# vLLM OpenAI Serving on Ray Serve Plan

## 1. Goal

This document describes the planned implementation for making `infer-nexus`
behave like a vLLM OpenAI-compatible service while keeping the current Ray
Serve based runtime architecture.

The target user experience is:

- Clients use one OpenAI-compatible base URL exposed by `infer-nexus`.
- Clients select models through the standard OpenAI `model` request field.
- Existing users who directly call a vLLM OpenAI-compatible server should be
  able to switch to `infer-nexus` with minimal or no request/response changes.
- Chat streaming, vLLM-specific request parameters, reasoning fields, tool
  calling fields, logprobs, and other OpenAI-compatible response details should
  be preserved as close to native vLLM behavior as possible.

The target runtime architecture remains:

```text
Client
  -> Single FastAPI Gateway
    -> model routing by request.body.model
      -> Ray Serve model deployment
        -> one of N equal Ray Serve replicas
          -> replica-local vLLM instance / engine
```

Each model can have multiple Ray Serve replicas. Replicas are peer workers owned
and scaled by Ray Serve. Each replica owns its own local vLLM runtime instance.

## 2. Non-Goals

This plan does not change the main runtime architecture to standalone
`vllm serve` processes behind HTTP proxying.

This plan does not implement every vLLM HTTP endpoint. The first implementation
focuses on:

- `/v1/chat/completions`
- native streaming behavior for chat
- vLLM/OpenAI-compatible chat request and response protocol fidelity

The following are explicitly out of scope for this implementation pass:

- `/v1/completions`
- `/v1/responses`
- `/tokenize`
- `/detokenize`
- `/pooling`
- `/score`
- additional vLLM control endpoints

Embedding and rerank protocol parity are future work. They are discussed in
section 11.

## 3. Current Architecture Summary

The current local `vllm` backend path is closest to:

```text
FastAPI route
  -> RuntimeDispatcher
    -> RuntimeExecutor
      -> Ray Serve deployment handle or local stub replica
        -> ModelRuntimeReplica
          -> VLLMBackend
            -> vllm.LLM.chat / embed / score
          -> internal payload
      -> RuntimeExecutor rebuilds OpenAI-style response
```

This is sufficient for basic local inference, but it is not equivalent to
vLLM's OpenAI-compatible server.

The main chat gaps are:

- `stream=True` is currently converted to `stream=False` before backend
  execution.
- Streaming responses are currently synthesized by the gateway after full
  generation has completed.
- Non-streaming chat responses are rebuilt into a narrow response shape with one
  choice and a small set of fields.
- vLLM-native OpenAI server behavior for reasoning, tool calls, logprobs,
  structured output, guided decoding, stream usage, and extra parameters is not
  fully preserved.

The existing `vllm_openai_proxy` backend already supports HTTP passthrough and
raw SSE proxying for upstream OpenAI-compatible services. That backend is useful
for compatibility-sensitive external or standalone upstreams, but it is not the
main path for Ray Serve owned local model replicas.

## 4. Why `vllm.LLM.chat()` Is Not Enough

`vllm.LLM.chat()` is a programmatic/offline inference API. It executes model
inference and returns vLLM internal result objects. It does not by itself expose
the same HTTP protocol behavior as vLLM's OpenAI-compatible server.

vLLM's OpenAI-compatible server includes additional serving-layer behavior:

- OpenAI request parsing
- vLLM extra parameter handling
- chat template handling
- tool calling request and response handling
- reasoning parser integration
- response format and guided decoding integration
- streaming chunk generation
- logprobs formatting
- usage formatting
- finish reason mapping
- error response behavior

If `infer-nexus` calls only `LLM.chat()` and then reconstructs responses itself,
the external protocol will depend on the local adapter implementation. That
causes drift from native vLLM server behavior.

The target implementation should therefore reuse vLLM's OpenAI-compatible
serving logic where practical, while still running inside Ray Serve replicas.

## 5. Target Design

The target design keeps Ray Serve as the runtime and autoscaling layer, but
changes the local chat execution path from:

```text
vllm.LLM.chat()
  -> infer-nexus custom response adapter
```

to:

```text
replica-local vLLM async engine
  -> vLLM OpenAI-compatible serving adapter
  -> OpenAI-compatible response or streaming chunks
```

Conceptually:

```text
Client
  -> FastAPI /v1/chat/completions
    -> infer-nexus model lookup and admission
      -> Ray Serve deployment handle
        -> selected replica
          -> VLLMBackend OpenAI serving adapter
            -> vLLM async engine
          -> OpenAI-compatible payload or chunks
    -> FastAPI returns JSON or StreamingResponse
```

The gateway should continue to own:

- authentication and future authorization
- model catalog lookup
- model alias / served model routing
- task validation
- admission control
- request ID / trace headers
- high-level error mapping for gateway-stage failures

The vLLM serving adapter should own:

- chat protocol semantics
- vLLM request parameter interpretation
- native streaming chunks
- reasoning/tool/logprob/usage field details

## 6. Implementation Areas

### 6.1 vLLM Version and Serving API Discovery

Before implementation, inspect the actual vLLM version used by the project and
identify the compatible internal serving classes.

Expected vLLM concepts may include:

- async engine or engine client
- OpenAI serving models registry
- chat serving adapter
- request models for chat completion
- streaming response generator types

Exact import paths should not be assumed without checking the installed vLLM
version. vLLM OpenAI serving internals may change across versions, so the
integration should be isolated behind a small local adapter in `VLLMBackend`.

### 6.2 Backend Initialization

`VLLMBackend.startup()` should initialize chat models differently from embedding
or rerank models.

For `task_mode == "generate"`:

- initialize the vLLM runtime required for online chat serving
- initialize or wrap the vLLM OpenAI-compatible chat serving adapter
- keep existing runtime spec fields such as model path, dtype,
  tensor parallel size, max model length, GPU memory utilization, and engine
  kwargs
- preserve Qwen3 reasoning configuration:
  - `enable_reasoning`
  - `reasoning_parser`

For `task_mode == "embed"` and `task_mode == "score"`:

- keep current behavior for this implementation pass
- do not force these tasks through the chat serving adapter

This prevents chat-specific changes from breaking embedding and rerank models.

### 6.3 Request Flow for Non-Streaming Chat

For `stream=False`:

1. FastAPI route validates the top-level request enough to find `model`.
2. Runtime dispatch resolves the model to a Ray Serve deployment.
3. The selected replica passes the request to the vLLM OpenAI serving adapter.
4. The adapter returns an OpenAI-compatible response payload.
5. `RuntimeExecutor` returns that payload without narrowing it to the current
   internal minimal schema.

The current `_build_chat_response_from_payload()` behavior should remain only as
a fallback for stub/dev paths or legacy backend payloads.

### 6.4 Request Flow for Streaming Chat

For `stream=True`:

1. FastAPI route dispatches the request as a streaming request.
2. `RuntimeExecutor` must not rewrite the request to `stream=False`.
3. `RuntimeExecutor` calls a streaming method on the Ray Serve deployment.
4. The selected replica calls `VLLMBackend.chat_completion_stream(...)`.
5. The backend yields OpenAI-compatible chunk dictionaries or raw SSE bytes.
6. `RuntimeExecutor` returns a `StreamingResponse`.

The gateway should not wait for full generation before returning the first
chunk.

The desired wire format is the native OpenAI SSE style:

```text
data: {"id":"...","object":"chat.completion.chunk","choices":[...]}

data: {"id":"...","object":"chat.completion.chunk","choices":[...]}

data: [DONE]
```

The gateway can perform SSE encoding when the backend yields dictionaries, but
it should not rewrite the semantic content of chunks.

### 6.5 RuntimeExecutor Changes

`RuntimeExecutor.execute_chat()` should split chat execution by stream mode:

- `stream=False`: call the existing non-streaming method.
- `stream=True`: call a new streaming path and return `StreamingResponse`.

The current behavior below should be removed for the Ray Serve local vLLM path:

```python
request.model_copy(update={"stream": False})
```

The executor should support a passthrough contract:

- If backend returns a full OpenAI-compatible response payload, return it as-is.
- If backend yields OpenAI-compatible chunks, encode them as SSE without
  changing their fields.
- If backend yields raw bytes, forward them as-is.
- If backend returns the old minimal payload, use the existing fallback adapter.

### 6.6 ModelRuntimeReplica Changes

`ModelRuntimeReplica` should expose a streaming chat method, for example:

```text
chat_completion_stream(payload: dict) -> async iterator
```

Responsibilities:

- validate or normalize the incoming payload into the internal request object
- call the backend streaming method
- yield backend chunks without rebuilding them

The implementation must be compatible with Ray Serve's streaming response
support. If the current Ray Serve version has specific streaming handle
requirements, the implementation should follow that version's API.

### 6.7 VLLMBackend Changes

`VLLMBackend` should add a chat streaming method:

```text
chat_completion_stream(runtime_spec, request, runtime_context)
```

For chat models using the new serving integration, this method should delegate to
vLLM's OpenAI-compatible chat serving layer.

For stub/dev paths, it may yield deterministic OpenAI-compatible test chunks.

The backend should also support non-streaming passthrough:

```text
chat_completion(...) -> full OpenAI-compatible response payload
```

When possible, the backend should avoid converting vLLM results into a reduced
custom shape. The goal is to preserve native fields.

### 6.8 Schema Changes

The current request and response schemas should become less lossy.

Request-side additions or passthrough support should include:

- `stream_options`
- `max_completion_tokens`
- `parallel_tool_calls`
- vLLM extra parameters such as `top_k`, `min_p`, `min_tokens`,
  `stop_token_ids`, `include_stop_str_in_output`, `skip_special_tokens`,
  `spaces_between_special_tokens`, `truncate_prompt_tokens`,
  `prompt_logprobs`, and guided decoding / structured output fields

Response-side schema should support or avoid rejecting:

- multiple choices
- `tool_calls`
- `function_call`
- `reasoning_content`
- `reasoning`
- `logprobs`
- stream usage fields
- additional usage subfields
- vLLM-specific extension fields

Where possible, gateway response validation should not strip fields produced by
the vLLM serving adapter.

### 6.9 Request Parameter Policy

Per-model request policy should remain available.

Existing policy concepts:

- allow tool-calling fields only when configured
- allow reasoning fields only when configured
- optionally allow unknown OpenAI/vLLM fields to pass through

The default behavior can remain conservative, but models intended to mimic
native vLLM OpenAI serving should enable passthrough for vLLM-compatible fields.

## 7. Compatibility Contract

The chat endpoint should aim to match native vLLM OpenAI-compatible behavior for
common client usage:

```text
POST /v1/chat/completions
```

The standard model selector is:

```json
{
  "model": "qwen3-chat"
}
```

The gateway should not introduce a custom `model_name` field.

Clients should be able to keep using OpenAI SDK style calls:

```python
client.chat.completions.create(
    model="qwen3-chat",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
    extra_body={"top_k": 50},
)
```

Expected behavior:

- `model` is resolved by `infer-nexus` catalog.
- request body is passed to the selected Ray Serve model deployment.
- vLLM-specific fields are preserved and interpreted by the vLLM serving layer.
- streaming chunks preserve vLLM/OpenAI-compatible fields.

## 8. Error Handling Policy

Gateway-stage errors should continue to be produced by `infer-nexus`.

Examples:

- unknown model
- unsupported task type
- missing local artifact
- admission rejected
- Ray Serve deployment unavailable
- backend misconfiguration

Backend protocol errors produced by vLLM's serving adapter should be preserved as
much as possible. The gateway should avoid converting all backend failures into a
generic internal error when the vLLM serving layer already produced a structured
OpenAI-compatible error payload.

## 9. Observability Requirements

The implementation should preserve or add request-level observability:

- gateway request ID
- model name / served model name
- Ray Serve app and deployment name
- stream start and completion logging
- stream cancellation logging
- backend execution errors
- time to first chunk where practical
- total stream duration

The gateway should not need to parse every SSE chunk for observability. Metrics
can be collected at stream lifecycle boundaries first.

## 10. Test Plan

The first implementation should include tests for:

- `stream=True` is not rewritten to `stream=False`.
- streaming path calls a Ray Serve replica streaming method.
- streaming response yields multiple chunks before `[DONE]`.
- raw bytes chunks are forwarded unchanged.
- chunk dictionaries are encoded as OpenAI-style SSE.
- reasoning delta fields are preserved.
- tool call fields are preserved.
- logprobs fields are preserved when present.
- `n > 1` choices are not collapsed to a single choice.
- `extra_body` vLLM parameters reach the backend.
- non-streaming OpenAI-compatible payloads are returned without field stripping.
- current stub/dev fallback behavior remains available.
- existing embedding tests still pass.
- existing rerank tests still pass.
- existing `vllm_openai_proxy` passthrough behavior is not regressed.

## 11. Future Work: Embedding and Rerank Serving Parity

The current implementation pass focuses on chat because chat has the largest
protocol surface and requires native streaming.

Embedding and rerank should be handled in a later implementation pass.

vLLM also has OpenAI-compatible or OpenAI-adjacent serving components for these
tasks:

- embeddings: vLLM OpenAI embedding serving
- rerank / score: vLLM serving score / rerank handling
- pooling: vLLM pooling serving for pooling models

Future work should evaluate replacing or supplementing the current direct
`engine.embed(...)` and `engine.score(...)` paths with vLLM's corresponding
serving-layer implementations.

Future embedding goals:

- preserve native `/v1/embeddings` response fields
- preserve embedding-specific input handling
- preserve `encoding_format`
- support future vLLM embedding extra parameters without local response drift

Future rerank goals:

- support vLLM-compatible `/rerank`, `/v1/rerank`, and `/v2/rerank` behavior
  where needed
- preserve compatibility with common Jina/Cohere-style rerank clients when
  vLLM supports those shapes
- avoid narrowing vLLM rerank responses to a custom-only schema when native
  compatibility is required

For this implementation pass, embedding and rerank should remain functionally
unchanged and covered by regression tests.

## 12. Incremental Delivery Plan

### Phase 1: Discovery and Adapter Boundary

- Identify the installed or target vLLM version.
- Inspect vLLM OpenAI serving internals for chat.
- Add a local adapter boundary in `VLLMBackend` so vLLM internals are isolated.
- Keep legacy `LLM.chat()` fallback for stub/dev and compatibility.

### Phase 2: Streaming Path Plumbing

- Add `chat_completion_stream` to `ModelRuntimeReplica`.
- Add streaming dispatch to `RuntimeExecutor`.
- Remove forced `stream=False` conversion for local Ray Serve vLLM chat.
- Add SSE encoding helpers that do not rewrite chunk semantics.

### Phase 3: vLLM Chat Serving Integration

- Initialize vLLM online serving components for chat models.
- Route non-streaming chat requests through the serving adapter.
- Route streaming chat requests through the serving adapter.
- Preserve OpenAI-compatible payloads and chunks.

### Phase 4: Schema and Policy Hardening

- Expand request and response schema support.
- Ensure vLLM extra parameters are preserved.
- Keep per-model policy controls explicit.
- Add compatibility tests for reasoning, tool calls, logprobs, and multi-choice.

### Phase 5: Regression and Rollout

- Run the existing API, dispatcher, runtime, embedding, rerank, and proxy tests.
- Validate against a real vLLM-compatible client.
- Validate with at least one reasoning chat model.
- Validate with multiple Ray Serve replicas for one model.
- Document operational caveats and required vLLM version constraints.

## 13. Acceptance Criteria

The implementation should be considered complete for this phase when:

- A client can call `infer-nexus /v1/chat/completions` with the same request
  shape used for direct vLLM OpenAI-compatible serving.
- `stream=True` produces real incremental SSE chunks.
- Gateway code no longer synthesizes streaming output from a completed
  non-streaming response for local Ray Serve vLLM chat.
- Reasoning chunks are preserved.
- Tool call fields are preserved when the model and policy allow them.
- vLLM extra parameters can be passed through in the supported configuration.
- Non-streaming response fields are not narrowed to the old minimal response
  shape when the backend provides a full OpenAI-compatible payload.
- Ray Serve remains responsible for model deployment replicas and autoscaling.
- Existing embedding and rerank behavior remains unchanged.
- Existing proxy backend passthrough behavior remains unchanged.

## 14. Current Implementation Status

Last updated: 2026-05-23.

Original target vLLM compatibility version confirmed by the project owner:

- vLLM `0.18.x`

Current local environment notes:

- The local machine is macOS arm64.
- The project `.venv` currently uses Python 3.13, which cannot install the
  PyPI `vllm-metal==0.1.0` wheel because that wheel is `cp312` only.
- A separate Python 3.12 virtual environment, `.venv-metal/`, was created for
  local vLLM Metal experiments and is ignored by git.
- Official Linux/CUDA vLLM dependencies still do not install in the macOS arm64
  project `.venv`; `uv sync --extra serve --extra vllm` fails on CUDA/NVIDIA
  wheel availability for this platform.
- `Qwen/Qwen3.5-0.8B` has been downloaded locally and configured as the main
  small chat smoke-test model with `max_model_len: 4096`.
- Development and verification should continue to use lightweight scripts, fake
  adapters, stub replicas, and narrow pytest coverage unless a real vLLM or
  vLLM Metal server is intentionally started.

### 14.1 Implemented

The following pieces have been implemented in the current working tree.

#### Streaming plumbing contract

- `InferenceBackend` now defines a `chat_completion_stream(...)` backend
  contract.
- `ModelRuntimeReplica` now exposes `chat_completion_stream(payload)`.
- `RuntimeExecutor.execute_chat(...)` now branches on `request.stream`.
- For local Ray Serve vLLM chat, `stream=True` is no longer rewritten to
  `stream=False`.
- `RuntimeExecutor` now has a dedicated streaming invocation path for both:
  - local stub/dev replica execution
  - Serve deployment handle execution
- Streaming invocation normalizes several return shapes into an async iterator:
  - async iterators
  - awaitables
  - Ray-like `.result()` objects
  - raw `bytes`
  - raw `str`
  - dictionaries
  - ordinary iterables

Primary files:

- `src/infer_nexus/backends/base.py`
- `src/infer_nexus/runtime/deployments.py`
- `src/infer_nexus/runtime/executor.py`
- `src/infer_nexus/backends/vllm.py`

#### SSE handling

- Gateway-side SSE encoding now wraps dictionary chunks as:

```text
data: {...}

```

- Raw SSE bytes are forwarded unchanged.
- Raw SSE strings are encoded to UTF-8 and forwarded.
- The gateway appends `data: [DONE]` only when the incoming stream did not
  already include it.
- Streaming chunk semantic fields are not rebuilt or narrowed when the backend
  provides dictionary chunks.

Covered behavior:

- reasoning delta fields are preserved in stream chunks
- raw SSE bytes are preserved
- proxy streaming passthrough behavior remains covered

#### Stub streaming

- `VLLMBackend.chat_completion_stream(...)` now supports the stub/dev path.
- In stub mode, the backend yields deterministic OpenAI-style
  `chat.completion.chunk` dictionaries.
- This enables testing executor/replica/gateway streaming behavior without
  starting a vLLM model.

Important limitation:

- When a real legacy `vllm.LLM` engine is initialized, streaming currently
  raises a validation error stating that the vLLM OpenAI serving adapter is not
  initialized yet. This is intentional until Phase 3 is implemented.

#### Non-streaming OpenAI payload passthrough

- `RuntimeExecutor` detects a full OpenAI chat payload when the backend returns:

```json
{
  "object": "chat.completion",
  "choices": []
}
```

- Such payloads are returned as a `JSONResponse` without narrowing to the old
  single-choice `ChatCompletionsResponse` adapter.
- `ModelRuntimeReplica.chat_completion(...)` also avoids adding the internal
  `"status": "ok"` wrapper when a full OpenAI chat payload is returned by the
  backend.

This is intended to preserve:

- multiple choices
- extended usage fields
- native vLLM/OpenAI response fields

#### Request schema widening

`ChatCompletionsRequest` and `ChatMessage` now accept more OpenAI/vLLM-style
fields without dropping them during validation.

Added or widened request/message fields include:

- `max_completion_tokens`
- `stream_options`
- `parallel_tool_calls`
- message `reasoning_content`
- message `reasoning`
- message `tool_calls`
- message `function_call`
- nullable message `content`
- extra fields on `ChatMessage`

`VLLMBackend` now also maps:

- `max_completion_tokens` into the local sampling parameter fallback path
- request-level `parallel_tool_calls` into chat kwargs when tools are allowed
- assistant message `tool_calls` and `function_call` into serialized messages

#### OpenAI serving runtime spec config

`VLLMBackend.build_runtime_spec(...)` now preserves the catalog
`vllm.openai_serving` config in the runtime spec.

When `vllm.openai_serving.enabled` is true, the following config is mirrored
into engine kwargs when present:

- `enable_reasoning`
- `reasoning_parser`

This was added to preserve Qwen reasoning configuration for the future vLLM
OpenAI serving adapter integration.

#### OpenAI serving adapter boundary and passthrough contract

`VLLMBackend` now has an explicit local adapter boundary for future native
vLLM OpenAI serving integration.

Implemented pieces:

- `OpenAIChatServingAdapter` protocol was added under the local vLLM backend.
- `VLLMBackend` now stores an `openai_serving_chat_adapter` handle.
- Chat requests now prefer the adapter path when that adapter is present.
- A dedicated request payload builder now serializes an OpenAI-style request
  body for serving passthrough instead of rebuilding the old local
  `LLM.chat()` payload shape.
- That request payload builder preserves:
  - top-level `extra_body` fields as OpenAI/vLLM request keys
  - `stream`
  - `stream_options`
  - nullable assistant `content`
  - assistant `tool_calls`
  - `parallel_tool_calls`
- Non-streaming adapter responses are returned as-is to the existing executor
  passthrough path.
- Streaming adapter chunks are yielded as-is and therefore feed directly into
  the existing SSE encoding/forwarding path.

Current limitation:

- `_initialize_openai_serving_chat_adapter()` now performs real dynamic import
  probing and constructor attempts for multiple candidate vLLM OpenAI serving
  layouts.
- The adapter initializer now records
  `openai_serving_adapter_init_error` when native import or construction fails.
- When no adapter is present, real-engine non-streaming still falls back to
  legacy `LLM.chat()`.
- When no adapter is present, real-engine streaming still raises the current
  intentional `backend_misconfigured` validation error, now including the
  adapter initialization failure reason when available.

#### Proxy request serialization

Proxy requests now use `exclude_none=True` when dumping Pydantic request models.

This prevents newly added nullable schema fields from being forwarded as explicit
`null` values to upstream OpenAI-compatible servers.

#### Local Qwen3.5 configuration

`config/models.yaml` has been changed for the current macOS development
environment:

- Enabled local model:
  - name: `Qwen3.5-0.8B`
  - alias: `qwen3.5-0.8b`
  - backend: `vllm`
  - model path:
    `/Users/zhoushujian/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17`
  - `vllm.openai_serving.enabled: true`
  - `vllm.openai_serving.enable_reasoning: true`
  - `vllm.openai_serving.reasoning_parser: qwen3`
  - default `chat_template_kwargs.enable_thinking: false`
  - `request_policy.allow_reasoning: true`
  - `request_policy.passthrough_unknown_openai_fields: true`

- Enabled local proxy smoke-test model:
  - name: `Qwen3.5-0.8B-Metal`
  - alias: `qwen3.5-0.8b-metal`
  - backend: `vllm_openai_proxy`
  - upstream: `http://127.0.0.1:18000/v1`
  - `require_local_artifacts: false`

- Production-only `/nas_data/...` model entries are commented out because those
  paths do not exist on this machine.

`config/settings.yaml` is currently set to:

- `runtime.execution_mode: stub`
- `runtime.backend_init_mode: stub`

This keeps normal local API startup independent of Ray Serve and real vLLM
engine initialization.

### 14.2 Verified

Narrow verification was run successfully with the local virtual environment and
without starting real vLLM models.

Command shape used:

```bash
UV_NO_CONFIG=1 uv run \
  --python /Users/zhoushujian/Projects/GitHub/infer-nexus/.venv/bin/python \
  --no-project \
  -m pytest ...
```

Passing targeted coverage:

- `tests/test_dispatcher.py::test_chat_stream_response_preserves_vllm_reasoning_deltas`
- `tests/test_dispatcher.py::test_chat_stream_response_forwards_raw_sse_bytes_unchanged`
- `tests/test_dispatcher.py::test_proxy_chat_dispatch_rewrites_only_model_and_passthrough_response`
- `tests/test_runtime.py::test_vllm_backend_accepts_streaming_messages_and_rejects_text_only_multimodal`
- `tests/test_runtime.py::test_vllm_backend_builds_chat_kwargs_for_tools_and_reasoning`
- `tests/test_runtime.py::test_vllm_backend_sampling_params_include_official_compatible_fields`
- `tests/test_runtime.py::test_vllm_backend_build_runtime_spec_includes_openai_serving_reasoning_config`
- `tests/test_proxy_streaming.py`

Result:

```text
8 passed
```

Additional targeted adapter verification performed on 2026-05-23:

- `tests/test_runtime.py::test_vllm_backend_chat_completion_passthroughs_openai_serving_payload`
- `tests/test_runtime.py::test_vllm_backend_chat_completion_stream_passthroughs_openai_serving_chunks`
- `tests/test_dispatcher.py::test_chat_stream_response_preserves_vllm_reasoning_deltas`
- `tests/test_dispatcher.py::test_chat_stream_response_forwards_raw_sse_bytes_unchanged`

Result:

```text
4 passed
```

Verified behavior from this round:

- A fake OpenAI serving adapter can now drive `VLLMBackend.chat_completion(...)`
  and return a full OpenAI-compatible payload without local narrowing.
- A fake OpenAI serving adapter can now drive
  `VLLMBackend.chat_completion_stream(...)` and yield OpenAI-compatible chunks
  or raw SSE bytes without semantic rewriting.
- Assistant messages with `content: null` and `tool_calls` are now preserved in
  the serving request payload builder, which is required for OpenAI/vLLM tool
  call compatibility.

Additional targeted adapter-constructor verification performed on 2026-05-23:

- `tests/test_runtime.py::test_vllm_backend_initializes_openai_serving_adapter_via_dynamic_imports`
- `tests/test_runtime.py::test_vllm_backend_openai_serving_adapter_init_reports_import_failure`
- `tests/test_runtime.py::test_vllm_backend_chat_completion_passthroughs_openai_serving_payload`
- `tests/test_runtime.py::test_vllm_backend_chat_completion_stream_passthroughs_openai_serving_chunks`
- `tests/test_dispatcher.py::test_chat_stream_response_preserves_vllm_reasoning_deltas`
- `tests/test_dispatcher.py::test_chat_stream_response_forwards_raw_sse_bytes_unchanged`

Result:

```text
6 passed
```

Verified behavior from this round:

- `VLLMBackend._initialize_openai_serving_chat_adapter()` no longer uses a
  placeholder; it now probes multiple candidate vLLM serving module layouts via
  dynamic import.
- The backend can now construct a native-style local serving adapter from fake
  upstream classes using a real constructor path:
  - `ChatCompletionRequest`
  - `OpenAIServingChat`
  - `OpenAIServingModels`
  - `BaseModelPath`
  - `OpenAIServingRender`
- The dynamic adapter wrapper now normalizes native Pydantic-style responses
  into dict payloads and forwards async stream chunks without semantic
  rewriting.
- Import/construction failures are now preserved as
  `openai_serving_adapter_init_error` for later surfacing in runtime failures.

Additional local smoke verification performed on 2026-05-23:

- Catalog loading resolves `qwen3.5-0.8b` to the local Hugging Face snapshot.
- Runtime context building produces:
  - app name: `infer-nexus-model-Qwen3.5-0.8B`
  - deployment name: `model-Qwen3.5-0.8B`
  - `openai_serving` config with Qwen reasoning parser preserved
- With `execution_mode: stub` and `backend_init_mode: stub`:
  - `/v1/models` returns `qwen3.5-0.8b`
  - non-streaming `/v1/chat/completions` for `qwen3.5-0.8b` returns `200`
  - streaming `/v1/chat/completions` for `qwen3.5-0.8b` returns
    `text/event-stream`
  - stub streaming output is OpenAI-style SSE and ends with `data: [DONE]`

Real vLLM Metal proxy verification performed on 2026-05-23:

- `vllm-metal==0.1.0` was installed successfully into `.venv-metal/`, a
  Python 3.12 environment.
- A direct in-process probe using `vllm_metal.server.create_engine(...)` loaded
  the local Qwen3.5 snapshot when run outside the sandbox with Metal access.
- A direct generation probe returned:

```text
Hello! How can I help you today
```

- A local `vllm_metal.server` process was started on `127.0.0.1:18000`.
- Direct request to `http://127.0.0.1:18000/v1/chat/completions` succeeded.
- infer-nexus proxy request through model alias `qwen3.5-0.8b-metal` succeeded
  and returned a real model OpenAI-style non-streaming response.
- The test server was stopped after verification.

Important vLLM Metal streaming finding:

- The installed PyPI `vllm-metal==0.1.0` package does not expose a `vllm`
  package or `from vllm import LLM` compatibility surface.
- Its bundled `vllm_metal.server` accepts a `stream` field in the request model
  but ignores it for the HTTP response path.
- `vllm_metal.server` returns `application/json` for `stream: true`, not
  `text/event-stream`.
- Internally, `MetalModelRunner.generate(...)` uses `mlx_lm.stream_generate`,
  but it accumulates the segments and returns a complete string.
- Therefore `vllm_metal.server` from PyPI `0.1.0` is useful for real
  non-streaming proxy validation, but it is not a valid streaming parity test
  target.
- The newer upstream `vllm-metal` installation path is different: the official
  install script installs vLLM core `0.21.0` plus the vLLM Metal plugin. That
  path may expose the official `vllm` CLI and OpenAI serving layer, and should
  be tested separately before drawing conclusions about the latest vLLM Metal
  project's streaming support.

### 14.3 Not Yet Implemented

The following parts of the original plan are still not implemented.

#### vLLM 0.18.x serving API discovery

- The project now has a small local adapter boundary in `VLLMBackend`, and that
  boundary is no longer a placeholder:
  - it probes multiple candidate vLLM serving import layouts
  - it attempts real constructor wiring for serving objects
  - it stores initialization failures for diagnosis
- Exact compatibility with production-targeted vLLM `0.18.x` is still not
  verified against a real installed package or runtime environment.

Additional note after vLLM Metal investigation:

- The PyPI `vllm-metal==0.1.0` package is not a direct replacement for the
  official `vllm` Python package.
- The official vLLM Metal install script now installs vLLM core `0.21.0` plus
  the Metal plugin. This may be a practical macOS path for inspecting a real
  `vllm` OpenAI serving surface, but it is no longer the originally targeted
  vLLM `0.18.x` environment.
- If the project continues to target production vLLM `0.18.x`, serving API
  discovery still needs to happen against that exact version.
- If local macOS validation becomes the priority, a separate discovery track
  should test official vLLM core `0.21.0` plus vLLM Metal plugin.
- An externally started `vllm serve` process on macOS may still be useful later
  as a reference surface for HTTP/SSE payload comparison, but it is not
  sufficient validation for the current goal. The main goal is replica-local
  integration of vLLM serving internals inside `infer-nexus` Ray Serve
  deployments, not proxying to a standalone vLLM server.

Expected next work:

- validate the current dynamic import candidates against real vLLM `0.18.x`
  source/install layout
- verify the exact constructor signatures and required engine-client objects in
  a real environment
- tighten the current heuristic engine-client selection once real-object
  inspection is possible

#### Real vLLM OpenAI serving adapter integration

The main Phase 3 work remains open.

Not yet implemented:

- validated replica-local vLLM async engine / engine-client initialization for
  serving against a real `vllm` package
- validated OpenAI serving model registry setup against a real `vllm` package
- validated chat serving adapter setup against a real `vllm` package
- real non-streaming chat routed through vLLM's OpenAI serving layer in a live
  environment
- real streaming chat routed through vLLM's OpenAI serving layer in a live
  environment
- preservation of native vLLM structured errors from the serving adapter in
  end-to-end runtime tests

Current state:

- legacy `LLM.chat()` remains the real-engine non-streaming fallback path
- real-engine streaming is intentionally blocked until the serving adapter is
  successfully initialized
- stub streaming is available for plumbing tests only
- fake adapter passthrough tests cover the backend contract shape
- fake native constructor tests now cover the backend dynamic import and object
  wiring path before real model startup is required

#### Full response schema parity

The gateway can now passthrough full OpenAI chat payloads as `JSONResponse`, but
the Pydantic `ChatCompletionsResponse` model itself is still narrow.

Still not fully modeled:

- multiple response choice variants beyond the current basic message type
- response `tool_calls`
- response `function_call`
- response `logprobs`
- additional usage subfields
- stream usage chunks
- vLLM extension fields

The likely preferred direction is to avoid validating native vLLM responses
through a narrowing response model whenever the serving adapter already returns
OpenAI-compatible payloads.

#### Full request parameter parity

Only part of the request surface has been widened.

Still incomplete:

- guided decoding fields
- structured output fields
- all vLLM extra parameters as explicit schema fields
- complete policy treatment for passthrough models
- native vLLM validation/error passthrough for unsupported request shapes

#### Observability

The following observability requirements are still not implemented:

- stream start logging
- stream completion logging
- stream cancellation logging
- backend stream execution error logging beyond existing executor errors
- time to first chunk
- total stream duration metrics

#### Real Ray Serve streaming verification

Streaming handle behavior is currently tested with fake Serve handles only.

Still needed:

- verify the implementation against the installed Ray Serve version
- confirm remote streaming handle return shape
- confirm cancellation behavior
- confirm multiple replicas still behave as peer workers

#### Real model/client validation

Partially done:

- The local Qwen3.5 snapshot exists and has been used for:
  - stub gateway tests through `qwen3.5-0.8b`
  - real non-streaming proxy tests through `qwen3.5-0.8b-metal`

Still needed once the local serving adapter is implemented:

- run `Qwen/Qwen3.5-0.8B` through the local Ray Serve `vllm` backend path
- test OpenAI SDK non-streaming request
- test OpenAI SDK streaming request
- test vLLM extra params such as `top_k`
- test Qwen reasoning configuration if supported by the model/config
- verify `stream=True` against a real backend that actually emits SSE chunks

Embedding and rerank native serving parity remain future work and are unchanged
from section 11.

### 14.4 Known Current Test/Repository Issues

Full test-suite results are currently not clean for reasons outside the Phase 2
streaming plumbing changes.

Known blockers:

- `uv run ...` currently prints a warning because this installed `uv` version
  does not recognize `tool.uv.extra-build-dependencies` in `pyproject.toml`.
- The current `config/models.yaml` is intentionally local-machine focused and
  now exposes `qwen3.5-0.8b` and `qwen3.5-0.8b-metal`, while many older tests
  still expect aliases such as:
  - `qwen3-chat`
  - `bge-embedding`
  - `bge-rerank`
- Embedding and rerank models are currently commented out in local config, so
  old broad API/runtime tests that expect those models will fail until fixtures
  are updated or a separate test catalog is introduced.
- `vllm-metal==0.1.0` is installed only in `.venv-metal/`, not in the project
  `.venv`.
- Attempting to install official vLLM core plus the Metal plugin into
  `.venv-metal/` did not yet produce a stable local validation environment in
  this round. The install path appears workable in principle, but the current
  environment hit a combination of long-running build/install behavior and
  network/proxy friction during source fetches.
- Accessing Metal from the default sandboxed command path failed with
  `No Metal device available`; the same model-load probe succeeded when run
  outside the sandbox.

Because of this, broad test failures are not currently a reliable signal for
the vLLM OpenAI serving work until config/test fixtures are reconciled.

### 14.5 Recommended Next Step

Continue with Phase 1 and Phase 3, but split production vLLM and macOS local
validation clearly:

1. Keep `qwen3.5-0.8b` as the local small-model catalog entry for future
   Ray Serve local backend tests.
2. Keep `qwen3.5-0.8b-metal` as a real non-streaming proxy smoke-test target
   when a local `vllm_metal.server` is running on `127.0.0.1:18000`.
3. Do not use PyPI `vllm-metal==0.1.0` `vllm_metal.server` as evidence for
   streaming parity; it does not emit SSE for `stream: true`.
4. Decide whether implementation discovery targets production vLLM `0.18.x` or
   local macOS vLLM core `0.21.0` plus vLLM Metal plugin.
5. Use the existing local `OpenAIChatServingAdapter` boundary under the vLLM
   backend as the single integration point for native vLLM serving internals.
6. Keep the current dynamic import + constructor probing path as the single
   native-serving initialization path under `VLLMBackend`.
7. Validate the current adapter constructor against upstream `vLLM 0.18.x`
   source structure and a runnable environment.
8. Use the local Qwen3.5 snapshot as the real smoke-test model only after a
   real `import vllm` environment is available.
9. Once a real environment exists, verify:
   - engine-client object selection
   - `OpenAIServingModels` construction
   - `OpenAIServingChat.create_chat_completion(...)` non-streaming behavior
   - native streaming chunk behavior
10. Only after that, decide whether the current fallback to legacy `LLM.chat()`
    should remain or be tightened for `openai_serving.enabled` models.

## 15. Handoff Summary After Local Qwen3.5, vLLM Metal, and Adapter Passthrough Validation

This section is a compact handoff record for continuing work after context
compaction.

### 15.1 Completed In This Round

- `config/models.yaml` was changed to a local-only working catalog:
  - enabled `qwen3.5-0.8b` as a local `vllm` chat model
  - enabled `qwen3.5-0.8b-metal` as a `vllm_openai_proxy` model
  - commented out unavailable production `/nas_data/...` models
- `config/settings.yaml` was changed to local-safe defaults:
  - `execution_mode: stub`
  - `backend_init_mode: stub`
- `.gitignore` now ignores `.venv-metal/`.
- `.venv-metal/` was created with Python 3.12 and `vllm-metal==0.1.0`.
- The local Qwen3.5 snapshot was confirmed to exist and load through
  `vllm-metal` when Metal access is available.
- infer-nexus successfully proxied a real non-streaming model response from
  local `vllm_metal.server`.
- `VLLMBackend` now includes a local OpenAI serving adapter boundary and
  passthrough request builder.
- Fake adapter tests now cover both non-streaming payload passthrough and
  streaming chunk passthrough at the backend layer.
- A serving request payload with assistant `content: null` plus `tool_calls`
  is now preserved correctly by the backend passthrough builder.
- `VLLMBackend._initialize_openai_serving_chat_adapter()` now includes a real
  dynamic import + constructor probing path for candidate vLLM serving layouts.
- `VLLMBackend` now stores `openai_serving_adapter_init_error` so failed native
  adapter initialization is diagnosable instead of silent.
- Local tests now cover both:
  - fake adapter passthrough behavior
  - fake native serving-object construction through the dynamic initializer
- An attempt was made to install official vLLM core plus the Metal plugin into
  `.venv-metal/`, but this did not yet produce a stable local `import vllm`
  validation environment.

### 15.2 Verified Behavior

- `qwen3.5-0.8b` catalog load and runtime context generation work.
- Stub non-streaming chat works through infer-nexus for `qwen3.5-0.8b`.
- Stub streaming chat works through infer-nexus for `qwen3.5-0.8b` and emits
  OpenAI-style SSE.
- Direct `vllm_metal.server` non-streaming chat works with the local Qwen3.5
  snapshot.
- infer-nexus `vllm_openai_proxy` non-streaming chat works against
  `vllm_metal.server`.
- Direct `vllm_metal.server` with `stream: true` does not stream; it returns a
  normal JSON response.
- infer-nexus proxying `stream: true` to `vllm_metal.server` also receives a
  normal JSON response because the upstream does not emit SSE.
- A fake OpenAI serving adapter can drive backend non-streaming chat and return
  a full OpenAI-compatible payload unchanged.
- A fake OpenAI serving adapter can drive backend streaming chat and emit
  reasoning/tool-call compatible chunks unchanged.
- The backend can construct a native-style serving adapter from dynamically
  imported fake upstream classes and invoke `create_chat_completion(...)`
  through the local wrapper.
- Native adapter import/construction failures are captured and available for
  later runtime error reporting.

### 15.3 Still Not Completed

- Native local Ray Serve `vllm` backend serving adapter initialization is
  implemented as a probing path, but it has not yet been validated against a
  real installed `vllm` package.
- Real local `vllm` backend streaming remains intentionally blocked when a real
  legacy `LLM` engine is initialized and the native adapter cannot be
  constructed.
- Reasoning is only implemented at config/policy/passthrough/schema level; it
  has not been verified through native vLLM OpenAI serving.
- Tool calling, logprobs, guided decoding, structured output, and full vLLM
  extra-parameter parity are not complete.
- Observability requirements for streaming lifecycle are not complete.
- Broad test suite remains misaligned with the local-only model catalog.

### 15.4 Environment Commands Used

The useful verification commands were ad hoc Python/curl probes, not permanent
test scripts. The important reproducible pieces are:

```bash
python3.12 -m venv .venv-metal
.venv-metal/bin/python -m pip install -U pip setuptools wheel
.venv-metal/bin/python -m pip install vllm-metal
```

Real model probe requires running outside the sandbox so MLX can access Metal:

```bash
.venv-metal/bin/python -c "from vllm_metal.server import create_engine; p='/Users/zhoushujian/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'; e=create_engine(p); print(e.generate('User: hello\nAssistant:', max_tokens=8, temperature=0.0))"
```

Local upstream server command:

```bash
.venv-metal/bin/python -m vllm_metal.server \
  --model /Users/zhoushujian/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
  --host 127.0.0.1 \
  --port 18000 \
  --log-level info
```

### 15.5 Decision Point

There are now two viable but different validation tracks:

1. Production target track:
   - inspect and integrate vLLM `0.18.x` OpenAI serving internals
   - likely requires Linux/CUDA or a compatible server environment for final
     real validation

2. macOS local target track:
   - install official vLLM core `0.21.0` plus vLLM Metal plugin using the
     upstream `install.sh`
   - verify whether `vllm serve ...` through that full stack emits SSE for
     `stream: true`
   - if it does, use it for local real streaming smoke tests while keeping
     production compatibility concerns separate

In either track, the next code step should be the same:

- validate and tighten the existing
  `VLLMBackend._initialize_openai_serving_chat_adapter()` constructor path
  against a real upstream vLLM environment
