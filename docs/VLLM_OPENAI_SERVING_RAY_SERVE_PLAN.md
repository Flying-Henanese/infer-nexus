# vLLM Chat Streaming on Ray Serve Plan

## 1. Goal

This document describes the implementation path for making `infer-nexus`
behave like an OpenAI-compatible chat service while keeping the current Ray
Serve based runtime architecture and replica ownership.

The original plan explored reusing vLLM's internal OpenAI serving classes
inside each Ray Serve replica. After implementation and remote validation, the
current primary path is different and more stable:

```text
Ray Serve replica
  -> VLLMBackend
    -> vLLM AsyncLLMEngine.generate(...)
      -> vLLM cumulative RequestOutput stream
        -> infer-nexus chat_delta events
          -> RuntimeExecutor OpenAI SSE chunks
```

The vLLM OpenAI serving adapter boundary remains useful as a compatibility and
experimentation seam, but local Ray Serve chat streaming no longer depends on
private `OpenAIServingChat` internals.

The target user experience is:

- Clients use one OpenAI-compatible base URL exposed by `infer-nexus`.
- Clients select models through the standard OpenAI `model` request field.
- Existing users who directly call a vLLM OpenAI-compatible server should be
  able to switch to `infer-nexus` with minimal or no request/response changes.
- Chat streaming, vLLM-specific request parameters, reasoning fields, tool
  calling fields, logprobs, and other OpenAI-compatible response details should
  be preserved as close to native vLLM behavior as possible.
- `stream=True` for local chat should produce real incremental SSE chunks when
  `AsyncLLMEngine` is available.

The target runtime architecture remains:

```text
Client
  -> Single FastAPI Gateway
    -> model routing by request.body.model
      -> Ray Serve model deployment
        -> one of N equal Ray Serve replicas
          -> replica-local vLLM engine
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
            -> AsyncLLMEngine.generate for chat
            -> sync vLLM LLM fallback for chat / embed / score
          -> internal chat_delta or task payload
      -> RuntimeExecutor encodes OpenAI-style JSON/SSE response
```

This is sufficient for basic local inference, but it is not equivalent to
vLLM's OpenAI-compatible server.

Historical chat gaps before the current streaming work were:

- `stream=True` was converted to `stream=False` before backend execution.
- Streaming responses were synthesized by the gateway after full generation had
  completed.
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

The current implementation therefore avoids treating `LLM.chat()` as the
streaming main path. It uses `AsyncLLMEngine.generate(...)` for local chat
streaming and keeps sync `LLM.chat()` as a compatibility fallback.

## 5. Target Design

The target design keeps Ray Serve as the runtime and autoscaling layer, but
changes the local chat execution path from:

```text
vllm.LLM.chat()
  -> infer-nexus custom response adapter
```

to:

```text
replica-local vLLM AsyncLLMEngine.generate(...)
  -> infer-nexus standardized chat_delta events
  -> RuntimeExecutor OpenAI-compatible JSON/SSE response
```

Conceptually:

```text
Client
  -> FastAPI /v1/chat/completions
    -> infer-nexus model lookup and admission
      -> Ray Serve deployment handle
        -> selected replica
          -> VLLMBackend
            -> vLLM AsyncLLMEngine for chat when available
            -> sync vLLM LLM fallback otherwise
          -> OpenAI-compatible payload or SSE chunks
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

The vLLM backend should own:

- chat prompt construction through the engine tokenizer chat template
- vLLM sampling parameter construction
- native incremental output consumption from `AsyncLLMEngine.generate(...)`
- conversion from cumulative vLLM output text to incremental `chat_delta` events
- sync fallback behavior when async engine initialization is unavailable

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

- prefer initializing a replica-local `AsyncLLMEngine`
- keep existing runtime spec fields such as model path, dtype,
  tensor parallel size, max model length, GPU memory utilization, and engine
  kwargs
- preserve Qwen3 reasoning configuration:
  - `enable_reasoning`
  - `reasoning_parser`
- if async engine initialization fails, fall back to sync `vllm.LLM`

For `task_mode == "embed"` and `task_mode == "score"`:

- keep current behavior for this implementation pass
- do not force these tasks through the chat serving adapter

This prevents chat-specific changes from breaking embedding and rerank models.

### 6.3 Request Flow for Non-Streaming Chat

For `stream=False`:

1. FastAPI route validates the top-level request enough to find `model`.
2. Runtime dispatch resolves the model to a Ray Serve deployment.
3. The selected replica calls `VLLMBackend.chat_completion(...)`.
4. If the backend is using `AsyncLLMEngine`, it builds a chat-template prompt,
   consumes `engine.generate(...)` until the final output, and returns an
   OpenAI-compatible chat completion payload.
5. If the backend is using sync `LLM`, it uses the existing compatibility
   conversion path.
6. `RuntimeExecutor` returns full OpenAI-compatible payloads without narrowing
   them when possible.

The current `_build_chat_response_from_payload()` behavior should remain only as
a fallback for stub/dev paths or legacy backend payloads.

### 6.4 Request Flow for Streaming Chat

For `stream=True`:

1. FastAPI route dispatches the request as a streaming request.
2. `RuntimeExecutor` must not rewrite the request to `stream=False`.
3. `RuntimeExecutor` calls a streaming method on the Ray Serve deployment.
4. The selected replica calls `VLLMBackend.chat_completion_stream(...)`.
5. The backend prefers `AsyncLLMEngine.generate(...)` and yields standardized
   internal `chat_delta` events with `delta_text` and `finish_reason`.
6. `RuntimeExecutor` maps `chat_delta` events into OpenAI-compatible SSE:
   - first chunk: `delta.role=assistant`
   - content chunks: `delta.content=<incremental text>`
   - terminal chunk: `finish_reason`
   - final frame: `data: [DONE]`
7. If the backend falls back to sync `LLM`, streaming remains protocol-compatible
   but may emit a completed response as one content chunk.

When `AsyncLLMEngine` is active, the gateway does not wait for full generation
before returning content chunks.

The desired wire format is the native OpenAI SSE style:

```text
data: {"id":"...","object":"chat.completion.chunk","choices":[...]}

data: {"id":"...","object":"chat.completion.chunk","choices":[...]}

data: [DONE]
```

The gateway performs SSE encoding and the `chat_delta -> OpenAI chunk` mapping,
but it should not parse model text or perform tokenization itself.

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

For chat models using the async path, this method should delegate to
`AsyncLLMEngine.generate(...)` and normalize cumulative vLLM outputs into
incremental `chat_delta` events.

For stub/dev paths, it may yield deterministic `chat_delta` test events.

The backend should also support non-streaming passthrough:

```text
chat_completion(...) -> full OpenAI-compatible response payload
```

When possible, the backend should avoid converting vLLM results into an overly
reduced custom shape. The current async path returns an OpenAI-compatible chat
completion payload; future work can widen response fidelity further.

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
the vLLM backend or any future serving adapter.

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
  "model": "qwen3-vl-chat-8b-instruct"
}
```

The gateway should not introduce a custom `model_name` field.

Clients should be able to keep using OpenAI SDK style calls:

```python
client.chat.completions.create(
    model="qwen3-vl-chat-8b-instruct",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
    extra_body={"top_k": 50},
)
```

Expected behavior:

- `model` is resolved by `infer-nexus` catalog.
- request body is passed to the selected Ray Serve model deployment.
- vLLM-specific fields are preserved and interpreted by the vLLM backend where
  supported.
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

Backend protocol errors produced by vLLM or by any future vLLM serving adapter
should be preserved as much as possible. The gateway should avoid converting all
backend failures into a generic internal error when the backend already produced
a structured OpenAI-compatible error payload.

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

- Initialize `AsyncLLMEngine` for local chat models when available.
- Route non-streaming chat requests through async `generate(...)` aggregation.
- Route streaming chat requests through async `generate(...)` incremental output.
- Preserve sync `LLM` fallback for environments where async engine startup fails.
- Keep the private vLLM OpenAI serving adapter boundary as an experimental
  compatibility seam, not the primary streaming path.

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
  non-streaming response for local Ray Serve vLLM chat when `AsyncLLMEngine` is
  available.
- Reasoning chunks are preserved.
- Tool call fields are preserved when the model and policy allow them.
- vLLM extra parameters can be passed through in the supported configuration.
- Non-streaming response fields are not narrowed to the old minimal response
  shape when the backend provides a full OpenAI-compatible payload.
- Ray Serve remains responsible for model deployment replicas and autoscaling.
- Existing embedding and rerank behavior remains unchanged.
- Existing proxy backend passthrough behavior remains unchanged.

## 14. Current Implementation Status

Last updated: 2026-05-25.

### 14.0 Current Production-Validated Path

The current verified local Ray Serve chat path is:

```text
FastAPI /v1/chat/completions
  -> RuntimeDispatcher
    -> RuntimeExecutor
      -> Ray Serve deployment handle with stream=True
        -> ModelRuntimeReplica.chat_completion_stream(...)
          -> VLLMBackend
            -> AsyncLLMEngine.generate(...)
              -> cumulative RequestOutput stream
            -> normalized chat_delta events
      -> RuntimeExecutor OpenAI SSE mapper
  -> text/event-stream
```

Remote validation against `192.168.0.194:8000` with the single model
`qwen3-vl-chat-8b-instruct` confirmed:

- `/v1/models` returns the expected single chat model.
- non-streaming `/v1/chat/completions` returns `200 OK`.
- `stream=true` returns `200 OK` with `content-type: text/event-stream`.
- long text generation emits many incremental `delta.content` chunks instead of
  one completed-response chunk.
- the stream terminates with `finish_reason="stop"` and `data: [DONE]`.

This validates that `AsyncLLMEngine` is active for the remote Qwen3-VL chat
deployment.

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
- In stub mode, the backend yields deterministic standardized `chat_delta`
  events which the executor maps to OpenAI-style SSE.
- This enables testing executor/replica/gateway streaming behavior without
  starting a vLLM model.

Current fallback behavior:

- When a real sync `vllm.LLM` engine is initialized and no async engine is
  available, streaming remains protocol-compatible by wrapping a completed chat
  result as SSE. This fallback is intentionally not token-level incremental.

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

#### AsyncLLMEngine chat path

`VLLMBackend` now has a production-validated async chat path.

Implemented pieces:

- chat `startup()` prefers `AsyncLLMEngine` for `task_mode == "generate"`
- async engine initialization uses `AsyncEngineArgs` plus
  `AsyncLLMEngine.from_engine_args(...)` when available
- `VLLMBackend` tracks `engine_kind` so request paths can distinguish async,
  sync, stub, and stopped states
- non-streaming chat consumes async `generate(...)` to the final output and
  returns an OpenAI-compatible response payload
- streaming chat consumes async `generate(...)` incrementally and emits
  standardized `chat_delta` events
- the executor maps `chat_delta` events into OpenAI-compatible SSE chunks
- sync `LLM` remains the fallback if async engine initialization fails

#### OpenAI serving adapter boundary and passthrough contract

`VLLMBackend` still contains an explicit local adapter boundary for experiments
with native vLLM OpenAI serving internals.

Current role:

- `OpenAIChatServingAdapter` and dynamic import probing remain available.
- `vllm.openai_serving.enabled` is preserved in runtime specs and catalog
  metadata.
- This path is no longer required for local Ray Serve chat streaming.
- The remote Qwen3-VL issue showed that `OpenAIServingChat + sync LLM` is not a
  reliable long-term streaming foundation; it should remain an experimental or
  legacy compatibility seam rather than the primary data plane.

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

Additional local verification performed on 2026-05-25 for the async chat path:

- `tests/test_runtime.py::test_vllm_backend_startup_prefers_async_engine_for_chat`
- `tests/test_runtime.py::test_vllm_backend_chat_completion_stream_uses_async_engine_generate_deltas`
- `tests/test_runtime.py::test_vllm_backend_chat_completion_uses_async_engine_generate_result`
- `tests/test_runtime.py::test_vllm_backend_chat_completion_stream_uses_native_vllm_deltas`
- `tests/test_runtime.py::test_vllm_backend_chat_completion_stream_wraps_non_incremental_vllm_result`
- `tests/test_runtime.py::test_vllm_backend_chat_completion_stream_does_not_use_openai_serving_for_sync_fallback`
- `tests/test_dispatcher.py::test_chat_stream_response_maps_standard_delta_events_to_openai_sse`

Result:

```text
7 passed
compileall passed
```

Remote verification performed on 2026-05-25:

- host: `192.168.0.194:8000`
- model: `qwen3-vl-chat-8b-instruct`
- `/v1/models`: `200 OK`
- non-streaming `/v1/chat/completions`: `200 OK`
- short `stream=true`: `200 OK`, `text/event-stream`, `data: [DONE]`
- long `stream=true`: emitted many incremental `delta.content` chunks and
  ended with `finish_reason="stop"` plus `data: [DONE]`

This remote validation confirms that the current branch achieves true
incremental streaming for the Qwen3-VL local Ray Serve vLLM backend.

### 14.3 Not Yet Implemented

The following parts of the original plan are still not implemented.

#### vLLM private OpenAI serving API discovery

- The project has a small local adapter boundary in `VLLMBackend` for vLLM
  OpenAI serving internals.
- That boundary is no longer the primary chat streaming strategy.
- Exact compatibility with vLLM private serving internals is not required for
  the current production-validated async engine path.

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

Expected next work for this optional seam:

- keep dynamic import candidates isolated behind `VLLMBackend`
- do not make private serving adapter success a requirement for local chat
  streaming
- only tighten this path if a future compatibility requirement needs native
  vLLM OpenAI serving fields that the async engine path cannot reproduce

#### Async engine parity hardening

The main chat streaming path is implemented and remotely verified. Remaining
hardening work:

- validate text-only and multimodal prompt construction separately
- add explicit cancellation propagation to abort async engine requests when
  clients disconnect
- improve usage accounting for async streaming terminal chunks if needed
- preserve or surface structured backend errors from async engine failures
- widen response fidelity for reasoning, tool calls, logprobs, and multi-choice
  when these are required by real clients

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

Single-model remote Ray Serve streaming has been verified with
`qwen3-vl-chat-8b-instruct` on `192.168.0.194:8000`.

Still needed:

- confirm cancellation behavior
- verify multiple replicas still behave as peer workers
- add production-oriented stream lifecycle metrics

#### Real model/client validation

Partially done:

- The local Qwen3.5 snapshot exists and has been used for:
  - stub gateway tests through `qwen3.5-0.8b`
  - real non-streaming proxy tests through `qwen3.5-0.8b-metal`

Current real backend status:

- `qwen3-vl-chat-8b-instruct` has been verified through the local Ray Serve
  `vllm` backend path on the remote server.
- non-streaming chat works.
- `stream=True` works and emits real incremental SSE chunks.

Still useful:

- test OpenAI SDK non-streaming request
- test OpenAI SDK streaming request
- test vLLM extra params such as `top_k`
- test Qwen reasoning configuration if supported by the model/config
- test multimodal Qwen3-VL image input on the async path or define fallback
  behavior explicitly

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

The next work should harden the now-working async engine path rather than return
to the private `OpenAIServingChat + sync LLM` path.

Recommended order:

1. Add cancellation propagation from client disconnect to async engine request
   abort.
2. Verify OpenAI SDK streaming against the remote `qwen3-vl-chat-8b-instruct`
   deployment.
3. Test vLLM extra parameters such as `top_k`, `min_p`, and stop sequences.
4. Test Qwen reasoning configuration if enabled for the model.
5. Decide the multimodal policy for Qwen3-VL on async engine:
   - support image inputs by passing the correct vLLM multimodal prompt shape
   - or explicitly fall back to sync `LLM.chat()` for multimodal requests
6. Keep embedding and rerank unchanged until their serving parity is addressed
   separately.

## 15. Current Handoff Summary

This is the current state after the remote Qwen3-VL validation on 2026-05-25.

Completed:

- Local Ray Serve chat now prefers `AsyncLLMEngine`.
- Non-streaming chat works through async generation aggregation.
- Streaming chat emits real incremental `delta.content` chunks through OpenAI
  SSE.
- Sync `LLM` remains as fallback and can still provide protocol-compatible SSE
  over a completed response.
- The private vLLM OpenAI serving adapter path remains isolated but is no
  longer required for working chat streaming.
- Embedding and rerank remain unchanged.

Verified remotely:

- `GET /v1/models` returns `qwen3-vl-chat-8b-instruct`.
- `POST /v1/chat/completions` non-streaming returns `200 OK`.
- `POST /v1/chat/completions` with `stream=true` returns
  `text/event-stream`.
- Long streaming generation returns many content chunks and ends with
  `data: [DONE]`.

Known remaining work:

- OpenAI SDK client-level streaming smoke tests.
- Async engine cancellation/abort on client disconnect.
- Multimodal Qwen3-VL image request handling on the async path.
- Reasoning/tool/logprob/structured-output parity.
- Stream lifecycle observability.
- Test fixture cleanup for old catalog aliases such as `qwen3-chat`.
