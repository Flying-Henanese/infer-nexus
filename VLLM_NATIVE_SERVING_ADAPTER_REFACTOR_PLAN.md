# vLLM Native Serving Adapter Refactor Plan

## Background

The current `infer-nexus` vLLM backend has a working runtime shape:

```text
FastAPI gateway
  -> RuntimeDispatcher
  -> Ray Serve replica
  -> VLLMBackend
  -> local vLLM engine
```

This shape has already been validated in the target environment:

```text
ray == 2.48.0
vllm == 0.18.x
```

The problem is not the Ray replica lifecycle. The problem is that
`src/infer_nexus/backends/vllm.py` currently owns too much protocol behavior:

- OpenAI chat request interpretation
- multimodal content conversion
- tool-calling and reasoning request filtering
- streaming delta reconstruction
- OpenAI response shape recreation
- embedding/rerank response recreation

This creates drift from vLLM's own serving behavior.

The goal is to keep the known-good Ray replica + vLLM engine lifecycle, but move
compatibility-sensitive request/response behavior into task-specific adapters
that reuse vLLM 0.18 native serving logic as much as possible.

## Decision

Do **not** migrate to Ray Serve LLM / `ray.serve.llm.build_openai_app` for now.

Reason:

- `build_openai_app` belongs to Ray Serve LLM, not vLLM.
- Ray 2.48.0 has Ray Serve LLM, but its bundled/validated vLLM compatibility is
  much older than vLLM 0.18.
- The current custom Ray replica lifecycle has already been validated with
  Ray 2.48.0 + vLLM 0.18.

Use the custom replica architecture and introduce a `VLLMNativeServingAdapter`
layer instead.

## Target Architecture

```text
FastAPI gateway
  -> auth / admission / model routing / observability
  -> RuntimeDispatcher
  -> Ray Serve replica
  -> VLLMBackend
      -> engine lifecycle only
      -> task dispatch only
      -> VLLMNativeServingAdapter for compatibility-sensitive paths
      -> VLLMLocalFallbackAdapter for local_best_effort paths
```

The gateway and backend should avoid reimplementing vLLM serving semantics.

## Config Semantics

Replace or supersede `strict_openai` with a vLLM-oriented compatibility mode:

```yaml
compat_mode: vllm_native | local_best_effort
```

Meaning:

- `vllm_native`
  - The model should follow vLLM 0.18 serving behavior as closely as possible.
  - Requests are passed into vLLM native serving adapters with minimal mutation.
  - No silent fallback to local handcrafted protocol handling.

- `local_best_effort`
  - The model uses the existing local engine APIs such as `LLM.chat`,
    `AsyncLLMEngine.generate`, `LLM.embed`, or `LLM.score`.
  - This path does not claim full vLLM serving compatibility.

Initial support matrix:

| Task | `vllm_native` | `local_best_effort` |
| --- | --- | --- |
| chat | Yes | Yes |
| embedding | Yes, if native embedding serving adapter can be constructed | Yes |
| rerank | No for first pass | Yes |

Rerank should stay `local_best_effort` until vLLM 0.18 rerank/score serving
internals are inspected and wrapped deliberately.

## Proposed Module Layout

Split `src/infer_nexus/backends/vllm.py` into smaller modules.

Recommended layout:

```text
src/infer_nexus/backends/vllm.py
src/infer_nexus/backends/vllm_native/
  __init__.py
  common.py
  chat.py
  embedding.py
  errors.py
src/infer_nexus/backends/vllm_local/
  __init__.py
  chat.py
  embedding.py
  rerank.py
```

Responsibilities:

- `backends/vllm.py`
  - `VLLMBackend`
  - engine startup/shutdown
  - runtime spec validation
  - task dispatch
  - adapter selection based on `compat_mode` and task

- `vllm_native/common.py`
  - dynamic import helpers for vLLM 0.18 serving internals
  - engine client resolution
  - common payload model dumping
  - common SSE passthrough helpers

- `vllm_native/chat.py`
  - wrapper around vLLM OpenAI chat serving internals
  - minimal request mutation:
    - rewrite `model` to `served_model_name`
    - merge `extra_body` into top-level payload
    - preserve `messages`, tools, reasoning, stream options, multimodal blocks
  - stream passthrough without local delta reconstruction

- `vllm_native/embedding.py`
  - wrapper around vLLM 0.18 native embeddings serving internals
  - minimal request mutation:
    - rewrite `model` to `served_model_name`
    - preserve `input`, `encoding_format`, `dimensions`, and vLLM extras
  - OpenAI-compatible embeddings response passthrough

- `vllm_local/*`
  - existing local best-effort logic moved out of `vllm.py`
  - preserves current behavior for dev/stub and fallback models

## Phase 1: Rename Compatibility Mode

### Changes

- Add enum values:

```python
class CompatibilityMode(StrEnum):
    VLLM_NATIVE = "vllm_native"
    LOCAL_BEST_EFFORT = "local_best_effort"
```

- Keep backward compatibility if `strict_openai` already exists:
  - either accept it as an alias for `vllm_native` for chat only
  - or migrate tests/config immediately if no published config depends on it

- `ModelConfig.compat_mode` default remains `local_best_effort`.

### Validation

- `vllm_native` allowed for:
  - `backend: vllm`, `task: chat`
  - `backend: vllm`, `task: embedding`
- `vllm_native` rejected for:
  - `task: rerank` in the first pass
- `vllm_openai_proxy` remains an external HTTP proxy and should not be confused
  with replica-local native serving.

## Phase 2: Extract Local Best-Effort Code

Move existing fallback behavior out of `src/infer_nexus/backends/vllm.py`.

Do not change behavior in this phase.

Suggested extraction order:

1. Move chat fallback helpers:
   - sampling params
   - chat kwargs
   - message serialization
   - async generate path
   - sync `LLM.chat` path
   - local delta reconstruction

2. Move embedding helpers:
   - input normalization
   - base64 encoding
   - `LLM.embed` result conversion

3. Move rerank helpers:
   - document normalization
   - `LLM.score` result conversion

Acceptance criteria:

- Existing tests pass with no semantic changes.
- `vllm.py` becomes an orchestration class instead of a protocol implementation
  file.

## Phase 3: Native Chat Adapter

Create `VLLMNativeChatServingAdapter`.

Behavior:

- Construct vLLM 0.18 OpenAI chat serving classes inside the Ray replica.
- Do not start an HTTP server inside the replica.
- Use the already-initialized local vLLM engine/client.
- Build request payload from `ChatCompletionsRequest.model_dump(...)`.
- Rewrite only:
  - `model` -> `served_model_name`
  - merge `extra_body` into top-level payload
- Preserve:
  - `messages`
  - content blocks
  - image blocks
  - tools
  - `tool_choice`
  - `parallel_tool_calls`
  - reasoning fields
  - `response_format`
  - `stream`
  - `stream_options`
  - vLLM extension fields

Strict behavior for `compat_mode: vllm_native`:

- Adapter init failure fails startup or request clearly.
- Adapter invocation failure is surfaced.
- No fallback to local chat protocol reconstruction.
- Streaming chunks are passed through:
  - bytes/string chunks unchanged
  - dict chunks framed as SSE without field mutation

## Phase 4: Native Embedding Adapter

Create `VLLMNativeEmbeddingServingAdapter`.

First inspect vLLM 0.18 internals for the embedding serving class and request
class used by `vllm serve` for `/v1/embeddings`.

Expected behavior:

- Construct vLLM native embeddings serving object inside the Ray replica.
- Use existing engine/client.
- Build request payload from `EmbeddingRequest.model_dump(...)`.
- Rewrite only:
  - `model` -> `served_model_name`
  - merge extension fields if the schema supports them
- Preserve:
  - `input`
  - `encoding_format`
  - `dimensions`
  - `user`
  - vLLM extra fields such as `truncate_prompt_tokens`, if accepted by schema
- Return vLLM's OpenAI-compatible embeddings payload with minimal mutation.

Strict behavior for `compat_mode: vllm_native`:

- Adapter init failure fails clearly.
- Adapter invocation failure is surfaced.
- No fallback to local embedding response reconstruction.

## Phase 5: Keep Rerank Local

For the first refactor pass:

- rerank remains `local_best_effort`
- rerank continues using the existing `LLM.score` path
- no `vllm_native` rerank mode

Reason:

- OpenAI has no official rerank API.
- vLLM's rerank endpoints are Jina/Cohere-compatible, not OpenAI-compatible.
- Wrapping vLLM 0.18 rerank serving internals should be a separate task.

Future option:

```yaml
compat_mode: vllm_native
protocol_profile: cohere_rerank | jina_rerank | vllm_score
```

Do not mix this into the first pass.

## Phase 6: Runtime and API Behavior

Gateway responsibilities:

- authentication
- routing
- admission/rate limiting
- model lookup
- model alias mapping
- request ID/logging/metrics
- minimal model field rewrite inside adapter

Gateway should not own:

- chat template semantics
- multimodal content interpretation for native mode
- tool protocol behavior
- reasoning field semantics
- strict stream chunk reconstruction
- native embedding output shaping

Runtime executor behavior:

- Native streaming chunks are passthrough.
- Local fallback chunks may still use existing mapped SSE behavior.
- `vllm_openai_proxy` remains a separate external-upstream path.

## Testing Plan

The target mac development environment cannot run real vLLM replicas, so local
validation should focus on syntax and mocked contract tests.

### Local tests

- `python -m compileall src tests`
- Unit tests with fake native serving adapters:
  - native chat preserves payload shape
  - native chat rewrites only model and merges `extra_body`
  - native chat stream passes through bytes/string/dict chunks
  - native chat adapter missing fails in `vllm_native`
  - native chat invocation failure does not fallback
  - native embedding preserves payload shape
  - native embedding adapter missing fails in `vllm_native`
  - rerank rejects `vllm_native`
  - local_best_effort behavior remains unchanged

### Target environment tests

Run on a Linux/GPU environment with:

```text
ray == 2.48.0
vllm == 0.18.x
```

Compare behavior against `vllm serve` for:

- `/v1/chat/completions`
  - string content
  - text content blocks
  - image content blocks
  - tools
  - `tool_choice`
  - reasoning fields
  - streaming
- `/v1/embeddings`
  - string input
  - list input
  - `encoding_format`
  - `dimensions`, if supported by the model
  - vLLM extension fields

Rerank target tests should only validate existing local best-effort behavior.

## Migration Notes

- Keep current working local fallback path until native adapters are proven in
  the target environment.
- Do not remove rerank fallback code.
- Do not introduce Ray Serve LLM in this branch.
- Do not start an internal HTTP server per replica.
- Avoid broad unrelated cleanup while extracting modules.

## Success Criteria

- `VLLMBackend` is primarily an engine lifecycle and adapter dispatch class.
- Native chat no longer uses gateway-local protocol reconstruction.
- Native embedding no longer uses gateway-local OpenAI response reconstruction.
- Rerank remains stable on the existing local path.
- `vllm_native` failures are explicit and never silently degrade into
  `local_best_effort`.
- The behavior of chat and embedding is closer to vLLM 0.18 external serving
  than to infer-nexus handcrafted compatibility code.
