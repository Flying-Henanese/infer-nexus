# Inference Backend Design

## 1. Overview

`src/infer_nexus/backends/` is the runtime backend adapter layer for locally hosted inference models. It sits below the API, catalog, routing, and Ray Serve orchestration layers, and above the replica-local vLLM runtime.

The backend layer has two main responsibilities:

- Convert model catalog configuration into executable runtime specs.
- Convert typed inference requests into backend-specific calls for chat completion, streaming chat completion, embeddings, and rerank.

At the moment, the local backend implementation is centered on `VLLMBackend`. The design keeps a stable project-level backend interface while allowing different vLLM execution paths inside the replica.

## 2. Source Layout

```text
src/infer_nexus/backends/
  base.py
  vllm.py
  vllm_strict.py
  vllm_local_best_effort.py
  vllm_native/
    __init__.py
    chat.py
    embedding.py
    common.py
```

### `base.py`

`base.py` defines the abstract backend contract:

```text
InferenceBackend
  - validate_runtime_spec()
  - startup()
  - shutdown()
  - build_runtime_spec()
  - chat_completion()
  - chat_completion_stream()
  - embedding()
  - rerank()
```

Runtime code depends on this interface instead of depending directly on vLLM. A future backend can implement the same interface without changing API routing semantics.

### `vllm.py`

`vllm.py` contains `VLLMBackend`, the main local vLLM backend implementation.

It is the facade and state owner for the vLLM backend:

- Owns the replica-local vLLM engine object.
- Owns initialized OpenAI serving adapters when native serving is enabled.
- Validates task mode, backend type, and native serving requirements.
- Builds runtime specs from model catalog entries.
- Starts and shuts down vLLM engine resources.
- Exposes the `InferenceBackend` methods used by the runtime replica.
- Delegates chat execution to strict/native or local best-effort executors.
- Handles local embedding and rerank paths where applicable.
- Provides shared helpers for request defaults, sampling params, chat kwargs, multimodal conversion, stub responses, and result conversion.

Internally it creates two executor helpers:

```python
self.local_best_effort_executor = LocalBestEffortVLLMExecutor(self)
self.strict_executor = StrictNativeVLLMExecutor(self)
```

The executors receive the `VLLMBackend` instance and use its shared state and helper methods.

### `vllm_strict.py`

`vllm_strict.py` contains `StrictNativeVLLMExecutor`.

This executor is responsible for the stricter, vLLM-native OpenAI serving path. Instead of manually translating OpenAI-style requests into low-level vLLM engine calls, it tries to construct vLLM's own OpenAI serving objects inside the Ray replica and call them directly.

It handles:

- Dynamic imports of vLLM OpenAI serving classes across supported vLLM internal layouts.
- Resolving a compatible engine client from the initialized backend engine.
- Constructing vLLM `OpenAIServingModels`, `OpenAIServingChat`, optional `OpenAIServingRender`, and embedding serving objects.
- Wrapping those native serving objects with project-local adapters from `vllm_native/`.
- Building OpenAI serving request payloads.
- Calling native chat, streaming chat, and embedding paths.
- Falling back to `LocalBestEffortVLLMExecutor` for chat when native serving is optional and fails.

When the runtime spec explicitly requests `vllm_native`, adapter failure is treated as a backend configuration or invocation error rather than silently falling back.

### `vllm_local_best_effort.py`

`vllm_local_best_effort.py` contains `LocalBestEffortVLLMExecutor`.

This executor is the project-maintained compatibility path. It does not use vLLM's OpenAI serving objects. It directly calls the replica-local vLLM engine and manually converts requests and responses.

It handles:

- Merging request defaults and request extras.
- Enforcing request policy for tools, reasoning fields, and unknown OpenAI fields.
- Building vLLM `SamplingParams`.
- Building vLLM chat kwargs.
- Serializing text and multimodal chat message content.
- Handling sync engine and async engine call differences.
- Calling `engine.chat(...)`, `engine.generate(...)`, or related engine methods.
- Building OpenAI-like streaming chunks from engine outputs.
- Returning stub responses when the engine is unavailable in development/test scenarios.

This path is more tolerant of vLLM OpenAI serving internals changing, but its behavior is only best-effort compatible with OpenAI/vLLM serving semantics.

### `vllm_native/`

`vllm_native/` is a small adapter toolkit used by `vllm_strict.py`. It is not an independent backend and does not implement `InferenceBackend`.

Its purpose is to isolate direct dependencies on vLLM OpenAI serving internals behind stable local protocols.

#### `vllm_native/chat.py`

Defines the chat serving adapter contract and implementation:

```text
OpenAIChatServingAdapter
  - chat_completion()
  - chat_completion_stream()

DynamicVLLMOpenAIChatServingAdapter
  - builds vLLM ChatCompletionRequest objects
  - calls serving_chat.create_chat_completion(...)
  - normalizes dict, bytes, str, and Pydantic-like payloads
  - supports normal and streaming results
```

#### `vllm_native/embedding.py`

Defines the embedding serving adapter contract and implementation:

```text
OpenAIEmbeddingServingAdapter
  - embedding()

DynamicVLLMOpenAIEmbeddingServingAdapter
  - builds vLLM embedding request objects
  - supports create_embedding(), create_pooling(), or callable serving objects
  - normalizes dict, Pydantic-like payloads, and response bodies into dicts
```

#### `vllm_native/common.py`

Contains shared compatibility helpers:

```text
ResolvedOpenAIServingImports
  - records dynamically imported vLLM serving classes

OpenAIServingEngineClientCompatProxy
  - wraps engine client candidates
  - provides generate(), encode(), errored, and fallback attribute access
  - adapts sync iterables, awaitables, and async iterators
  - filters method arguments against callable signatures
```

#### `vllm_native/__init__.py`

Re-exports the adapter protocols and helper classes so callers can import from `infer_nexus.backends.vllm_native`.

## 3. High-Level Data Flow

The HTTP/API layer does not call the vLLM engine directly. Requests first flow through routing and runtime orchestration:

```text
Client
  -> OpenAI-compatible API routes
  -> RuntimeDispatcher
  -> RuntimeExecutor
  -> Ray Serve deployment handle
  -> ModelRuntimeReplica
  -> VLLMBackend
  -> selected replica-local execution path
```

The Ray Serve handle is used to reach the model-specific `ModelRuntimeReplica`. Once the request is inside a replica, both strict/native and local best-effort modes operate on replica-local Python objects.

That boundary is important:

```text
API process -> Ray Serve replica
  Uses Ray Serve handle.

Inside the replica -> vLLM engine
  Does not use another Ray handle.
  Calls local backend, adapter, and engine objects in the replica process.
```

## 4. Runtime Replica Flow

When a local model deployment starts:

```text
ModelRuntimeReplica.__init__
  -> _build_backend(runtime_spec["backend"])
  -> VLLMBackend(runtime_spec)
  -> backend.validate_runtime_spec(runtime_spec, runtime_context)
  -> backend.startup()
```

For local vLLM models, `_build_backend("vllm")` creates `VLLMBackend`.

For a chat request:

```text
ModelRuntimeReplica.chat_completion()
  -> ChatCompletionsRequest.model_validate(...)
  -> VLLMBackend.chat_completion(...)
  -> StrictNativeVLLMExecutor.chat_completion(...)
  -> native OpenAI serving adapter if available
  -> otherwise local best-effort fallback when allowed
```

For a streaming chat request:

```text
ModelRuntimeReplica.chat_completion_stream()
  -> ChatCompletionsRequest.model_validate(...)
  -> VLLMBackend.chat_completion_stream(...)
  -> StrictNativeVLLMExecutor.chat_completion_stream(...)
  -> native OpenAI serving stream if available
  -> otherwise local best-effort stream when allowed
```

For an embedding request:

```text
ModelRuntimeReplica.embedding()
  -> EmbeddingRequest.model_validate(...)
  -> VLLMBackend.embedding(...)
  -> if vllm_native: StrictNativeVLLMExecutor.embedding(...)
  -> else: engine.embed(...) and VLLMBackend result conversion
```

For a rerank request:

```text
ModelRuntimeReplica.rerank()
  -> RerankRequest.model_validate(...)
  -> VLLMBackend.rerank(...)
  -> local backend scoring/conversion path
```

`vllm_native` is explicitly not supported for rerank.

## 5. Execution Modes

The backend supports two important local vLLM execution styles.

### Strict / vLLM Native

This path tries to reuse vLLM's own OpenAI-compatible serving implementation.

```text
VLLMBackend
  -> StrictNativeVLLMExecutor
  -> vllm_native adapters
  -> vLLM OpenAI serving objects
  -> replica-local vLLM engine
```

Properties:

- More closely follows vLLM's official OpenAI serving behavior.
- Constructs vLLM request objects such as chat completion and embedding requests.
- Uses vLLM serving objects such as `OpenAIServingChat` or embedding serving classes.
- Depends on vLLM internal module layouts, so dynamic import and compatibility wrappers are required.
- Requires `openai_serving.enabled=true` for native local chat and embedding models.
- Treats adapter unavailability as an error when the model explicitly uses `vllm_native`.
- Does not support rerank.

This mode is appropriate when OpenAI-compatible semantics should match vLLM's own serving layer as closely as possible.

### Local Best Effort

This path is maintained by infer-nexus and directly calls the vLLM engine.

```text
VLLMBackend
  -> LocalBestEffortVLLMExecutor
  -> engine.chat(...) / engine.generate(...) / engine.embed(...)
  -> project-side response conversion
```

Properties:

- Avoids direct dependency on vLLM OpenAI serving internals.
- Manually builds sampling params, chat kwargs, messages, multimodal payloads, and streaming chunks.
- Supports more tolerant degradation and stub behavior in some development/test cases.
- Provides best-effort OpenAI compatibility, but may differ from vLLM's official OpenAI serving behavior.
- Can be used as a fallback when native OpenAI serving is optional and fails.

This mode is appropriate when robustness across vLLM internal changes is more important than exact vLLM OpenAI serving semantics.

## 6. Strict Fallback Semantics

`StrictNativeVLLMExecutor` is always involved in `VLLMBackend.chat_completion()` and `VLLMBackend.chat_completion_stream()`, but it does not always mean native serving is mandatory.

The chat behavior is:

```text
if openai_serving_chat_adapter exists:
  try native OpenAI serving
  if it fails:
    if runtime_spec is vllm_native:
      raise the error
    else:
      clear the adapter and fall back to local_best_effort

elif runtime_spec is vllm_native:
  raise backend configuration error

else:
  use local_best_effort
```

Embedding behavior is stricter for native mode:

```text
if runtime_spec is vllm_native:
  require openai_serving_embedding_adapter
  call native embedding serving

else:
  use VLLMBackend local embedding path
```

Rerank behavior:

```text
if runtime_spec is vllm_native:
  raise BackendConfigurationError

else:
  use VLLMBackend rerank path
```

## 7. Configuration Inputs

The model catalog drives backend behavior through `ModelConfig` and its nested `VLLMConfig`.

Relevant fields include:

```text
ModelConfig
  - task
  - backend
  - compat_mode
  - model_path
  - model_loading_config
  - dtype
  - tensor_parallel_size
  - max_model_len
  - capabilities
  - engine_kwargs
  - vllm

VLLMConfig
  - engine_kwargs
  - request_defaults
  - request_policy
  - openai_serving

VLLMOpenAIServingConfig
  - enabled
  - enable_reasoning
  - reasoning_parser

VLLMRequestPolicy
  - allow_tools
  - allow_reasoning
  - passthrough_unknown_openai_fields
```

Validation enforces important constraints:

- Local vLLM native chat and embedding models require `vllm.openai_serving.enabled=true`.
- `strict_openai` is only supported as an alias for local vLLM chat models.
- `vllm_native` is not supported for local vLLM rerank models.
- The runtime task mode must match the catalog task.

## 8. Architecture Summary

The backend design separates responsibilities into layers:

```text
InferenceBackend
  Stable project-level backend contract.

VLLMBackend
  Main local vLLM implementation, lifecycle manager, runtime spec builder,
  state owner, and facade.

StrictNativeVLLMExecutor
  vLLM OpenAI serving path coordinator.

vllm_native/
  Thin compatibility adapter layer around vLLM OpenAI serving internals.

LocalBestEffortVLLMExecutor
  Project-maintained direct engine execution path.
```

The resulting design gives the project two useful operating modes:

- A stricter native mode for closer compatibility with vLLM's OpenAI serving behavior.
- A more tolerant local mode for direct engine use, fallback behavior, and reduced dependence on vLLM internal serving APIs.

In both cases, requests reach the model via Ray Serve handle first, but strict/native and local best-effort execution happen inside the same model replica against replica-local backend and vLLM engine objects.
