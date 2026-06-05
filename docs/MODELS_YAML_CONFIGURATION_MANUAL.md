# models.yaml Configuration Manual

This document explains how to configure different model types in [config/models.yaml](../config/models.yaml), with an emphasis on one recurring source of errors:

- which parameters belong to `infer-nexus`
- which parameters belong to `vLLM`
- where each parameter should be placed

The behavior described here is based on the current implementation in:

- [src/infer_nexus/catalog/models.py](../src/infer_nexus/catalog/models.py)
- [src/infer_nexus/backends/vllm.py](../src/infer_nexus/backends/vllm.py)
- [src/infer_nexus/runtime/executor.py](../src/infer_nexus/runtime/executor.py)
- [src/infer_nexus/model_store.py](../src/infer_nexus/model_store.py)

## 1. Mental Model

Each item under `models:` declares one registered model.

`infer-nexus` reads this file and uses it to decide:

- how the model is identified
- what task it serves
- whether it is a local vLLM model or an upstream proxy model
- how many CPU/GPU resources each replica consumes
- how the request should be routed
- what runtime defaults or compatibility rules should apply

The most important rule is:

- top-level fields are mostly `infer-nexus` platform fields
- `vllm.engine_kwargs` contains vLLM engine parameters
- `vllm.request_defaults`, `vllm.request_policy`, and `vllm.openai_serving` are vLLM-related request and compatibility settings
- `proxy_config` is only for upstream proxy models

## 2. Parameter Categories

### 2.1 infer-nexus Parameters

These parameters are primarily consumed by `infer-nexus` itself rather than passed through unchanged to the vLLM engine.

#### Identity and routing

- `name`
- `alias`
- `served_model_name`
- `task`
- `backend`
- `compat_mode`
- `capabilities`
- `labels`
- `status`

What they do:

- `name` is the canonical model id in the registry.
- `alias` is a user-facing alias that can also be used in API requests.
- `served_model_name` is another externally visible name. If set, it is also registered as a lookup name and is used in response metadata.
- `task` selects the inference path. Current practical values are `chat`, `embedding`, and `rerank`.
- `backend` selects between local vLLM and upstream proxy mode.
- `compat_mode` selects the local request handling style for `backend: vllm`.

#### Model source and artifact policy

- `model_path`
- `model_loading_config.model_id`
- `model_loading_config.revision`
- `require_local_artifacts`

What they do:

- `model_path` points to the local model directory, either absolute or relative to `model_store.root_dir` from [config/settings.yaml](../config/settings.yaml).
- `model_loading_config.model_id` stores the remote-style repo id, such as `Qwen/Qwen3-8B`.
- `require_local_artifacts: false` allows a model to rely on `model_loading_config.model_id` instead of requiring an existing local model directory.

#### Deployment and resource parameters

- `cpu_per_replica`
- `gpu_per_replica`
- `min_replicas`
- `max_replicas`
- `deployment_config.autoscaling_config`
- `deployment_config.ray_actor_options`
- `deployment_config.request_router_config`

What they do:

- These fields define resource reservation, autoscaling bounds, and deployment behavior in Ray Serve.
- They are platform-level deployment settings, not raw vLLM engine settings.

### 2.2 vLLM Runtime Parameters

These are the parameters that affect local vLLM runtime behavior.

There are two subgroups:

#### Top-level vLLM runtime fields

These are still written at the model top level, but they ultimately affect vLLM runtime initialization:

- `dtype`
- `tensor_parallel_size`
- `max_model_len`
- `gpu_memory_utilization`

These are not written under `vllm.engine_kwargs`, but they should be understood as vLLM runtime parameters.

#### `vllm.engine_kwargs`

These should contain parameters that you would conceptually pass into vLLM engine construction.

Typical examples:

- `trust_remote_code`
- `limit_mm_per_prompt`
- `enable_auto_tool_choice`
- `tool_call_parser`
- model-specific vLLM engine options supported by your installed vLLM version

Recommended rule:

- if a parameter is a vLLM engine option and is not one of the explicitly supported top-level runtime fields, place it under `vllm.engine_kwargs`

### 2.3 vLLM OpenAI Compatibility Parameters

These fields are vLLM-related, but they are not plain model-loading engine options.

#### `vllm.request_defaults`

Use this block for default request values that should be applied when the caller does not explicitly provide them.

Typical examples:

- `temperature`
- `top_p`
- `max_tokens`
- `top_k`
- `tool_choice`
- `parallel_tool_calls`
- `chat_template_kwargs`

#### `vllm.request_policy`

Use this block to define what kinds of OpenAI-style request fields are allowed or passed through.

Current supported fields:

- `allow_tools`
- `allow_reasoning`
- `passthrough_unknown_openai_fields`

#### `vllm.openai_serving`

Use this block for vLLM native OpenAI-serving adapter behavior.

Current supported fields:

- `enabled`
- `enable_reasoning`
- `reasoning_parser`

Important behavior:

- for local `chat` and `embedding` models using `compat_mode: vllm_native`, `vllm.openai_serving.enabled: true` is required

### 2.4 Proxy Parameters

These fields are only used when:

- `backend: vllm_openai_proxy`

All such settings belong under `proxy_config`.

Typical fields:

- `upstream_base_url`
- `upstream_model_name`
- `auth.mode`
- `auth.env_var`
- `auth.token`
- `timeout.connect_seconds`
- `timeout.read_seconds`
- `timeout.write_seconds`
- `timeout.pool_seconds`
- `retry.max_attempts`
- `retry.backoff_ms`
- `retry.retry_on_status`
- `streaming.enabled`
- `streaming.passthrough_sse`
- `headers_policy.pass_request_id`
- `headers_policy.forward_authorization`

These are proxy/gateway settings. They are not local vLLM engine parameters.

## 3. Quick Classification Table

| Parameter area | Belongs to | Where to write it |
| --- | --- | --- |
| model identity, alias, task, backend | `infer-nexus` | top level |
| local model path or repo id | `infer-nexus` | top level / `model_loading_config` |
| replica count, CPU/GPU reservation, autoscaling | `infer-nexus` | top level / `deployment_config` |
| `dtype`, `tensor_parallel_size`, `max_model_len`, `gpu_memory_utilization` | vLLM runtime | top level |
| engine startup options like `trust_remote_code` | vLLM engine | `vllm.engine_kwargs` |
| default sampling and request defaults | vLLM-related request behavior | `vllm.request_defaults` |
| tool/reasoning passthrough policy | `infer-nexus` request policy for vLLM models | `vllm.request_policy` |
| native OpenAI serving adapter toggles | vLLM compatibility layer | `vllm.openai_serving` |
| upstream base URL, auth, retry, timeout | `infer-nexus` proxy layer | `proxy_config` |

## 4. Minimal Local vLLM Template

Use this as the base template for a local model:

```yaml
models:
  - name: your-model-name
    alias: your-model-alias
    task: chat
    backend: vllm
    compat_mode: local_best_effort
    model_path: Qwen/Your-Model
    dtype: auto
    tensor_parallel_size: 1
    max_model_len: 8192
    cpu_per_replica: 4
    gpu_per_replica: 1
    gpu_memory_utilization: 0.9
    min_replicas: 1
    max_replicas: 1
    vllm:
      engine_kwargs: {}
      request_defaults: {}
      request_policy: {}
      openai_serving:
        enabled: false
```

## 5. Templates by Model Type

### 5.1 Local Chat Model

```yaml
- name: qwen3-32b
  alias: qwen3-32b
  task: chat
  backend: vllm
  compat_mode: vllm_native
  model_path: Qwen/Qwen3-32B
  dtype: auto
  tensor_parallel_size: 1
  max_model_len: 4096
  cpu_per_replica: 4
  gpu_per_replica: 0.9
  gpu_memory_utilization: 0.9
  min_replicas: 1
  max_replicas: 1
  vllm:
    engine_kwargs:
      enable_auto_tool_choice: true
      tool_call_parser: hermes
    request_defaults:
      tool_choice: auto
      parallel_tool_calls: true
    request_policy:
      allow_tools: true
      allow_reasoning: true
    openai_serving:
      enabled: true
```

Classification:

- top-level identity and resource fields: `infer-nexus`
- `dtype`, `tensor_parallel_size`, `max_model_len`, `gpu_memory_utilization`: vLLM runtime
- `vllm.engine_kwargs`: vLLM engine parameters
- `vllm.request_defaults`, `vllm.request_policy`, `vllm.openai_serving`: vLLM-related compatibility/request settings

### 5.2 Local Vision/Multimodal Chat Model

```yaml
- name: qwen3-vl-8b
  alias: qwen3-vl-8b
  task: chat
  backend: vllm
  compat_mode: vllm_native
  model_path: Qwen/Qwen3-VL-8B-Instruct
  dtype: bfloat16
  tensor_parallel_size: 1
  max_model_len: 10000
  cpu_per_replica: 4
  gpu_per_replica: 0.5
  gpu_memory_utilization: 0.5
  min_replicas: 1
  max_replicas: 1
  capabilities:
    - vision
  vllm:
    engine_kwargs:
      trust_remote_code: true
      limit_mm_per_prompt:
        image: 10
    openai_serving:
      enabled: true
```

Classification highlights:

- `capabilities` is an `infer-nexus` metadata field
- `limit_mm_per_prompt` is a vLLM engine parameter and should be placed under `vllm.engine_kwargs`

### 5.3 Local Embedding Model

```yaml
- name: qwen3-embedding-8b
  alias: qwen3-embedding-8b
  task: embedding
  backend: vllm
  compat_mode: vllm_native
  model_path: Qwen/Qwen3-Embedding-8B
  dtype: bfloat16
  tensor_parallel_size: 1
  max_model_len: 8192
  cpu_per_replica: 2
  gpu_per_replica: 0.3
  gpu_memory_utilization: 0.3
  min_replicas: 1
  max_replicas: 1
  vllm:
    openai_serving:
      enabled: true
```

Important:

- `embedding` with `compat_mode: vllm_native` requires `vllm.openai_serving.enabled: true`

### 5.4 Local Rerank Model

```yaml
- name: bge-reranker-large
  alias: bge-reranker
  task: rerank
  backend: vllm
  model_path: bge-reranker-large
  dtype: float16
  tensor_parallel_size: 1
  max_model_len: 512
  cpu_per_replica: 4
  gpu_per_replica: 0.2
  gpu_memory_utilization: 0.2
  min_replicas: 1
  max_replicas: 1
  vllm:
    engine_kwargs:
      trust_remote_code: true
```

Important:

- do not use `compat_mode: vllm_native` for `rerank`
- current implementation explicitly rejects local vLLM rerank models in `vllm_native`

### 5.5 Upstream Proxy Model

```yaml
- name: mineru-proxy
  alias: mineru
  task: chat
  backend: vllm_openai_proxy
  tensor_parallel_size: 1
  cpu_per_replica: 1
  gpu_per_replica: 0
  min_replicas: 1
  max_replicas: 1
  proxy_config:
    upstream_base_url: http://upstream.local/v1
    upstream_model_name: opendatalab/MinerU2.5-2509-1.2B
    auth:
      mode: bearer_env
      env_var: OPENAI_API_KEY
    timeout:
      connect_seconds: 3
      read_seconds: 180
      write_seconds: 30
      pool_seconds: 5
    retry:
      max_attempts: 2
      backoff_ms: 200
      retry_on_status: [502, 503, 504]
    streaming:
      enabled: true
      passthrough_sse: true
```

Classification:

- top-level identity/resource fields: `infer-nexus`
- `proxy_config`: `infer-nexus` proxy layer
- this model does not use local `vllm.engine_kwargs`

### 5.6 Repo-ID-Based Loading Without Requiring Local Artifacts

```yaml
- name: mineru-remote
  alias: mineru-remote
  task: chat
  backend: vllm
  compat_mode: vllm_native
  model_loading_config:
    model_id: opendatalab/MinerU2.5-2509-1.2B
    revision: main
  require_local_artifacts: false
  tensor_parallel_size: 1
  cpu_per_replica: 4
  gpu_per_replica: 1
  min_replicas: 1
  max_replicas: 1
  vllm:
    openai_serving:
      enabled: true
```

Important:

- use `model_loading_config.model_id` when you want repo-id-style loading
- use `require_local_artifacts: false` when the model should not require an already existing local path

## 6. Common Misconfigurations

### 6.1 Putting vLLM engine parameters at the wrong level

Common mistake:

```yaml
engine_kwargs:
  trust_remote_code: true
```

This still works today because the schema merges legacy top-level `engine_kwargs` into `vllm.engine_kwargs`, but it is better to write:

```yaml
vllm:
  engine_kwargs:
    trust_remote_code: true
```

Recommended rule:

- do not add new configs using top-level `engine_kwargs`
- use `vllm.engine_kwargs` instead

### 6.2 Forgetting `openai_serving.enabled` in `vllm_native`

For local `chat` and `embedding` models:

```yaml
compat_mode: vllm_native
```

requires:

```yaml
vllm:
  openai_serving:
    enabled: true
```

Otherwise validation fails.

### 6.3 Using `vllm_native` for rerank

This is currently invalid:

```yaml
task: rerank
compat_mode: vllm_native
```

Use the default local path instead.

### 6.4 Writing proxy fields on local models

Do not put:

- `proxy_config.*`

on a local:

```yaml
backend: vllm
```

model.

`proxy_config` only belongs to:

```yaml
backend: vllm_openai_proxy
```

### 6.5 Confusing `model_path` with a path relative to `models.yaml`

Relative `model_path` values are resolved relative to `model_store.root_dir`, not relative to `config/models.yaml`.

Current default model store root is defined in [config/settings.yaml](../config/settings.yaml):

```yaml
model_store:
  root_dir: models
```

So:

```yaml
model_path: Qwen/Qwen3-32B
```

means the runtime resolves it under:

```text
<repo-root>/models/Qwen/Qwen3-32B
```

unless you use an absolute path.

### 6.6 Using unsupported or misleading values

Watch out for these:

- `task: vlm`
  - it exists in the enum, but current local vLLM runtime mapping does not support it as a normal task type
- `compat_mode: strict_openai`
  - still accepted as a backward-compatible alias, but new configs should use `vllm_native`

## 7. Practical Rules for Writing New Entries

When adding a new model, follow this checklist:

1. Decide whether it is local or proxy.
   - local: `backend: vllm`
   - upstream proxy: `backend: vllm_openai_proxy`
2. Decide the task.
   - usually `chat`, `embedding`, or `rerank`
3. Put identity and deployment settings at the top level.
4. Put vLLM engine-specific options under `vllm.engine_kwargs`.
5. Put request defaults under `vllm.request_defaults`.
6. Put compatibility and passthrough policy under `vllm.request_policy` and `vllm.openai_serving`.
7. If using `vllm_native` for local `chat` or `embedding`, enable `vllm.openai_serving.enabled`.
8. If using repo-id loading instead of local artifacts, set `model_loading_config.model_id` and usually `require_local_artifacts: false`.

## 8. Recommended Layout

For maintainability, prefer this layout order for each model:

```yaml
- name: ...
  alias: ...
  served_model_name: ...
  task: ...
  backend: ...
  compat_mode: ...

  model_path: ...
  model_loading_config:
    model_id: ...
    revision: ...
  require_local_artifacts: ...

  dtype: ...
  tensor_parallel_size: ...
  max_model_len: ...
  cpu_per_replica: ...
  gpu_per_replica: ...
  gpu_memory_utilization: ...
  min_replicas: ...
  max_replicas: ...

  capabilities: []
  labels: []

  deployment_config:
    autoscaling_config: {}
    ray_actor_options: {}
    request_router_config: {}

  vllm:
    engine_kwargs: {}
    request_defaults: {}
    request_policy: {}
    openai_serving:
      enabled: false

  proxy_config: {}
```

Not every field is required for every model. The point is to keep the category boundaries stable and obvious.

## 9. Summary

If you only remember one rule, remember this:

- top level is mostly `infer-nexus`
- `vllm.engine_kwargs` is for vLLM engine parameters
- `vllm.request_defaults`, `vllm.request_policy`, and `vllm.openai_serving` are for request/compatibility behavior around vLLM
- `proxy_config` is only for `vllm_openai_proxy`

That separation is the easiest way to avoid most `models.yaml` mistakes.
