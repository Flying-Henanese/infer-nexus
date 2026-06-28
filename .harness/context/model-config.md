# Model Configuration Baseline

Use this file before editing `config/models.yaml`.

## Source Of Truth

- Schema: `src/infer_nexus/catalog/models.py`
- Loader: `src/infer_nexus/catalog/loader.py`
- Registry: `src/infer_nexus/catalog/registry.py`
- Manual: `docs/MODELS_YAML_CONFIGURATION_MANUAL.md`
- Active catalog: `config/models.yaml`

## Current Behavior

- The application loads one model catalog file from `settings.catalog.models_path`.
- There is no active `config/models.d/` loading path.
- Registry lookup supports canonical `name`, `alias`, and `served_model_name`.
- Local models use `backend: vllm`.
- Proxy models use `backend: vllm_openai_proxy` and require `proxy_config`.

## Enabled Model Pattern

Current enabled models are local `backend: vllm` entries. Do not assume that proxy models are active just because proxy support exists in code.

## Placement Rules

- Top-level fields describe infer-nexus identity, task, backend, resources, and deployment settings.
- `vllm.engine_kwargs` holds vLLM engine construction options.
- `vllm.request_defaults` holds default request values.
- `vllm.request_policy` controls allowed OpenAI-style fields.
- `vllm.openai_serving` controls native vLLM OpenAI serving adapter behavior.
- `proxy_config` is only for `backend: vllm_openai_proxy`.

## Current Non-Goals

- Do not add dynamic model registration unless explicitly requested.
- Do not add multi-file catalog loading unless explicitly requested.
- Do not add heterogeneous placement schema unless explicitly requested.

