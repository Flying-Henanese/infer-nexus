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
- Top-level `engine_kwargs` is still accepted for compatibility. Model validation merges it with `vllm.engine_kwargs`, with values under `vllm.engine_kwargs` taking precedence.

## Enabled Model Pattern

Current enabled models are five local `backend: vllm` entries. Do not assume that proxy models are active just because proxy support exists in code. Tests consume the active catalog directly; check `.harness/checklists/verification.md` for the current catalog-coupled test baseline.

## Placement Rules

- Top-level fields describe infer-nexus identity, task, backend, resources, and deployment settings.
- `vllm.engine_kwargs` holds vLLM engine construction options.
- `vllm.request_defaults` holds default request values.
- `vllm.request_policy` controls allowed OpenAI-style fields.
- `vllm.openai_serving` controls native vLLM OpenAI serving adapter behavior.
- `deployment_config.request_router_config` holds Ray Serve request-router options. Cache affinity is configured here, for example `request_router_class: ray.serve.llm.request_router.PrefixCacheAffinityRouter`.
- `proxy_config` is only for `backend: vllm_openai_proxy`.

## Current Path Mismatch

- Default `config/settings.yaml` resolves relative model paths under the repository-local `models/` directory.
- Both Compose variants mount the model store at `/models` and set `model_store.root_dir: /models`.
- Enabled entries currently use absolute `/app/models/...` values.
- `LocalModelStore` preserves absolute paths instead of resolving them under `root_dir`, so those entries point at neither configured storage location.
- Use paths relative to the configured model-store root, or absolute paths that match the selected runtime mount. This file records the mismatch but does not change `config/models.yaml`.

## Current Non-Goals

- Do not add dynamic model registration unless explicitly requested.
- Do not add multi-file catalog loading unless explicitly requested.
- Do not add heterogeneous placement schema unless explicitly requested.
