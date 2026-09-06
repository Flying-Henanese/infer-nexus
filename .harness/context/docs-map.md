# Architecture Document Map

`docs/` contains long-form reference material. Some files describe current
behavior, while others are future designs, historical debug notes, or superseded
plans.

Use this map to decide which long-form documents are relevant for a task and
whether they describe current implementation. Verify implementation details
against source before changing code.

## Current References

- `docs/ARCHITECTURE.md`
  - Main architecture reference.
  - Still broadly valid for the implemented gateway/catalog/runtime/backend shape.
  - Some implementation status notes may lag the current code; verify against source before citing current behavior.

- `docs/inference_backend_design.md`
  - Current backend adapter reference.
  - Use for `backends/`, `VLLMBackend`, native vLLM serving adapters, and local best-effort behavior.

- `docs/MODELS_YAML_CONFIGURATION_MANUAL.md`
  - Current model catalog configuration manual.
  - Use when editing or explaining `config/models.yaml`.

- `docs/PERFORMANCE_MONITORING_AND_BENCHMARK_PLAN.md`
  - Current monitoring and benchmark status reference.
  - Already separates implemented, partially implemented, and live-validation gaps.

- `docs/RAY_SERVE_VLLM_MONITORING.md`
  - Current runbook for gateway metrics plus Ray Serve hosted vLLM monitoring.

- `docs/DEPLOYMENT_CHECKLIST.md`
  - Current deployment bring-up checklist for Ray Serve validation.

- `docs/KUBERAY_SERVE_NATIVE_GATEWAY_INGRESS.md`
  - Current KubeRay Serve-native ingress topology and controlled-rollout runbook.
  - Use for the NodePort Service, the one-shot RayJob, the HeadOnly proxy invariant, and the one-model smoke catalog.

- `docs/DOCKER_COMPOSE_RUNTIME_SPLIT_PLAN.md`
  - Current reference for the root Compose container lifecycle split.
  - Use for `gateway`, `ray-head`, `ray-worker`, `serve-deployer`, `Dockerfile`, `docker-compose.yml`, and `config/settings.compose.yaml` questions.

- `docs/ASCEND_DEPLOYMENT_CONFIGURATION.md`
  - Current deployment reference for `ascend_deploy/`, `pyproject.ascend.toml`, and `config/settings.ascend-compose.yaml`.
  - It documents the `/models` Compose model-store mount. Model paths remain selected per deployment environment and matching settings file.

## Current But Narrow

- `docs/RAY_SERVE_GATEWAY_PERFORMANCE_PLAN.md`
  - Use for current Ray Serve handle/router risk, gateway backpressure, per-model guards, and future worker isolation.
  - Its current-path analysis is useful; later phases are still future work.

- `docs/OPENAI_PROXY_REFACTOR_PLAN.md`
  - Use as proxy architecture background.
  - Current code supports proxy dispatch for chat, embeddings, and rerank, but production config currently still uses local `backend: vllm` for enabled models such as MinerU.

- `docs/PREFIX_CACHE_STICKY_DESIGN.md`
  - Current narrow reference for the prefix-cache affinity design position.
  - Use when discussing why cache locality should be handled through Ray Serve replica-level routing instead of gateway-owned session stickiness.
  - The supported configuration entrypoint is `config/models.yaml` under `deployment_config.request_router_config`, for example `request_router_class: ray.serve.llm.request_router.PrefixCacheAffinityRouter`.
  - Treat runtime benefit and payload-shape compatibility as environment-dependent; verify against the active Ray Serve version and request path.

- `docs/RAY_SERVE_LLM_EVALUATION_PLAN.md`
  - Current evaluation plan, not an implemented backend.
  - Use when evaluating `ray.serve.llm`, `LLMConfig`, or `build_openai_app`; the active runtime still uses the custom `ModelRuntimeReplica` and backend adapters.

## Future Design Only

Do not treat these as current implementation:

- `docs/DYNAMIC_MODEL_REGISTRATION_DESIGN_AND_PLAN.md`
- `docs/MULTI_NODE_INFERENCE_CONTROL_PLANE.md`
- `docs/VLLM_NATIVE_METRICS_DESIGN.md`

They are useful for future planning, but current source code does not implement their main target capabilities.

## Debug Records

- `docs/DEBUG_REPORT_2026-05-20.md`
- `docs/qwen3_reranker_vllm_notes.md`
- `docs/KUBERAY_CLUSTER_SNAPSHOT_2026-08-28.md`
  - Historical cluster snapshot. Its standalone Uvicorn Gateway description is superseded by `docs/KUBERAY_SERVE_NATIVE_GATEWAY_INGRESS.md`; do not use it as current topology guidance.

Use these for historical troubleshooting context, not as general architecture source of truth.
