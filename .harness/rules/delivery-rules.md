# Delivery Rules

Read this when planning implementation order, checking Phase 1 scope, or deciding whether work is complete.

## Development Sequence

Unless the user asks otherwise, build in this order:

1. project scaffolding with `uv`
2. configuration schema and loader
3. model catalog models and registry
4. FastAPI app with health and catalog routes
5. OpenAI-compatible `/v1/models`
6. Ray Serve runtime skeleton
7. backend abstraction and vLLM backend skeleton
8. admission control interfaces
9. metrics and load inspection skeleton
10. chat and embedding request paths
11. scaling and reconciliation hooks

This sequence keeps the implementation aligned with architecture and reduces rework.

## Phase 1 Acceptance Criteria

A Phase 1 implementation should satisfy these conditions:
- models are defined declaratively in config
- the service exposes one API entrypoint
- `/v1/models` works from catalog state
- native catalog and health endpoints work
- resource requirements are declared per model replica
- no static per-model device pinning is introduced
- a path exists for Ray Serve-backed deployments
- admission and load-awareness have explicit interfaces, even if some behavior is initially stubbed
