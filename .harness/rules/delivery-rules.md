# Delivery Rules

Read this when planning implementation order, checking Phase 1 scope, or deciding whether work is complete.

The sequence and acceptance criteria below are delivery targets, not a claim that every item is fully implemented. Check `.harness/context/current-architecture.md` for active behavior and known gaps.

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

A complete Phase 1 delivery must satisfy `docs/ARCHITECTURE.md` section 17,
including the following. Interfaces or stubs alone do not complete a requirement:

- models are defined declaratively in config
- the service exposes one API entrypoint
- `/v1/models` works from catalog state
- OpenAI-compatible chat, embeddings, and rerank APIs work
- native catalog, load inspection, liveness, and readiness endpoints work
- resource requirements are declared per model replica
- no static per-model device pinning is introduced
- each locally hosted model has its own Ray Serve deployment; proxy models use their configured upstream
- basic admission control rejects invalid task/model combinations, unavailable models, overload, and cluster capacity pressure explicitly
- inference-metric-driven autoscaling hooks are connected
- Prometheus gateway/runtime guard metrics are exposed
- API key authentication is enforced

Current catalog/API/runtime/metrics paths provide much of this foundation.
Serve ingress readiness and native Ray Serve autoscaling configuration are
wired. Readiness- and cluster-capacity-aware per-request admission, full load
inspection, the project's inference-metric scaling controller, and public API
authentication remain partial or unimplemented. Do not report full Phase 1
acceptance based on the current scaffolding.

## Shared Accelerator Deployment And Validation

For a code change that requires real-runtime validation, follow
`../workflows/remote-deploy-and-validate.md` before reporting completion. The
workflow requires an intentional local commit and push, a remote fast-forward
pull, Compose recreation, and runtime health validation. CUDA is the default;
select the Ascend Compose file and settings for an Ascend target. Keep the same
Compose selection through configuration, deployment, status, and log checks.
