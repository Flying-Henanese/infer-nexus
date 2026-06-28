# AGENT.md

## Purpose

This file guides Codex and other implementation agents working in the `infer-nexus` repository.

The source of truth for architectural decisions is [ARCHITECTURE.md](./ARCHITECTURE.md). This file translates that design into implementation rules, guardrails, sequencing, and repository expectations so that code changes stay aligned with the intended system.

## Project Intent

`infer-nexus` is a shared inference service factory for internal development and testing environments.

It exists to replace ad hoc per-user model deployments on shared servers with a centrally managed service that:
- exposes a single API entrypoint
- shares a bounded accelerator pool
- serves pre-registered models
- uses `Ray Serve` for deployment lifecycle and scaling
- uses `vLLM` as the default inference backend
- stays compatible with OpenAI-style client integrations where practical

## Source of Truth

When there is ambiguity, use these priorities:
1. `ARCHITECTURE.md`
2. this `AGENT.md`
3. local code and configuration conventions introduced in this repository

Do not invent architecture that conflicts with `ARCHITECTURE.md`.

## Harness Guidance

This repository uses `.harness/` as an optional Codex collaboration workspace. For complex design, refactor, debugging, or review tasks, read the relevant files under `.harness/` when additional project context, planning templates, checklists, or execution notes would reduce ambiguity.

## Non-Goals

Unless the user explicitly changes scope, do not add:
- Web UI
- dynamic model registration in Phase 1
- tenant-specific hardware isolation
- per-user or per-team scheduling partitions
- multi-backend coexistence
- cross-cluster scheduling
- static per-deployment device pinning

## Core Implementation Rules

### 1. Pre-registered models only
Phase 1 supports only pre-registered models loaded from configuration.

Implications:
- model inventory must live in config files, not hardcoded Python lists
- model registration happens at startup or through a controlled reconcile path
- do not add APIs that let users arbitrarily register models at runtime unless explicitly requested

### 2. One deployment per model
Use one Ray Serve deployment per model in Phase 1.

Implications:
- each model has a distinct deployment identity
- scaling, health, and metrics should remain model-specific
- avoid grouping unrelated models into a single deployment abstraction

### 3. Single shared accelerator pool
The platform uses one shared accelerator pool.

Implications:
- the pool boundary is defined at the Ray runtime boundary, not inside individual model configs
- model configs describe resource demand per replica, not fixed device IDs
- do not implement static `CUDA_VISIBLE_DEVICES` assignment per model or deployment

### 4. Platform-level pool boundary
For CUDA, treat `CUDA_VISIBLE_DEVICES` as a mechanism for defining the overall `infer-nexus` pool visible to Ray, not as a mechanism for manual model pinning.

For Ascend, follow the same principle with the relevant accelerator visibility controls.

Implementation rule:
- pool boundary belongs to deployment or ops startup configuration
- per-replica resource demand belongs to model configuration and Ray Serve deployment settings

### 5. OpenAI-compatible northbound APIs
Northbound APIs must prefer OpenAI-compatible request and response shapes for model inference.

Required first:
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`

Do not let OpenAI compatibility distort the internal architecture. Internal abstractions should still distinguish task types such as `chat`, `embedding`, `rerank`, and future `vlm`.

### 6. Native platform APIs are separate
Platform-native APIs must remain distinct from OpenAI-compatible inference APIs.

Keep separate routes and internal handlers for:
- model discovery
- model status
- cluster load
- cluster capacity
- liveness and readiness

### 7. Admission control is required
Do not implement a gateway that blindly accepts every request.

At minimum, request routing must be able to reject based on:
- model readiness
- model overload
- cluster capacity pressure
- invalid task or model combinations

### 8. Autoscaling should use inference-aware signals
Prefer scaling logic based on:
- queue length
- TTFT
- p95 latency
- inflight request count

Do not design scaling purely around raw CPU or GPU utilization.

### 9. Use Ray Serve as runtime, not as the full control plane
Ray Serve handles:
- deployment lifecycle
- replica routing
- autoscaling support
- runtime metrics exposure

`infer-nexus` itself must provide:
- model catalog
- request routing
- admission control
- platform-native APIs
- runtime state aggregation

### 10. Preserve backend substitution path
Even though Phase 1 uses `vLLM`, do not spread vLLM-specific assumptions across unrelated modules.

Implementation rule:
- keep a thin backend abstraction layer
- isolate engine-specific startup and request adaptation logic
- keep the future path open for `vllm-ascend`

## Required Repository Structure

As implementation begins, prefer this structure unless a strong reason appears to adjust it:

```text
src/
  infer_nexus/
    api/
      openai_routes.py
      platform_routes.py
      health_routes.py
    auth/
      api_keys.py
    backends/
      base.py
      vllm.py
    catalog/
      models.py
      registry.py
      loader.py
    control/
      admission.py
      load_inspector.py
      reconciler.py
      scaler.py
      policies.py
    runtime/
      deployments.py
      serve_app.py
    observability/
      metrics.py
      logging.py
    core/
      config.py
      enums.py
      errors.py
      schemas.py
    main.py
config/
  settings.yaml
  models.yaml
tests/
scripts/
```

Small deviations are acceptable if they improve clarity, but keep the same separation of responsibilities.

## Data and Config Rules

### Config belongs in files
Use configuration files for:
- model inventory
- replica limits
- resource requirements
- backend selection
- cluster and scheduler settings

Do not hardcode these in source unless there is a temporary bootstrap reason.

### Recommended config split
- `config/settings.yaml` for platform-level settings
- `config/models.yaml` for model registry definitions

### Model config expectations
Each model config should be able to describe at least:
- canonical model name
- client-facing alias
- task type
- backend
- model path
- dtype
- tensor parallel size
- max model length
- CPU and accelerator requirements per replica
- min replicas
- max replicas
- optional capabilities and labels

## API Rules

### OpenAI-compatible APIs
Implement in a compatibility-first way, but keep behavior explicit where internal constraints matter.

Rules:
- route requests by `model`
- resolve aliases through the catalog
- return consistent JSON error payloads
- preserve HTTP semantics for unknown model, overload, and backend failure cases

### Native APIs
Expose internal control-plane information without forcing it into OpenAI semantics.

Required initial endpoints:
- `GET /api/catalog/models`
- `GET /api/catalog/models/{model_name}`
- `GET /api/models/{model_name}/status`
- `GET /api/cluster/load`
- `GET /api/cluster/capacity`
- `GET /healthz`
- `GET /readyz`

## Metrics and Observability Rules

Use Ray Serve and Ray metrics where available. Add application-level metrics for business and control-plane visibility.

At minimum, support metrics for:
- total requests
- per-model requests
- request latency
- TTFT
- inflight requests
- queue length
- admission rejections
- model status

Prefer structured logs with fields such as:
- request id
- model name
- task type
- admission decision
- deployment name
- latency summary
- failure reason

## Error Handling Rules

Prefer explicit errors over silent fallback behavior.

Expected error classes include:
- model not found
- model not ready
- model overloaded
- cluster capacity exhausted
- invalid request
- backend runtime failure

Do not hide overload or readiness problems behind generic timeouts if they can be detected earlier.

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

## Change Discipline

When making changes:
- prefer small, architecture-aligned commits
- avoid speculative abstractions unless they clearly preserve known extension paths
- document any intentional deviation from `ARCHITECTURE.md`
- update `ARCHITECTURE.md` and this file if a user-approved design change alters repository expectations

## When in Doubt

If an implementation choice would trade away architecture clarity for short-term convenience, prefer architecture clarity.

If a requested change conflicts with `ARCHITECTURE.md`, stop and surface the conflict explicitly before proceeding.
