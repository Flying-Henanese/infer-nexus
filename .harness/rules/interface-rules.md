# Interface Rules

Read this when a task changes configuration schema, model catalog behavior, HTTP APIs, metrics, logs, or error semantics.

This file is normative. Consult `.harness/context/current-architecture.md`, `model-config.md`, and `monitoring-benchmark.md` for current implementation status and known checked-in inconsistencies.

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

Deployment-specific settings also exist:
- `config/settings.compose.yaml` for the root CUDA Compose flow
- `config/settings.ascend-compose.yaml` for the Ascend Compose flow

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
