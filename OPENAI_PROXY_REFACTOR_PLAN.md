# infer-nexus OpenAI Proxy Refactor Plan

## Current Branch Status Snapshot

Snapshot basis:
- Branch/worktree inspected on `2026-05-21`
- This section tracks implementation status relative to this plan
- Status meanings:
  - `Implemented`: the planned behavior is already present in the current branch
  - `Partially Implemented`: core path exists, but production hardening or full contract coverage is incomplete
  - `Not Implemented`: not found in the current branch, or still only described in docs

| Plan Area | Status | Notes |
| --- | --- | --- |
| Goal / proxy-first direction | Partially Implemented | Proxy backend exists and is wired into runtime dispatch, but the branch is not yet fully operating as a single production proxy gateway for all target models. |
| Scope: per-model backend routing | Implemented | `vllm` and `vllm_openai_proxy` routing both exist in runtime and API flow. |
| Scope: `chat/completions` proxy path | Implemented | Non-streaming and streaming chat proxy paths are implemented. |
| Scope: per-model upstream config | Partially Implemented | `proxy_config` schema exists. Current production model entries may intentionally remain on local `vllm` when Ray Serve owns the model runtime. |
| Scope: non-streaming passthrough | Implemented | Upstream requests are forwarded with minimal transformation and upstream response bytes/status/content type are returned without schema revalidation. |
| Scope: streaming passthrough | Implemented | SSE path proxies raw bytes through `StreamingResponse`. |
| Scope: error/status passthrough | Implemented | Upstream status/body/content type are preserved; only gateway-stage failures are converted to infer-nexus OpenAI-style errors. |
| Current gap summary | Still Relevant | The architectural motivation is still valid: local `vllm` semantics can diverge from official upstream OpenAI-compatible behavior. |
| Target architecture | Partially Implemented | Backend split and proxy dispatch exist, but migration to proxy-first production operation is still incomplete. |
| Model config design | Partially Implemented | Schema exists, but the example target state is not yet reflected in `config/models.yaml`. |
| New schema contract | Partially Implemented | `ProxyConfig` fields and required `proxy_config` validation exist; allowlist validation and the phase-1 `task == chat` restriction are not implemented. |
| Request passthrough contract | Partially Implemented | `model` rewrite only is implemented; headers/body passthrough is not fully contract-complete. |
| Header policy | Partially Implemented | Request ID and auth injection exist; `forward_authorization` policy is defined in schema but not fully realized as documented. |
| Non-streaming response contract | Implemented | Status/body/content type are preserved and `X-Infer-Nexus-Request-ID` is added for observability. |
| Streaming response contract | Implemented | Raw SSE byte passthrough is present and content type is preserved from upstream response headers. |
| Error policy | Partially Implemented | Unknown model/task mismatch and proxy timeout mappings exist; broader gateway error taxonomy can still be refined. |
| Routing and fallback strategy | Implemented | Proxy and local backends are explicitly separated, and there is no automatic proxy-to-local fallback. |
| Observability and SLO metrics | Not Implemented | Full per-model metrics, latency histograms, and stream counters described here are not yet evident in the branch. |
| Security controls | Partially Implemented | Per-model auth from env/static token exists; strict upstream allowlist, body size limits, and stream duration guardrails are not yet evident. |
| Incremental delivery plan | Partially Implemented | Phase 0-2 core mechanics are largely present; Phase 3-4 hardening and migration remain incomplete. |
| Test matrix | Partially Implemented | Proxy dispatcher tests cover model rewrite and basic passthrough for chat/embeddings/rerank, but consistency, allowlist, disconnect propagation, and chaos-style tests are not complete. |
| Acceptance criteria | Partially Implemented | Some criteria are mechanically satisfied, but real `mineru` config migration and production-equivalent validation are still incomplete. |
| Open decisions | Mostly Superseded | One decision is effectively resolved in code: proxy support already extends beyond chat-only to embeddings and rerank. |

## Current Branch Status by Phase

| Phase | Status | Notes |
| --- | --- | --- |
| Phase 0: schema/config prep | Implemented | `vllm_openai_proxy` and `proxy_config` schema are in place. |
| Phase 1: MinerU non-streaming proxy | Partially Implemented | Runtime support exists, but `mineru` is not yet switched to proxy in `config/models.yaml`. |
| Phase 2: SSE passthrough | Partially Implemented | SSE passthrough exists, but the metrics/cancellation-hardening parts are not fully evident. |
| Phase 3: hardening | Partially Implemented | Retry and auth pieces exist; allowlist, request-size limits, and full consistency/chaos validation are not complete. |
| Phase 4: migration | Not Implemented | The config and rollout state do not yet show broad migration of multimodal models to proxy backend. |

## 1. Goal

Make `infer-nexus` behave like a model service provider gateway:

- Client uses one endpoint: `base_url=http(s)://infer-nexus/v1`
- Client selects model only via `model=...`
- Gateway handles platform controls (auth/rate-limit/routing/observability)
- Model protocol semantics come from upstream OpenAI-compatible servers (vLLM official `api_server`)

Core principle:
- Prefer transparent proxying over protocol re-implementation.

## 2. Scope and Non-Scope

In scope:
- `chat/completions` proxy path first
- Per-model backend routing (`vllm` vs `vllm_openai_proxy`)
- Per-model upstream config in `config/models.yaml`
- Non-streaming and streaming passthrough
- Error/status passthrough policy

Out of scope (first iteration):
- Dynamic model registration
- Gateway-managed upstream instance pools for one model. Replica pooling, health, and autoscaling are owned by the upstream serving layer such as Ray Serve.
- Full `responses` API support

## 3. Current Gap Summary

Current `vllm` backend path:
- Local schema validation -> local message normalization -> local `LLM.chat(...)` -> local response adaptation

Problem:
- Not equivalent to official `vllm.entrypoints.openai.api_server` behavior
- Multimodal/template/parser/tool fields can diverge, especially for models like MinerU

## 4. Target Architecture

```text
Client SDK
  -> infer-nexus /v1/chat/completions
      -> model lookup (alias / served_model_name / name)
      -> backend dispatch
          -> vllm_openai_proxy:
                forward to upstream /v1/chat/completions
                passthrough response (JSON or SSE)
          -> vllm (fallback / legacy):
                existing local LLM.chat path
```

## 5. Model Config Design (`config/models.yaml`)

Keep existing generic fields and add backend-specific `proxy_config`.

Example:

```yaml
models:
  - name: mineru
    alias: mineru
    task: chat
    backend: vllm_openai_proxy
    served_model_name: mineru
    proxy_config:
      upstream_base_url: http://10.0.0.21:8001/v1
      upstream_model_name: opendatalab/MinerU2.5-2509-1.2B
      auth:
        mode: bearer_env
        env_var: MINERU_UPSTREAM_API_KEY
      timeout:
        connect_seconds: 3
        read_seconds: 180
        write_seconds: 30
        pool_seconds: 5
      retry:
        max_attempts: 1
        backoff_ms: 0
        retry_on_status: [502, 503, 504]
      streaming:
        enabled: true
        passthrough_sse: true
      headers_policy:
        pass_request_id: true
        forward_authorization: false
```

## 6. New Schema Contract

Add `ProxyConfig` in model schema layer:

- `upstream_base_url: str` (required)
- `upstream_model_name: str | None` (optional; fallback to requested model name)
- `auth`:
  - `mode: "none" | "bearer_env" | "static_bearer"`
  - `env_var: str | None`
  - `token: str | None` (discouraged for prod)
- `timeout`:
  - `connect_seconds: int | float > 0`
  - `read_seconds: int | float > 0`
  - `write_seconds: int | float > 0`
  - `pool_seconds: int | float > 0`
- `retry`:
  - `max_attempts: int >= 1`
  - `backoff_ms: int >= 0`
  - `retry_on_status: list[int]`
- `streaming`:
  - `enabled: bool`
  - `passthrough_sse: bool`
- `headers_policy`:
  - `pass_request_id: bool`
  - `forward_authorization: bool`

Validation rules:
- `backend == vllm_openai_proxy` requires `proxy_config`.
- `task` must be `chat` in phase 1 proxy rollout.
- `upstream_base_url` must be in trusted host allowlist.

Scope note:
- `proxy_config` describes a single upstream OpenAI-compatible endpoint.
- If that endpoint is backed by Ray Serve, Ray Serve owns replica pools, autoscaling, and replica health. The gateway should not duplicate that instance-pool logic in this phase.

## 7. Request/Response Passthrough Contract

### 7.1 Request handling

For proxy backend, gateway should:

1. Parse request minimally for routing (`model` only).
2. Keep original JSON body as source of truth.
3. Replace only `model` field when `upstream_model_name` is configured.
4. Do not rewrite `messages`, `tools`, `response_format`, `extra_body`, etc.

### 7.2 Headers handling

Forward:
- `Content-Type`
- `Accept`
- `X-Request-ID` (or gateway-generated trace id)

Conditional:
- `Authorization` only if policy allows.

Drop:
- `Host`
- hop-by-hop headers

### 7.3 Non-streaming response

- Preserve upstream status code.
- Preserve upstream response body as-is.
- Add `X-Infer-Nexus-Request-ID` header for observability.

### 7.4 Streaming response (`stream=true`)

- Proxy raw SSE bytes.
- Do not parse/reassemble SSE chunks.
- Preserve content type `text/event-stream`.

## 8. Error Policy

Priority:
1. Upstream HTTP status and error payload.
2. Gateway-generated errors only for gateway-stage failures.

Gateway-stage error examples:
- unknown model -> `404 model_not_found`
- backend misconfig -> `500 backend_misconfigured`
- upstream connect timeout -> `504 upstream_timeout`
- allowlist violation -> `502 upstream_not_allowed`

For upstream errors:
- keep status/body unchanged
- append trace header only

## 9. Routing and Fallback Strategy

Backend dispatch matrix:

- `backend = vllm_openai_proxy` -> proxy executor path
- `backend = vllm` -> existing local backend path

Proxy routing:
- A proxy model forwards to exactly one configured `upstream_base_url`.
- Gateway-managed multi-upstream routing, circuit breaking, and per-instance ejection are intentionally out of scope for this phase.
- When the upstream endpoint is Ray Serve-backed, Ray Serve handles model replica routing, health, and autoscaling behind that endpoint.

Fallback policy:
- default no automatic fallback from proxy to local backend (avoid semantic surprise)
- rollback via config switch (`backend` toggle)

## 10. Observability and SLO Metrics

Per-model metrics:
- request_count
- success_count
- error_count_by_status
- upstream_latency_ms (p50/p95/p99)
- timeout_count
- stream_open_count
- stream_error_count

Log fields:
- request_id
- public_model_name
- upstream_model_name
- backend
- upstream_host
- status_code
- latency_ms
- stream(bool)

## 11. Security Controls

- Upstream host allowlist (strict)
- Optional per-model auth secret from environment only
- Header forwarding policy per model
- Max request body size limit
- Read timeout and max stream duration guardrails

## 12. Incremental Delivery Plan

### Phase 0: Design + config/schema prep
- add schema and validation for `vllm_openai_proxy` and `proxy_config`
- no runtime behavior change

### Phase 1: MinerU non-streaming proxy
- route MinerU chat requests to upstream
- preserve JSON response
- basic metrics and trace headers

### Phase 2: SSE passthrough
- implement `stream=true` passthrough
- add stream metrics and cancellation handling

### Phase 3: hardening
- retries against the same upstream endpoint, allowlist, auth policy, request size limits
- run consistency and chaos tests

### Phase 4: migration
- migrate more multimodal models to proxy backend
- keep local `vllm` as fallback path only

## 13. Test Matrix

Functional:
- model routing by alias/name/served_model_name
- upstream model rewrite correctness
- JSON passthrough for standard chat request
- multimodal payload passthrough (`image_url`, `data:` URL)
- tool call fields passthrough

Streaming:
- SSE passthrough correctness
- client disconnect propagation
- long-running stream timeout behavior

Error:
- upstream 4xx passthrough
- upstream 5xx passthrough
- connect/read timeout mapping
- allowlist reject path

Consistency:
- compare direct upstream vs infer-nexus proxy outputs under same request
- allow differences only in gateway-added headers and request id

## 14. Acceptance Criteria

- MinerU works through infer-nexus with no business client code change.
- For proxy models, payload semantics match direct upstream behavior.
- `stream=true` works via official SDK with stable chunk delivery.
- Rollback to previous local path is one config change (`backend`).

## 15. Open Decisions for Review

1. Should proxy phase 1 include embeddings/rerank or chat-only?
2. Should upstream auth be per-model only, or support shared provider profile?
3. Is automatic retry needed for `chat/completions`, or network-failure-only?
4. Should we expose upstream model names in `/v1/models` metadata (native API only)?

