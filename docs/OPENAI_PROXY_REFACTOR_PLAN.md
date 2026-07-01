# infer-nexus OpenAI Proxy Refactor Plan

## Remaining Work Only

This document keeps only the items that are still incomplete or need explicit decision. Already implemented proxy wiring, passthrough paths, and supporting tests are omitted on purpose.

## 1. Current Gaps

- `config/models.yaml` has not been migrated to the proxy-first target state for production models such as MinerU.
- Full upstream allowlist validation is still missing.
- The phase-1 `task == chat` restriction is not enforced.
- `forward_authorization` is defined in schema, but its full policy behavior is not implemented.
- Request body size limits are not implemented.
- Stream duration and cancellation hardening are not fully implemented.
- Per-model metrics and SLO histograms/counters are not yet fully implemented.
- Consistency and chaos-style validation are incomplete.
- A broader migration of multimodal models to `vllm_openai_proxy` is still pending.

## 2. Remaining Phase Work

### Phase 1: MinerU non-streaming proxy
- Switch MinerU chat requests to the proxy backend in `config/models.yaml`.
- Validate that the proxy path is the default runtime path for MinerU.
- Confirm the production config uses the intended upstream base URL and auth settings.

### Phase 2: SSE passthrough hardening
- Finish stream metrics.
- Add cancellation handling and long-running stream guardrails.
- Verify client disconnect behavior under load.

### Phase 3: Security and hardening
- Enforce trusted upstream host allowlist.
- Finish request-size limits.
- Finalize auth policy and header-forwarding behavior.
- Run consistency and chaos tests against proxy models.

### Phase 4: Migration
- Migrate additional multimodal models to the proxy backend.
- Keep local `vllm` as the fallback path only.
- Validate that rollback is still a config-only `backend` toggle.

## 3. Open Decisions

1. Should phase 1 proxy support include embeddings and rerank, or stay chat-only?
2. Should upstream auth stay per-model only, or also support a shared provider profile?
3. Should retries apply to all `chat/completions` failures, or only network failures?
4. Should upstream model names be exposed in `/v1/models` metadata for the native API?

## 4. Acceptance Criteria Still to Prove

- MinerU works through `infer-nexus` with no business client code change.
- Proxy-model payload semantics match direct upstream behavior.
- `stream=true` works through the official SDK with stable chunk delivery.
- Rollback to the previous local path remains a single `backend` config change.
