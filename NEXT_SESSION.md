# Next Session Handoff

## Current Snapshot (2026-05-19)

The codebase has started a proxy-first refactor while keeping local `vllm` backend as fallback.

Current backend options:
- `vllm`: local Ray Serve + `LLM.chat/embed/score`
- `vllm_openai_proxy`: forwards `/v1/*` requests to upstream OpenAI-compatible endpoints

Proxy path is now implemented for:
- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/rerank`

## What Was Recently Changed

1. New backend type and schema
- Added `BackendType.VLLM_OPENAI_PROXY`.
- Added `proxy_config` model schema (`upstream_base_url`, optional model remap, timeout/retry/streaming/header policy/auth).

2. Runtime dispatch split by backend
- Proxy models no longer require local runtime spec construction.
- Serve builder skips deployment/runtime validation for proxy models.

3. Proxy execution path in runtime executor
- Requests are forwarded upstream with minimal transformation (`model` remap only).
- Non-streaming responses are validated into API schemas on success.
- Upstream 4xx/5xx are returned as-is through gateway response objects.
- Chat stream path proxies SSE byte stream.

4. API route behavior updates
- Proxy models skip local model artifact checks.
- Route return types now allow direct `Response` passthrough.

## Known Limitations / Validation Status

- Full pytest suite execution is currently blocked in this workspace due to existing local environment/lockfile issues:
  - `uv` parsing failure on current `pyproject.toml`/`uv.lock` state
  - no local pytest runtime available via `python3 -m pytest`
- Syntax validation was completed successfully:
  - `python3 -m compileall src tests`

## Recommended Next Actions

1. Environment repair for test execution
- Fix local `uv.lock` merge/parse issues.
- Restore a runnable pytest environment.

2. Add first real proxy model in `config/models.yaml`
- Set `backend: vllm_openai_proxy`
- Set `proxy_config.upstream_base_url`
- Set optional `proxy_config.upstream_model_name`

3. Run live smoke tests against real upstream vLLM endpoint
- `chat` non-streaming and streaming
- `embeddings`
- `rerank`

4. Harden proxy path after live validation
- Add upstream host allowlist enforcement
- Add per-model concurrency/timeout guardrails
- Add metrics/logging for upstream latency and status buckets

## Fast Verification Commands

Health:
```bash
curl http://127.0.0.1:8000/healthz
```

Model list:
```bash
curl http://127.0.0.1:8000/v1/models
```

Proxy chat test:
```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mineru",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": false
  }'
```

## Files Most Likely To Touch Next

- `config/models.yaml`
- `src/infer_nexus/runtime/executor.py`
- `src/infer_nexus/runtime/dispatcher.py`
- `src/infer_nexus/api/openai_routes.py`
- `tests/test_api.py`
- `tests/test_dispatcher.py`
