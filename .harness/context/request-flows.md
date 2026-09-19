# Request Flows

Use this map for cross-component behavior. Verify the details in source before
changing a route, middleware, deployment, or backend.

## Startup Modes

- `src/infer_nexus/main.py` loads the selected settings and the single model
  catalog, builds the registry, model store, Serve builder, executor, worker
  admission controller, and dispatcher, then attaches them to `app.state`.
- `scripts/run_gateway.py` starts FastAPI for local or standalone debugging.
  In Compose Serve mode, the same app runs inside
  `InferNexusGatewayIngress`; there is no separate Uvicorn gateway service.
- Both Compose profiles start `ray-head`, `ray-worker`, and one-shot
  `serve-deployer`. `scripts/run_serve_runtime.py` submits one Serve
  application per local `backend: vllm` model, waits for them, then submits
  the CPU-only Gateway application at `/`. The public HTTP proxy is on
  `ray-head:8000`.
- Root Compose selects `config/settings.compose.yaml` and GPU resources.
  Ascend Compose selects `config/settings.ascend-compose.yaml` and custom
  Ray `NPU` resources. The local `scripts/start_minimal_ascend.sh` still
  defaults to `config/settings.yaml`; pass
  `--settings config/settings.ascend-compose.yaml` for an Ascend launch.

## Shared Inference Path

1. In Serve mode, `RequestIdProxyMiddleware` normalizes inbound
   `X-Request-ID` before Ray captures it. Direct Uvicorn runs resolve the ID
   in `RequestLoggingMiddleware` instead.
2. The Serve proxy forwards to `InferNexusGatewayIngress`, whose FastAPI app
   runs request logging outside process-local worker admission. A full
   Gateway worker guard can reject with HTTP 503 before a route runs.
3. An OpenAI-compatible route resolves the requested model through
   `ModelRegistry`, checks its task type, validates local artifacts where
   applicable, and calls the model-level admission hook.
4. `RuntimeDispatcher` selects a target. `RuntimeExecutor` applies its
   per-model guards and chooses a local Serve handle, configured upstream
   proxy, or local/stub path. The runtime-worker client protocol exists but
   is not injected by `main.py`; it is not an active isolation path.
5. A local Serve handle passes an allowlisted `RequestContext` to
   `ModelRuntimeReplica` and its replica-local vLLM backend. Proxy execution
   rewrites the model field and forwards the request ID according to
   `headers_policy.pass_request_id`.
6. The outer request lifecycle records one terminal request event and
   finalizes gateway metrics after the response body ends. Streaming also
   emits one stream terminal event. Explicit admission rejections increment
   the cross-layer admission metric once; the runtime guard separately
   records its layer-specific counter.

Ray Serve can reject a saturated public Gateway queue with HTTP 503 before
FastAPI middleware or gateway metrics run. Inspect
`serve_deployment_queued_queries` for that outer pressure.

## Endpoint Differences

| Endpoint | Distinction |
| --- | --- |
| `GET /v1/models` | Lists catalog model IDs, preferring served name then alias then canonical name; it does not check runtime readiness. |
| `POST /v1/chat/completions` | Requires a chat model and supports unary or streaming execution. |
| `POST /v1/embeddings` | Requires an embedding model. |
| `POST /v1/rerank`, `POST /rerank` | Require a rerank model and share the same implementation. |

## Platform, Metrics, And Health

- `api/platform_routes.py` exposes read-only catalog, model status, load,
  and capacity views. Model status checks local artifacts but does not
  connect to full runtime state; load is lightweight and capacity is stubbed.
- `GET /metrics` returns Gateway Prometheus metrics. Ray Serve and embedded
  vLLM metrics need their own metrics targets when available.
- `GET /healthz` checks basic process liveness. In Serve mode,
  `GET /readyz` returns 503 until every configured local model application
  and deployment is healthy; local/stub mode returns a basic success response.

See `.harness/context/monitoring-benchmark.md` for metric ownership and
`.harness/checklists/verification.md` for live validation boundaries.
