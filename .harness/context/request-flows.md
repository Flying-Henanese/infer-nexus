# Request Flows

Use this file to orient yourself before tracing request behavior. Verify details in source before changing code.

## App Startup

1. `scripts/run_gateway.py` starts the FastAPI app from `src/infer_nexus/main.py`.
2. `lifespan()` loads the selected settings file, usually `config/settings.yaml` locally or `config/settings.compose.yaml` in the root Compose flow.
3. `load_model_catalog(settings.catalog.models_path)` loads `config/models.yaml`.
4. `ModelRegistry` indexes model names, aliases, and served model names.
5. `LocalModelStore`, `ServeApplicationBuilder`, `RuntimeExecutor`, `WorkerAdmissionController`, and `RuntimeDispatcher` are created.
6. These objects are attached to `app.state`.
7. `create_app()` registers health, metrics, OpenAI-compatible, compatibility, and platform routers.

## Compose Runtime Startup

1. `docker-compose.yml` starts `ray-head`.
2. `ray-worker` joins the Ray cluster and registers the accelerator budget exposed by the Compose/runtime environment.
3. `serve-deployer` runs `scripts/run_serve_runtime.py --settings config/settings.compose.yaml`, submits Ray Serve apps, waits for readiness, and exits.
4. `gateway` runs `scripts/run_gateway.py --settings config/settings.compose.yaml` and serves the public API.
5. The gateway reaches local models through cached Ray Serve deployment handles.

## Serve Runtime Startup

1. `scripts/run_serve_runtime.py` loads settings and the model catalog.
2. `ServeApplicationBuilder.build_serve_bindings()` builds one Ray Serve application binding per local `backend: vllm` model.
3. `DeploymentFactory` maps each model to deployment kwargs and Ray actor options, including optional `deployment_config.request_router_config` such as a cache-affinity `request_router_class`.
4. Each Ray Serve app is submitted with `serve.run(..., route_prefix=None, blocking=False)`.
5. The script waits for Serve applications to become ready.
6. The gateway later reaches these apps through cached deployment handles.

## `/v1/models`

1. `api/openai_routes.py:list_models()` receives the request.
2. FastAPI injects `ModelRegistry` through `api/deps.py:get_registry()`.
3. The route iterates `registry.list_models()`.
4. Response model ids prefer `served_model_name`, then `alias`, then canonical `name`.
5. No runtime readiness or model artifact check is performed on this path.

## `/v1/chat/completions`

1. `WorkerAdmissionMiddleware` may acquire a process-local gateway slot for the inference path.
2. `api/openai_routes.py:create_chat_completion()` receives a `ChatCompletionsRequest`.
3. The route resolves `request.model` through `ModelRegistry` and requires `task: chat`.
4. `_check_model_ready()` validates local model artifacts for non-proxy models and runs admission checks.
5. `RuntimeDispatcher.dispatch_chat()` resolves a `RuntimeTarget`.
6. `RuntimeExecutor.execute_chat()` chooses proxy, runtime-worker, serve-handle, or local/stub behavior.
7. For local serve mode, the executor gets a cached Ray Serve deployment handle and calls `chat_completion.remote(...)` or `chat_completion_stream.remote(...)`.
8. For proxy mode, the executor rewrites only the model field and forwards to the configured upstream OpenAI-compatible endpoint.
9. Route-level metrics record request status, errors, inflight state, and token usage when available.

## `/v1/embeddings`

1. `api/openai_routes.py:create_embedding()` receives an `EmbeddingRequest`.
2. The route resolves `request.model` through `ModelRegistry` and requires `task: embedding`.
3. `_check_model_ready()` validates local artifacts for non-proxy models and runs admission checks.
4. `RuntimeDispatcher.dispatch_embedding()` resolves a `RuntimeTarget`.
5. `RuntimeExecutor.execute_embedding()` chooses proxy or local/serve execution.
6. Route-level metrics record request status, errors, inflight state, and token usage when available.

## `/v1/rerank` And `/rerank`

1. Both paths use `api/openai_routes.py:create_rerank()`.
2. `_create_rerank_impl()` resolves `request.model` and requires `task: rerank`.
3. `_check_model_ready()` validates local artifacts for non-proxy models and runs admission checks.
4. `RuntimeDispatcher.dispatch_rerank()` resolves a `RuntimeTarget`.
5. `RuntimeExecutor.execute_rerank()` chooses proxy or local/serve execution.
6. Route-level metrics use the actual HTTP path when available.

## Platform Catalog APIs

### `/api/catalog/models`

1. `api/platform_routes.py:list_catalog_models()` receives the request.
2. FastAPI injects `ModelRegistry`.
3. The route returns all loaded catalog entries as `CatalogModelResponse`.
4. This is a read-only catalog view, not a deployment control API.

### `/api/catalog/models/{model_name}`

1. The route resolves `model_name` through canonical name, alias, or served model name.
2. Missing models return 404.
3. Matching catalog config is returned without invoking runtime inference.

### `/api/models/{model_name}/status`

1. The route resolves `model_name` through the registry.
2. `LocalModelStore` checks local artifact availability.
3. Missing artifacts return a degraded status.
4. Present artifacts return catalog status plus a message that runtime status is not connected.

## Cluster View APIs

### `/api/cluster/load`

1. The route injects `ModelRegistry` and `LoadInspector`.
2. `LoadInspector.snapshot(registry)` returns a lightweight current snapshot.
3. This is not a full scheduler, quota, or Ray capacity view.

### `/api/cluster/capacity`

1. The route returns a stub capacity response.
2. It includes the number of registered models.
3. No real capacity backend is connected.

## `/metrics`

1. `api/metrics_routes.py:metrics()` receives the request.
2. It calls `observability.metrics.render_prometheus_metrics()`.
3. It returns Prometheus text format directly from the gateway process.
4. Ray Serve and embedded vLLM metrics should be scraped from their own metrics targets when available.

## `/healthz` And `/readyz`

1. `api/health_routes.py` returns a basic `HealthResponse`.
2. Current readiness is the same as liveness.
3. These paths do not validate Ray Serve, model artifacts, or downstream model readiness.

