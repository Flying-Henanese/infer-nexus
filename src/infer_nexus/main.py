"""FastAPI 应用入口与生命周期装配。"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI

from infer_nexus.api import health_routes, metrics_routes, openai_routes, platform_routes
from infer_nexus.api.worker_admission_middleware import WorkerAdmissionMiddleware
from infer_nexus.catalog.loader import load_model_catalog
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.control.admission import AdmissionController
from infer_nexus.control.load_inspector import LoadInspector
from infer_nexus.control.worker_admission import WorkerAdmissionController
from infer_nexus.core.config import load_settings
from infer_nexus.model_store import LocalModelStore
from infer_nexus.observability.logging import configure_logging
from infer_nexus.runtime.dispatcher import RuntimeDispatcher
from infer_nexus.runtime.executor import RuntimeExecutor
from infer_nexus.runtime.handles import ServeDeploymentHandleResolver
from infer_nexus.runtime.serve_app import ServeApplicationBuilder


def initialize_ray_connection(ray_address: str) -> None:
    """Connect this gateway process to the configured Ray cluster."""
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError(
            "Ray is not installed. Install the 'serve' extra to enable serve execution."
        ) from exc
    ray.init(address=ray_address, ignore_reinit_error=True)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """在应用启动阶段装配运行期依赖，并在关闭时释放生命周期上下文。"""
    # 1) 加载基础配置并初始化日志。
    settings = load_settings(os.getenv("INFER_NEXUS_SETTINGS", "config/settings.yaml"))
    configure_logging(settings.service.log_level)
    # 2) 通过读取./config/models.yaml配置文件构建模型注册表和本地模型仓库。
    registry = ModelRegistry(load_model_catalog(settings.catalog.models_path))
    model_store = LocalModelStore.from_settings(settings.model_store)
    # 3) 生成 Serve 构建器并提前校验所有模型的运行时配置，尽早暴露配置错误。
    serve_builder = ServeApplicationBuilder(
        model_store=model_store,
        backend_init_mode=settings.runtime.backend_init_mode,
        service_name=settings.service.name,
    )
    serve_builder.validate_registry_runtime_configs(registry)
    handle_resolver = None
    # 4) 仅在 serve 模式下准备句柄解析器；stub 模式不依赖 Ray Serve。
    if settings.runtime.execution_mode == "serve":
        if settings.runtime.ray_address:
            initialize_ray_connection(settings.runtime.ray_address)
        handle_resolver = ServeDeploymentHandleResolver()
    runtime_executor = RuntimeExecutor(
        mode=settings.runtime.execution_mode,
        handle_resolver=handle_resolver,
        serve_request_timeout_seconds=settings.runtime.serve_request_timeout_seconds,
        serve_stream_idle_timeout_seconds=settings.runtime.serve_stream_idle_timeout_seconds,
        serve_stream_max_lifetime_seconds=settings.runtime.serve_stream_max_lifetime_seconds,
        max_inflight_per_model=settings.runtime.max_inflight_per_model,
        max_streaming_inflight_per_model=settings.runtime.max_streaming_inflight_per_model,
        max_non_streaming_inflight_per_model=settings.runtime.max_non_streaming_inflight_per_model,
        max_queued_per_model=settings.runtime.max_queued_per_model,
        admission_acquire_timeout_seconds=settings.runtime.admission_acquire_timeout_seconds,
        admission_queue_timeout_seconds=settings.runtime.admission_queue_timeout_seconds,
        circuit_breaker_enabled=settings.runtime.circuit_breaker_enabled,
        circuit_breaker_failure_threshold=settings.runtime.circuit_breaker_failure_threshold,
        circuit_breaker_cooldown_seconds=settings.runtime.circuit_breaker_cooldown_seconds,
    )

    # 5) 将依赖对象挂载到 app.state，供 FastAPI 依赖注入层读取。
    app.state.settings = settings
    app.state.registry = registry
    app.state.model_store = model_store
    app.state.admission = AdmissionController(enabled=settings.scheduler.enable_admission_control)
    app.state.load_inspector = LoadInspector()
    app.state.serve_builder = serve_builder
    app.state.serve_plan = serve_builder.build_plan(registry)
    app.state.runtime_executor = runtime_executor
    app.state.worker_admission = WorkerAdmissionController(
        max_inflight=settings.runtime.gateway_worker_max_inflight,
        retry_after_seconds=settings.runtime.gateway_worker_retry_after_seconds,
    )
    app.state.runtime_dispatcher = RuntimeDispatcher(
        registry=registry,
        serve_builder=serve_builder,
        executor=runtime_executor,
    )
    yield


def create_app() -> FastAPI:
    """创建并注册全部路由的 FastAPI 应用实例。"""
    app = FastAPI(title="infer-nexus", version="0.1.0", lifespan=lifespan)
    app.add_middleware(WorkerAdmissionMiddleware)
    app.include_router(health_routes.router)
    app.include_router(metrics_routes.router)
    app.include_router(openai_routes.router)
    app.include_router(openai_routes.compat_router)
    app.include_router(platform_routes.router)
    return app


app = create_app()
