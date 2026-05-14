from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from infer_nexus.api import health_routes, openai_routes, platform_routes
from infer_nexus.catalog.loader import load_model_catalog
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.control.admission import AdmissionController
from infer_nexus.control.load_inspector import LoadInspector
from infer_nexus.core.config import load_settings
from infer_nexus.model_store import LocalModelStore
from infer_nexus.observability.logging import configure_logging
from infer_nexus.runtime.dispatcher import RuntimeDispatcher
from infer_nexus.runtime.executor import RuntimeExecutor
from infer_nexus.runtime.handles import ServeDeploymentHandleResolver
from infer_nexus.runtime.serve_app import ServeApplicationBuilder


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    configure_logging(settings.service.log_level)
    registry = ModelRegistry(load_model_catalog(settings.catalog.models_path))
    model_store = LocalModelStore.from_settings(settings.model_store)
    serve_builder = ServeApplicationBuilder(
        model_store=model_store,
        backend_init_mode=settings.runtime.backend_init_mode,
    )
    serve_builder.validate_registry_runtime_configs(registry)
    handle_resolver = None
    if settings.runtime.execution_mode == "serve":
        handle_resolver = ServeDeploymentHandleResolver(app_name=settings.service.name)
    runtime_executor = RuntimeExecutor(
        mode=settings.runtime.execution_mode,
        handle_resolver=handle_resolver,
    )

    app.state.settings = settings
    app.state.registry = registry
    app.state.model_store = model_store
    app.state.admission = AdmissionController(enabled=settings.scheduler.enable_admission_control)
    app.state.load_inspector = LoadInspector()
    app.state.serve_builder = serve_builder
    app.state.serve_plan = serve_builder.build_plan(registry)
    app.state.runtime_executor = runtime_executor
    app.state.runtime_dispatcher = RuntimeDispatcher(
        registry=registry,
        serve_builder=serve_builder,
        executor=runtime_executor,
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="infer-nexus", version="0.1.0", lifespan=lifespan)
    app.include_router(health_routes.router)
    app.include_router(openai_routes.router)
    app.include_router(openai_routes.compat_router)
    app.include_router(platform_routes.router)
    return app


app = create_app()
