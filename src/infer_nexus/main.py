from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from infer_nexus.api import health_routes, openai_routes, platform_routes
from infer_nexus.catalog.loader import load_model_catalog
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.control.admission import AdmissionController
from infer_nexus.control.load_inspector import LoadInspector
from infer_nexus.core.config import load_settings
from infer_nexus.observability.logging import configure_logging
from infer_nexus.runtime.serve_app import ServeApplicationBuilder


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    configure_logging(settings.service.log_level)
    registry = ModelRegistry(load_model_catalog(settings.catalog.models_path))

    app.state.settings = settings
    app.state.registry = registry
    app.state.admission = AdmissionController(enabled=settings.scheduler.enable_admission_control)
    app.state.load_inspector = LoadInspector()
    app.state.serve_builder = ServeApplicationBuilder()
    app.state.serve_plan = app.state.serve_builder.build_plan(registry)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="infer-nexus", version="0.1.0", lifespan=lifespan)
    app.include_router(health_routes.router)
    app.include_router(openai_routes.router)
    app.include_router(platform_routes.router)
    return app


app = create_app()
