"""FastAPI 应用入口与生命周期装配。"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI

from infer_nexus.api import health_routes, metrics_routes, openai_routes, platform_routes
from infer_nexus.api.worker_admission_middleware import WorkerAdmissionMiddleware
from infer_nexus.core.config import Settings, load_settings
from infer_nexus.gateway_runtime import GatewayRuntime
from infer_nexus.observability.logging import configure_logging


def initialize_ray_connection(ray_address: str) -> None:
    """Connect this gateway process to the configured Ray cluster."""
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError(
            "Ray is not installed. Install the 'serve' extra to enable serve execution."
        ) from exc
    ray.init(address=ray_address, ignore_reinit_error=True)

def create_app(
    *,
    settings: Settings | None = None,
    runtime: GatewayRuntime | None = None,
    model_targets: dict[str, dict[str, str]] | None = None,
    connect_ray: bool = True,
) -> FastAPI:
    """Create the reusable HTTP API for Uvicorn or a Serve ingress replica."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        active_runtime = runtime
        if active_runtime is None:
            resolved_settings = settings or load_settings(
                os.getenv("INFER_NEXUS_SETTINGS", "config/settings.yaml")
            )
            if (
                connect_ray
                and resolved_settings.runtime.execution_mode == "serve"
                and resolved_settings.runtime.ray_address
            ):
                initialize_ray_connection(resolved_settings.runtime.ray_address)
            active_runtime = GatewayRuntime.create(
                resolved_settings,
                model_targets=model_targets,
            )
        active_runtime.attach(app)
        try:
            yield
        finally:
            await active_runtime.shutdown()

    app = FastAPI(title="infer-nexus", version="0.1.0", lifespan=lifespan)
    app.add_middleware(WorkerAdmissionMiddleware)
    app.include_router(health_routes.router)
    app.include_router(metrics_routes.router)
    app.include_router(openai_routes.router)
    app.include_router(openai_routes.compat_router)
    app.include_router(platform_routes.router)
    return app


app = create_app()
