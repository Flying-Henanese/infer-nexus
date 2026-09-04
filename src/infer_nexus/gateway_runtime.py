"""Reusable dependency assembly for the HTTP gateway.

The same API routers run under Uvicorn for local debugging and inside a Ray
Serve ingress replica for production inference. This module deliberately
constructs no Ray client connection: a Serve replica already runs in the Ray
cluster data plane and resolves deployment handles locally.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI

from infer_nexus.catalog.loader import load_model_catalog
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.control.admission import AdmissionController
from infer_nexus.control.load_inspector import LoadInspector
from infer_nexus.control.worker_admission import WorkerAdmissionController
from infer_nexus.core.config import Settings
from infer_nexus.core.enums import BackendType
from infer_nexus.model_store import LocalModelStore
from infer_nexus.observability.logging import configure_logging
from infer_nexus.runtime.dispatcher import RuntimeDispatcher
from infer_nexus.runtime.executor import RuntimeExecutor
from infer_nexus.runtime.handles import ServeDeploymentHandleResolver
from infer_nexus.runtime.serve_app import ServeApplicationBuilder


@dataclass(slots=True)
class GatewayRuntime:
    """The complete dependency graph used by one FastAPI gateway instance."""

    settings: Settings
    registry: ModelRegistry
    model_store: LocalModelStore
    admission: AdmissionController
    load_inspector: LoadInspector
    serve_builder: ServeApplicationBuilder
    executor: RuntimeExecutor
    worker_admission: WorkerAdmissionController
    dispatcher: RuntimeDispatcher
    serve_ingress: bool

    @classmethod
    def create(
        cls,
        settings: Settings,
        *,
        handle_resolver: ServeDeploymentHandleResolver | None = None,
        model_targets: dict[str, dict[str, str]] | None = None,
    ) -> "GatewayRuntime":
        """Create one gateway dependency graph without initializing Ray."""
        configure_logging(settings.service.log_level)
        registry = ModelRegistry(load_model_catalog(settings.catalog.models_path))
        model_store = LocalModelStore.from_settings(settings.model_store)
        serve_builder = ServeApplicationBuilder(
            model_store=model_store,
            backend_init_mode=settings.runtime.backend_init_mode,
            service_name=settings.service.name,
        )
        serve_builder.validate_registry_runtime_configs(registry)

        resolver = handle_resolver
        if settings.runtime.execution_mode == "serve" and resolver is None:
            resolver = ServeDeploymentHandleResolver()
        executor = RuntimeExecutor(
            mode=settings.runtime.execution_mode,
            handle_resolver=resolver,
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
        worker_admission = WorkerAdmissionController(
            max_inflight=settings.runtime.gateway_worker_max_inflight,
            retry_after_seconds=settings.runtime.gateway_worker_retry_after_seconds,
        )
        return cls(
            settings=settings,
            registry=registry,
            model_store=model_store,
            admission=AdmissionController(enabled=settings.scheduler.enable_admission_control),
            load_inspector=LoadInspector(),
            serve_builder=serve_builder,
            executor=executor,
            worker_admission=worker_admission,
            dispatcher=RuntimeDispatcher(
                registry=registry,
                serve_builder=serve_builder,
                executor=executor,
                model_targets=model_targets,
            ),
            serve_ingress=model_targets is not None,
        )

    def attach(self, app: FastAPI) -> None:
        """Attach the existing dependency keys consumed by the API routers."""
        app.state.settings = self.settings
        app.state.registry = self.registry
        app.state.model_store = self.model_store
        app.state.admission = self.admission
        app.state.load_inspector = self.load_inspector
        app.state.serve_builder = self.serve_builder
        app.state.serve_plan = self.serve_builder.build_plan(self.registry)
        app.state.runtime_executor = self.executor
        app.state.worker_admission = self.worker_admission
        app.state.runtime_dispatcher = self.dispatcher
        app.state.runtime_model_targets = self.dispatcher.model_target_table()
        app.state.readiness_checker = self.is_ready if self.serve_ingress else None

    def is_ready(self) -> bool:
        """Report model application readiness for the public Serve ingress."""
        if self.settings.runtime.execution_mode != "serve":
            return True
        resolver = self.executor.handle_resolver
        if resolver is None:
            return False
        try:
            applications = resolver.require_serve().status().applications
        except Exception:
            return False
        for target in self.dispatcher.model_target_table().values():
            if target.backend != BackendType.VLLM:
                continue
            application = applications.get(target.app_name)
            if application is None or not _serve_status_is_ready(application):
                return False
        return True

    async def shutdown(self) -> None:
        """Release explicit gateway resources when FastAPI shuts down."""
        # Backend lifetimes belong to their model Serve deployments.
        return None


def _serve_status_is_ready(application: object) -> bool:
    """Return whether one Serve application and its deployments are healthy."""
    application_status = _status_text(getattr(application, "status", application))
    if application_status not in {"RUNNING", "HEALTHY"}:
        return False
    for deployment in getattr(application, "deployments", {}).values():
        if _status_text(getattr(deployment, "status", deployment)) not in {"RUNNING", "HEALTHY"}:
            return False
    return True


def _status_text(status: object) -> str:
    """Normalize Ray's status enum or string without importing Ray types."""
    value = getattr(status, "value", status)
    return str(value).upper().rsplit(".", maxsplit=1)[-1]
