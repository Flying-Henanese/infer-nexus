"""Component tests for the Serve-native Gateway ingress binding."""

from fastapi.testclient import TestClient

from infer_nexus.catalog.loader import load_model_catalog
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.core.config import Settings
from infer_nexus.model_store import LocalModelStore
from infer_nexus.runtime.serve_app import ServeApplicationBuilder


class _BoundDeployment:
    def __init__(self, kwargs: dict) -> None:
        self.kwargs = kwargs

    def bind(self) -> dict[str, dict]:
        return {"deployment_kwargs": self.kwargs}


class FakeServe:
    """Minimal Serve interface for checking ingress construction without Ray."""

    def __init__(self) -> None:
        self.app = None

    def ingress(self, app):
        self.app = app
        return lambda replica_cls: replica_cls

    def deployment(self, **kwargs):
        return lambda replica_cls: _BoundDeployment(kwargs)


def test_gateway_ingress_reuses_the_full_fastapi_surface_without_ray_init() -> None:
    """Health, catalog, and runtime dependencies are attached inside the ASGI ingress app."""
    settings = Settings()
    settings.runtime.execution_mode = "stub"
    registry = ModelRegistry(load_model_catalog(settings.catalog.models_path))
    serve = FakeServe()
    builder = ServeApplicationBuilder(
        model_store=LocalModelStore(settings.model_store.root_dir),
        backend_init_mode=settings.runtime.backend_init_mode,
        service_name=settings.service.name,
    )

    binding = builder.build_gateway_binding(registry, settings=settings, serve=serve)

    assert binding["deployment_kwargs"]["ray_actor_options"] == {"num_cpus": 0.5}
    assert serve.app is not None
    with TestClient(serve.app) as client:
        assert client.get("/healthz").status_code == 200
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert len(models.json()["data"]) == len(registry.list_models())
        assert client.app.state.runtime_executor.handle_resolver is None
        assert set(client.app.state.gateway_model_targets) >= {
            model.alias for model in registry.list_models() if model.alias
        }
