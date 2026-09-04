"""主应用生命周期测试。"""

from pathlib import Path

from fastapi.testclient import TestClient

from infer_nexus.core.config import Settings
from infer_nexus.main import create_app
from infer_nexus.runtime.handles import ServeDeploymentHandleResolver


def test_app_lifespan_uses_stub_executor_by_default(
    monkeypatch,
    prepared_model_store: Path,
) -> None:
    """默认配置下应使用 stub 执行器且不创建 serve 句柄解析器。"""
    settings = Settings()
    settings.model_store.root_dir = str(prepared_model_store)
    settings.runtime.gateway_worker_max_inflight = 7
    settings.runtime.gateway_worker_retry_after_seconds = 3

    monkeypatch.setattr('infer_nexus.main.load_settings', lambda _path: settings)
    monkeypatch.setattr('infer_nexus.main.configure_logging', lambda _level: None)

    app = create_app()
    with TestClient(app) as client:
        assert client.app.state.runtime_executor.mode == 'stub'
        assert client.app.state.runtime_executor.handle_resolver is None
        assert client.app.state.runtime_dispatcher.executor is client.app.state.runtime_executor
        assert client.app.state.worker_admission.snapshot.max_inflight == 7
        assert client.app.state.worker_admission.retry_after_seconds == 3


def test_app_lifespan_builds_serve_handle_resolver_in_serve_mode(
    monkeypatch,
    prepared_model_store: Path,
) -> None:
    """serve 模式下应创建 serve 句柄解析器并透传网关保护配置。"""
    settings = Settings()
    settings.model_store.root_dir = str(prepared_model_store)
    settings.runtime.execution_mode = 'serve'
    settings.runtime.gateway_worker_max_inflight = 64
    settings.runtime.max_queued_per_model = 8
    settings.runtime.serve_stream_idle_timeout_seconds = 15

    monkeypatch.setattr('infer_nexus.main.load_settings', lambda _path: settings)
    monkeypatch.setattr('infer_nexus.main.configure_logging', lambda _level: None)

    app = create_app()
    with TestClient(app) as client:
        assert client.app.state.runtime_executor.mode == 'serve'
        assert client.app.state.runtime_executor.handle_resolver is not None
        assert client.app.state.worker_admission.snapshot.max_inflight == 64
        assert client.app.state.runtime_executor.max_queued_per_model == 8
        assert client.app.state.runtime_executor.serve_stream_idle_timeout_seconds == 15


def test_app_lifespan_initializes_ray_when_serve_mode_has_ray_address(
    monkeypatch,
    prepared_model_store: Path,
) -> None:
    """Serve-mode gateway workers connect to Ray once during lifespan startup."""
    settings = Settings()
    settings.model_store.root_dir = str(prepared_model_store)
    settings.runtime.execution_mode = 'serve'
    settings.runtime.ray_address = 'ray-head:6379'
    initialized: list[str] = []

    monkeypatch.setattr('infer_nexus.main.load_settings', lambda _path: settings)
    monkeypatch.setattr('infer_nexus.main.configure_logging', lambda _level: None)
    monkeypatch.setattr('infer_nexus.main.initialize_ray_connection', initialized.append)

    app = create_app()
    with TestClient(app) as client:
        assert client.app.state.runtime_executor.mode == 'serve'
        assert client.app.state.runtime_executor.handle_resolver is not None

    assert initialized == ['ray-head:6379']


def test_prebuilt_gateway_runtime_attaches_without_ray_client_connection(
    monkeypatch,
    prepared_model_store: Path,
) -> None:
    """Serve ingress can reuse the normal API app without calling ``ray.init()``."""
    from infer_nexus.gateway_runtime import GatewayRuntime

    settings = Settings()
    settings.model_store.root_dir = str(prepared_model_store)
    settings.runtime.execution_mode = 'serve'
    settings.runtime.ray_address = 'ray-head:6379'
    initialized: list[str] = []

    monkeypatch.setattr('infer_nexus.main.initialize_ray_connection', initialized.append)
    monkeypatch.setattr('infer_nexus.gateway_runtime.configure_logging', lambda _level: None)

    runtime = GatewayRuntime.create(settings)
    app = create_app(runtime=runtime, connect_ray=False)
    with TestClient(app) as client:
        assert client.app.state.runtime_dispatcher.executor is runtime.executor
        assert client.app.state.runtime_executor.handle_resolver is not None

    assert initialized == []


def test_readyz_rejects_a_serve_gateway_when_a_model_application_is_unavailable(
    monkeypatch,
    prepared_model_store: Path,
) -> None:
    """Serve ingress readiness is distinct from its process-level health check."""
    from infer_nexus.gateway_runtime import GatewayRuntime

    class UnreadyServe:
        def status(self):
            return type("ServeStatus", (), {"applications": {}})()

    settings = Settings()
    settings.model_store.root_dir = str(prepared_model_store)
    settings.runtime.execution_mode = "serve"
    monkeypatch.setattr("infer_nexus.gateway_runtime.configure_logging", lambda _level: None)
    provisional_runtime = GatewayRuntime.create(settings)
    model_targets = provisional_runtime.serve_builder.build_gateway_spec(
        provisional_runtime.registry,
        settings=settings,
    ).model_targets
    runtime = GatewayRuntime.create(
        settings,
        handle_resolver=ServeDeploymentHandleResolver(serve=UnreadyServe()),
        model_targets=model_targets,
    )
    app = create_app(runtime=runtime, connect_ray=False)

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
