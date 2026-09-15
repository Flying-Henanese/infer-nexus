"""主应用生命周期测试。"""

import json
from pathlib import Path
from uuid import UUID

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
    monkeypatch.setattr('infer_nexus.gateway_runtime.configure_logging', lambda *_args: None)

    app = create_app()
    with TestClient(app) as client:
        assert client.app.state.runtime_executor.mode == 'stub'
        assert client.app.state.runtime_executor.handle_resolver is None
        assert client.app.state.runtime_dispatcher.executor is client.app.state.runtime_executor
        assert client.app.state.worker_admission.snapshot.max_inflight == 7
        assert client.app.state.worker_admission.retry_after_seconds == 3


def test_request_logging_returns_canonical_id_and_emits_one_terminal_event(
    prepared_model_store: Path,
    capsys,
) -> None:
    """The outer ASGI lifecycle owns request IDs and terminal application logs."""
    settings = Settings()
    settings.model_store.root_dir = str(prepared_model_store)
    settings.observability.logging.format = "json"
    app = create_app(settings=settings)

    with TestClient(app) as client:
        response = client.get("/v1/models", headers={"X-Request-ID": "req-test-123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-test-123"
    records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if '"event":"request.completed"' in line
    ]
    assert len(records) == 1
    assert records[0]["request_id"] == "req-test-123"
    assert records[0]["route"] == "/v1/models"
    assert records[0]["process_role"] == "gateway"


def test_request_logging_replaces_invalid_id_in_direct_uvicorn_mode(
    prepared_model_store: Path,
) -> None:
    settings = Settings()
    settings.model_store.root_dir = str(prepared_model_store)
    app = create_app(settings=settings)

    with TestClient(app) as client:
        response = client.get("/v1/models", headers={"X-Request-ID": "invalid id"})

    assert response.status_code == 200
    request_ids = response.headers.get_list("X-Request-ID")
    assert len(request_ids) == 1
    assert UUID(hex=request_ids[0]).hex == request_ids[0]
    assert request_ids[0] != "invalid id"


def test_unhandled_error_response_still_returns_request_id(
    prepared_model_store: Path,
    monkeypatch,
) -> None:
    """The outer ASGI middleware wraps generated 500s and preserves correlation."""
    settings = Settings()
    settings.model_store.root_dir = str(prepared_model_store)
    app = create_app(settings=settings)

    async def fail_dispatch(*_args, **_kwargs):
        raise RuntimeError("intentional test failure")

    with TestClient(app, raise_server_exceptions=False) as client:
        monkeypatch.setattr(client.app.state.runtime_dispatcher, "dispatch_chat", fail_dispatch)
        monkeypatch.setattr(client.app.state.model_store, "require_model_path", lambda _model: None)
        chat_model = next(
            model.alias or model.name
            for model in client.app.state.registry.list_models()
            if model.task.value == "chat"
        )
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Request-ID": "req-failed-123"},
            json={
                "model": chat_model,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "req-failed-123"


def test_success_sampling_never_suppresses_request_errors(
    prepared_model_store: Path,
    capsys,
) -> None:
    """The success sample rate only filters ordinary successful requests."""
    settings = Settings()
    settings.model_store.root_dir = str(prepared_model_store)
    settings.observability.logging.format = "json"
    settings.observability.logging.success_sample_rate = 0
    app = create_app(settings=settings)

    with TestClient(app) as client:
        success = client.get("/v1/models", headers={"X-Request-ID": "req-sampled-success"})
        failure = client.post(
            "/v1/chat/completions",
            headers={"X-Request-ID": "req-unsampled-error"},
            json={
                "model": "unknown-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert success.status_code == 200
    assert failure.status_code == 404
    records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if '"event":"request.completed"' in line
    ]
    request_ids = {record.get("request_id") for record in records}
    assert "req-sampled-success" not in request_ids
    assert "req-unsampled-error" in request_ids


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
    monkeypatch.setattr('infer_nexus.gateway_runtime.configure_logging', lambda *_args: None)

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
    monkeypatch.setattr('infer_nexus.gateway_runtime.configure_logging', lambda *_args: None)
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
    monkeypatch.setattr('infer_nexus.gateway_runtime.configure_logging', lambda *_args: None)

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
    monkeypatch.setattr("infer_nexus.gateway_runtime.configure_logging", lambda *_args: None)
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
