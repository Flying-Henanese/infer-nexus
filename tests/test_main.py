"""主应用生命周期测试。"""

from pathlib import Path

from fastapi.testclient import TestClient

from infer_nexus.core.config import Settings
from infer_nexus.main import create_app


def test_app_lifespan_uses_stub_executor_by_default(
    monkeypatch,
    prepared_model_store: Path,
) -> None:
    """默认配置下应使用 stub 执行器且不创建 serve 句柄解析器。"""
    settings = Settings()
    settings.model_store.root_dir = str(prepared_model_store)

    monkeypatch.setattr('infer_nexus.main.load_settings', lambda _path: settings)
    monkeypatch.setattr('infer_nexus.main.configure_logging', lambda _level: None)

    app = create_app()
    with TestClient(app) as client:
        assert client.app.state.runtime_executor.mode == 'stub'
        assert client.app.state.runtime_executor.handle_resolver is None
        assert client.app.state.runtime_dispatcher.executor is client.app.state.runtime_executor


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
        assert client.app.state.runtime_executor.gateway_worker_max_inflight == 64
        assert client.app.state.runtime_executor.max_queued_per_model == 8
        assert client.app.state.runtime_executor.serve_stream_idle_timeout_seconds == 15
