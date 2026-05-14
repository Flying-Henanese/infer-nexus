from pathlib import Path

from fastapi.testclient import TestClient

from infer_nexus.core.config import Settings
from infer_nexus.main import create_app


def test_app_lifespan_uses_stub_executor_by_default(
    monkeypatch,
    prepared_model_store: Path,
) -> None:
    settings = Settings()
    settings.model_store.root_dir = str(prepared_model_store)

    monkeypatch.setattr('infer_nexus.main.load_settings', lambda: settings)
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
    settings = Settings()
    settings.model_store.root_dir = str(prepared_model_store)
    settings.runtime.execution_mode = 'serve'

    monkeypatch.setattr('infer_nexus.main.load_settings', lambda: settings)
    monkeypatch.setattr('infer_nexus.main.configure_logging', lambda _level: None)

    app = create_app()
    with TestClient(app) as client:
        assert client.app.state.runtime_executor.mode == 'serve'
        assert client.app.state.runtime_executor.handle_resolver is not None
        assert client.app.state.runtime_executor.handle_resolver.app_name == settings.service.name
