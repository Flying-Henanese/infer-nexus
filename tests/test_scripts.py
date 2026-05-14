from __future__ import annotations

import importlib.util
from pathlib import Path
import types

from infer_nexus.core.config import Settings


def load_script_module(module_name: str, relative_path: str):
    script_path = Path(relative_path)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_gateway = load_script_module('run_gateway_script', 'scripts/run_gateway.py')
run_serve_runtime = load_script_module('run_serve_runtime_script', 'scripts/run_serve_runtime.py')


def test_run_gateway_main_uses_settings_and_cli_overrides(monkeypatch) -> None:
    settings = Settings()
    settings.service.host = '0.0.0.0'
    settings.service.port = 8000
    captured: dict[str, object] = {}

    monkeypatch.setattr(run_gateway, 'parse_args', lambda: types.SimpleNamespace(
        settings='config/settings.yaml',
        host='127.0.0.1',
        port=9000,
        reload=True,
    ))
    monkeypatch.setattr(run_gateway, 'load_settings', lambda _path: settings)
    monkeypatch.setattr(
        run_gateway.uvicorn,
        'run',
        lambda app, host, port, reload: captured.update(
            {
                'app': app,
                'host': host,
                'port': port,
                'reload': reload,
            }
        ),
    )

    run_gateway.main()

    assert captured == {
        'app': 'infer_nexus.main:app',
        'host': '127.0.0.1',
        'port': 9000,
        'reload': True,
    }


def test_run_serve_runtime_main_deploys_named_app(monkeypatch, prepared_model_store) -> None:
    settings = Settings()
    settings.service.name = 'infer-nexus'
    settings.model_store.root_dir = str(prepared_model_store)
    settings.runtime.backend_init_mode = 'stub'
    captured: dict[str, object] = {}

    monkeypatch.setattr(run_serve_runtime, 'parse_args', lambda: types.SimpleNamespace(
        settings='config/settings.yaml',
        ray_address='auto',
        proxy_location='Disabled',
        blocking=False,
    ))
    monkeypatch.setattr(run_serve_runtime, 'load_settings', lambda _path: settings)

    def fake_build_app(self, registry, *, serve=None, replica_cls=None, root_cls=None, root_name='infer-nexus-root'):
        captured['registry_size'] = len(registry.list_models())
        captured['root_name'] = root_name
        return {'app': 'serve'}

    monkeypatch.setattr(
        run_serve_runtime.ServeApplicationBuilder,
        'build_serve_application',
        fake_build_app,
    )

    fake_serve = types.SimpleNamespace(
        start=lambda proxy_location: captured.update({'proxy_location': proxy_location}),
        run=lambda app, name, route_prefix, blocking: captured.update(
            {
                'app': app,
                'name': name,
                'route_prefix': route_prefix,
                'blocking': blocking,
            }
        ),
    )
    fake_ray = types.ModuleType('ray')
    fake_ray.init = lambda address=None: captured.update({'ray_address': address})
    fake_ray.serve = fake_serve
    monkeypatch.setitem(__import__('sys').modules, 'ray', fake_ray)

    run_serve_runtime.main()

    assert captured['ray_address'] == 'auto'
    assert captured['proxy_location'] == 'Disabled'
    assert captured['name'] == 'infer-nexus'
    assert captured['route_prefix'] is None
    assert captured['blocking'] is False
    assert captured['registry_size'] == 3
    assert captured['root_name'] == 'infer-nexus-root'
    assert captured['app'] == {'app': 'serve'}
