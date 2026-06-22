"""脚本入口行为测试。"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import types

from infer_nexus.core.config import Settings


def load_script_module(module_name: str, relative_path: str):
    """按相对路径动态加载脚本模块。"""
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
    """run_gateway.main 应使用配置并允许 CLI 覆盖 host/port。"""
    settings = Settings()
    settings.service.host = '0.0.0.0'
    settings.service.port = 8000
    settings.service.workers = 2
    captured: dict[str, object] = {}

    monkeypatch.setattr(run_gateway, 'parse_args', lambda: types.SimpleNamespace(
        settings='config/settings.yaml',
        host='127.0.0.1',
        port=9000,
        reload=True,
        workers=1,
    ))
    monkeypatch.setattr(run_gateway, 'load_settings', lambda _path: settings)
    monkeypatch.setattr(
        run_gateway.uvicorn,
        'run',
        lambda app, host, port, reload, workers: captured.update(
            {
                'app': app,
                'host': host,
                'port': port,
                'reload': reload,
                'workers': workers,
            }
        ),
    )
    monkeypatch.delenv('INFER_NEXUS_SETTINGS', raising=False)

    run_gateway.main()

    assert captured == {
        'app': 'infer_nexus.main:app',
        'host': '127.0.0.1',
        'port': 9000,
        'reload': True,
        'workers': 1,
    }
    assert os.environ['INFER_NEXUS_SETTINGS'] == 'config/settings.yaml'


def test_run_serve_runtime_main_deploys_per_model_apps(monkeypatch, prepared_model_store) -> None:
    """run_serve_runtime.main 应按服务名部署 Serve 应用。"""
    settings = Settings()
    settings.service.name = 'infer-nexus'
    settings.model_store.root_dir = str(prepared_model_store)
    settings.runtime.backend_init_mode = 'stub'
    settings.cluster.inference_device_type = 'npu'
    captured: dict[str, object] = {}

    monkeypatch.setattr(run_serve_runtime, 'parse_args', lambda: types.SimpleNamespace(
        settings='config/settings.yaml',
        ray_address='auto',
        proxy_location='Disabled',
        blocking=False,
        ready_timeout_seconds=30.0,
        ready_poll_interval_seconds=0.01,
    ))
    monkeypatch.setattr(run_serve_runtime, 'load_settings', lambda _path: settings)

    def fake_build_bindings(self, registry, *, serve=None, replica_cls=None):
        captured['registry_size'] = len(registry.list_models())
        captured['inference_device_type'] = self.deployment_factory.inference_device_type
        return {
            'qwen3-32b-instruct': {'app': 'qwen'},
            'bge-large-zh-v1_5': {'app': 'embed'},
            'bge-reranker-v2-m3': {'app': 'rerank'},
        }

    monkeypatch.setattr(
        run_serve_runtime.ServeApplicationBuilder,
        'build_serve_bindings',
        fake_build_bindings,
    )
    monkeypatch.setattr(
        run_serve_runtime.ServeApplicationBuilder,
        'build_specs',
        lambda self, registry: [
            types.SimpleNamespace(model_name='qwen3-32b-instruct'),
            types.SimpleNamespace(model_name='bge-large-zh-v1_5'),
            types.SimpleNamespace(model_name='bge-reranker-v2-m3'),
        ],
    )

    class FakeDeploymentStatus:
        def __init__(self, status: str) -> None:
            self.status = status

    class FakeApplicationStatus:
        def __init__(self) -> None:
            self.status = 'RUNNING'
            self.deployments = {'model-qwen3-32b-instruct': FakeDeploymentStatus('HEALTHY')}

    class FakeServeStatus:
        def __init__(self) -> None:
            self.applications = {
                'infer-nexus-model-qwen3-32b-instruct': FakeApplicationStatus(),
                'infer-nexus-model-bge-large-zh-v1_5': FakeApplicationStatus(),
                'infer-nexus-model-bge-reranker-v2-m3': FakeApplicationStatus(),
            }

    captured_runs: list[dict[str, object]] = []
    fake_serve = types.SimpleNamespace(
        start=lambda proxy_location: captured.update({'proxy_location': proxy_location}),
        run=lambda app, name, route_prefix, blocking: captured_runs.append(
            {'app': app, 'name': name, 'route_prefix': route_prefix, 'blocking': blocking}
        ),
        status=lambda: FakeServeStatus(),
    )
    fake_ray = types.ModuleType('ray')
    fake_ray.init = lambda address=None, runtime_env=None: captured.update(
        {'ray_address': address, 'runtime_env': runtime_env}
    )
    fake_ray.serve = fake_serve
    monkeypatch.setitem(__import__('sys').modules, 'ray', fake_ray)

    run_serve_runtime.main()

    assert captured['ray_address'] == 'auto'
    assert captured['proxy_location'] == 'Disabled'
    assert captured['runtime_env']['working_dir'] == '.'
    assert captured['registry_size'] == 3
    assert captured['inference_device_type'] == 'npu'
    assert [item['name'] for item in captured_runs] == [
        'infer-nexus-model-qwen3-32b-instruct',
        'infer-nexus-model-bge-large-zh-v1_5',
        'infer-nexus-model-bge-reranker-v2-m3',
    ]
    assert all(item['route_prefix'] is None for item in captured_runs)
    assert all(item['blocking'] is False for item in captured_runs)
