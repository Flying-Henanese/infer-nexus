import sys
import types

import pytest

from infer_nexus.catalog.loader import load_model_catalog
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.runtime.deployments import ModelRuntimeReplica
from infer_nexus.runtime.serve_app import ServeApplicationBuilder


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry(load_model_catalog('config/models.yaml'))


class FakeBoundDeployment:
    def __init__(self, kwargs: dict, replica_cls: type[ModelRuntimeReplica]) -> None:
        self.kwargs = kwargs
        self.replica_cls = replica_cls

    def bind(self, runtime_context: dict) -> dict:
        return {
            'deployment_kwargs': self.kwargs,
            'replica_cls': self.replica_cls,
            'runtime_context': runtime_context,
        }


class FakeServe:
    def deployment(self, **kwargs):
        def wrapper(replica_cls: type[ModelRuntimeReplica]) -> FakeBoundDeployment:
            return FakeBoundDeployment(kwargs, replica_cls)

        return wrapper


def test_serve_builder_plan_contains_all_registered_models(registry: ModelRegistry) -> None:
    builder = ServeApplicationBuilder()

    plan = builder.build_plan(registry)

    assert set(plan) == {
        'qwen3-32b-instruct',
        'bge-large-zh-v1_5',
        'bge-reranker-v2-m3',
    }
    assert plan['qwen3-32b-instruct']['deployment_name'] == 'model-qwen3-32b-instruct'
    assert plan['qwen3-32b-instruct']['ray_actor_options']['num_gpus'] == 4
    assert plan['bge-large-zh-v1_5']['autoscaling_config']['min_replicas'] == 1
    assert plan['bge-reranker-v2-m3']['autoscaling_config']['max_replicas'] == 2


def test_serve_builder_local_dev_summary(registry: ModelRegistry) -> None:
    builder = ServeApplicationBuilder()

    summary = builder.build_local_dev_summary(registry)

    assert summary['models'] == ['qwen3-chat', 'bge-embedding', 'bge-rerank']
    assert summary['deployments'] == [
        'model-qwen3-32b-instruct',
        'model-bge-large-zh-v1_5',
        'model-bge-reranker-v2-m3',
    ]
    assert summary['total_declared_gpu_per_minimum_pool'] == 6


def test_build_runtime_context_contains_backend_spec(registry: ModelRegistry) -> None:
    builder = ServeApplicationBuilder()

    runtime_context = builder.build_runtime_context(registry, 'qwen3-chat')

    assert runtime_context['model_name'] == 'qwen3-32b-instruct'
    assert runtime_context['model_alias'] == 'qwen3-chat'
    assert runtime_context['runtime_spec']['backend'] == 'vllm'
    assert runtime_context['runtime_spec']['tensor_parallel_size'] == 4


def test_build_serve_bindings_from_fake_serve(registry: ModelRegistry) -> None:
    builder = ServeApplicationBuilder()

    bindings = builder.build_serve_bindings(registry, serve=FakeServe())

    qwen_binding = bindings['qwen3-32b-instruct']
    assert qwen_binding['deployment_kwargs']['name'] == 'model-qwen3-32b-instruct'
    assert qwen_binding['deployment_kwargs']['ray_actor_options']['num_gpus'] == 4
    assert qwen_binding['deployment_kwargs']['autoscaling_config'] == {
        'min_replicas': 1,
        'max_replicas': 2,
    }
    assert qwen_binding['runtime_context']['runtime_spec']['model_path'] == 'Qwen/Qwen3-32B-Instruct'


def test_require_ray_serve_raises_without_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = ServeApplicationBuilder()
    monkeypatch.delitem(sys.modules, 'ray', raising=False)

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'ray':
            raise ImportError('ray is not installed')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr('builtins.__import__', fake_import)

    with pytest.raises(RuntimeError, match='Ray Serve is not installed'):
        builder.require_ray_serve()


def test_require_ray_serve_returns_injected_module(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = ServeApplicationBuilder()
    fake_serve = object()
    fake_ray = types.SimpleNamespace(serve=fake_serve)
    monkeypatch.setitem(sys.modules, 'ray', fake_ray)

    assert builder.require_ray_serve() is fake_serve
