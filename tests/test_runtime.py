import sys
import types
import asyncio

import pytest

from infer_nexus.catalog.loader import load_model_catalog
from infer_nexus.catalog.models import ModelCatalogFile, ModelConfig
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.core.errors import BackendConfigurationError, BackendRequestValidationError
from infer_nexus.core.enums import TaskType
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest
from infer_nexus.backends.vllm import VLLMBackend
from infer_nexus.model_store import LocalModelStore
from infer_nexus.runtime.deployments import ModelRuntimeReplica
from infer_nexus.runtime.serve_app import ServeApplicationBuilder


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry(load_model_catalog('config/models.yaml'))


@pytest.fixture
def model_store() -> LocalModelStore:
    return LocalModelStore('models')


class FakeBoundDeployment:
    def __init__(self, kwargs: dict, replica_cls: type[ModelRuntimeReplica]) -> None:
        self.kwargs = kwargs
        self.replica_cls = replica_cls

    def bind(self, *args, **kwargs) -> dict:
        return {
            'deployment_kwargs': self.kwargs,
            'replica_cls': self.replica_cls,
            'args': args,
            'kwargs': kwargs,
        }


class FakeServe:
    def deployment(self, **kwargs):
        def wrapper(replica_cls: type[ModelRuntimeReplica]) -> FakeBoundDeployment:
            return FakeBoundDeployment(kwargs, replica_cls)

        return wrapper


def test_serve_builder_plan_contains_all_registered_models(
    registry: ModelRegistry,
    model_store: LocalModelStore,
) -> None:
    builder = ServeApplicationBuilder(model_store=model_store)

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


def test_serve_builder_local_dev_summary(
    registry: ModelRegistry,
    model_store: LocalModelStore,
) -> None:
    builder = ServeApplicationBuilder(model_store=model_store)

    summary = builder.build_local_dev_summary(registry)

    assert summary['models'] == ['qwen3-chat', 'bge-embedding', 'bge-rerank']
    assert summary['deployments'] == [
        'model-qwen3-32b-instruct',
        'model-bge-large-zh-v1_5',
        'model-bge-reranker-v2-m3',
    ]
    assert summary['model_store_root'].endswith('/models')
    assert summary['total_declared_gpu_per_minimum_pool'] == 6


def test_build_runtime_context_contains_resolved_local_model_path(
    registry: ModelRegistry,
    model_store: LocalModelStore,
) -> None:
    builder = ServeApplicationBuilder(model_store=model_store)

    runtime_context = builder.build_runtime_context(registry, 'qwen3-chat')

    assert runtime_context['model_name'] == 'qwen3-32b-instruct'
    assert runtime_context['model_alias'] == 'qwen3-chat'
    assert runtime_context['resolved_model_path'].endswith('/models/Qwen/Qwen3-32B-Instruct')
    assert runtime_context['runtime_spec']['backend'] == 'vllm'
    assert runtime_context['runtime_spec']['model_path'].endswith('/models/Qwen/Qwen3-32B-Instruct')
    assert runtime_context['runtime_spec']['tensor_parallel_size'] == 4


def test_build_serve_bindings_from_fake_serve(
    registry: ModelRegistry,
    model_store: LocalModelStore,
) -> None:
    builder = ServeApplicationBuilder(model_store=model_store)

    bindings = builder.build_serve_bindings(registry, serve=FakeServe())

    qwen_binding = bindings['qwen3-32b-instruct']
    assert qwen_binding['deployment_kwargs']['name'] == 'model-qwen3-32b-instruct'
    assert qwen_binding['deployment_kwargs']['ray_actor_options']['num_gpus'] == 4
    assert qwen_binding['deployment_kwargs']['autoscaling_config'] == {
        'min_replicas': 1,
        'max_replicas': 2,
    }
    assert qwen_binding['args'][0]['runtime_spec']['model_path'].endswith(
        '/models/Qwen/Qwen3-32B-Instruct'
    )


def test_build_serve_application_wraps_all_model_bindings(
    registry: ModelRegistry,
    model_store: LocalModelStore,
) -> None:
    builder = ServeApplicationBuilder(model_store=model_store)

    app = builder.build_serve_application(registry, serve=FakeServe())

    assert app['deployment_kwargs']['name'] == 'infer-nexus-root'
    assert set(app['kwargs']) == {
        'qwen3-32b-instruct',
        'bge-large-zh-v1_5',
        'bge-reranker-v2-m3',
    }


def test_require_ray_serve_raises_without_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = ServeApplicationBuilder(model_store=LocalModelStore('models'))
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
    builder = ServeApplicationBuilder(model_store=LocalModelStore('models'))
    fake_serve = object()
    fake_ray = types.SimpleNamespace(serve=fake_serve)
    monkeypatch.setitem(sys.modules, 'ray', fake_ray)

    assert builder.require_ray_serve() is fake_serve


def test_model_runtime_replica_calls_backend_for_chat(
    registry: ModelRegistry,
    model_store: LocalModelStore,
) -> None:
    builder = ServeApplicationBuilder(model_store=model_store)
    runtime_context = builder.build_runtime_context(registry, 'qwen3-chat')
    replica = ModelRuntimeReplica(runtime_context)

    response = asyncio.run(
        replica.chat_completion(
            ChatCompletionsRequest(
                model='qwen3-chat',
                messages=[{'role': 'user', 'content': 'hello'}],
            ).model_dump(mode='json')
        )
    )

    assert response['status'] == 'ok'
    assert response['backend'] == 'vllm'
    assert response['deployment'] == 'model-qwen3-32b-instruct'
    assert 'backend stub response from vllm' in response['content']
    assert 'temperature=0.7' in response['content']
    assert 'top_p=1.0' in response['content']
    assert 'max_tokens=512' in response['content']


def test_model_runtime_replica_calls_backend_for_embedding(
    registry: ModelRegistry,
    model_store: LocalModelStore,
) -> None:
    builder = ServeApplicationBuilder(model_store=model_store)
    runtime_context = builder.build_runtime_context(registry, 'bge-embedding')
    replica = ModelRuntimeReplica(runtime_context)

    response = asyncio.run(
        replica.embedding(
            EmbeddingRequest(
                model='bge-embedding',
                input='hello',
            ).model_dump(mode='json')
        )
    )

    assert response['status'] == 'ok'
    assert response['backend'] == 'vllm'
    assert response['deployment'] == 'model-bge-large-zh-v1_5'
    assert response['data'][0]['embedding'] == [5.0, 0.0, 4.0]


def test_model_runtime_replica_calls_backend_for_embedding_base64(
    registry: ModelRegistry,
    model_store: LocalModelStore,
) -> None:
    builder = ServeApplicationBuilder(model_store=model_store)
    runtime_context = builder.build_runtime_context(registry, 'bge-embedding')
    replica = ModelRuntimeReplica(runtime_context)

    response = asyncio.run(
        replica.embedding(
            EmbeddingRequest(
                model='bge-embedding',
                input='hello',
                encoding_format='base64',
            ).model_dump(mode='json')
        )
    )

    assert response['status'] == 'ok'
    assert response['backend'] == 'vllm'
    assert response['deployment'] == 'model-bge-large-zh-v1_5'
    assert isinstance(response['data'][0]['embedding'], str)


def test_vllm_backend_builds_sampling_params_and_text_messages() -> None:
    backend = VLLMBackend({})
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
        temperature=0.2,
        top_p=0.9,
        max_tokens=128,
    )

    assert backend._build_sampling_params(request) == {
        'temperature': 0.2,
        'top_p': 0.9,
        'max_tokens': 128,
    }
    assert backend._build_chat_messages(request) == [{'role': 'user', 'content': 'hello'}]


def test_vllm_backend_rejects_streaming_and_multimodal_messages() -> None:
    backend = VLLMBackend({})
    streaming = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
        stream=True,
    )
    multimodal = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]}],
    )

    with pytest.raises(BackendRequestValidationError, match='Streaming chat completions'):
        backend._build_chat_messages(streaming)

    with pytest.raises(BackendRequestValidationError, match='Only text chat messages'):
        backend._build_chat_messages(multimodal)


def test_vllm_backend_normalizes_embedding_inputs_and_rejects_empty() -> None:
    backend = VLLMBackend({})

    assert backend._normalize_embedding_inputs(
        EmbeddingRequest(model='bge-embedding', input='hello')
    ) == ['hello']
    assert backend._normalize_embedding_inputs(
        EmbeddingRequest(model='bge-embedding', input=['hello', 'world'])
    ) == ['hello', 'world']

    with pytest.raises(BackendRequestValidationError, match='at least one input'):
        backend._normalize_embedding_inputs(
            EmbeddingRequest(model='bge-embedding', input=[])
        )


def test_model_runtime_replica_calls_backend_for_rerank(
    registry: ModelRegistry,
    model_store: LocalModelStore,
) -> None:
    builder = ServeApplicationBuilder(model_store=model_store)
    runtime_context = builder.build_runtime_context(registry, 'bge-rerank')
    replica = ModelRuntimeReplica(runtime_context)

    response = asyncio.run(
        replica.rerank(
            RerankRequest(
                model='bge-rerank',
                query='capital of france',
                documents=[
                    'The capital of Brazil is Brasilia.',
                    'The capital of France is Paris.',
                    'Python is a programming language.',
                ],
                top_n=2,
            ).model_dump(mode='json')
        )
    )

    assert response['status'] == 'ok'
    assert response['backend'] == 'vllm'
    assert response['deployment'] == 'model-bge-reranker-v2-m3'
    assert len(response['results']) == 2
    assert response['results'][0]['index'] == 1
    assert response['results'][0]['document']['text'] == 'The capital of France is Paris.'


def test_vllm_backend_normalizes_rerank_documents_and_rejects_empty() -> None:
    backend = VLLMBackend({})

    assert backend._normalize_rerank_documents(
        RerankRequest(model='bge-rerank', query='q', documents='doc')
    ) == ['doc']
    assert backend._normalize_rerank_documents(
        RerankRequest(model='bge-rerank', query='q', documents=['a', 'b'])
    ) == ['a', 'b']

    with pytest.raises(BackendRequestValidationError, match='at least one document'):
        backend._normalize_rerank_documents(
            RerankRequest(model='bge-rerank', query='q', documents=[])
        )


def test_vllm_backend_rejects_runtime_spec_task_mode_mismatch() -> None:
    backend = VLLMBackend({})
    runtime_context = {
        'model_name': 'bge-large-zh-v1_5',
        'task': TaskType.EMBEDDING,
    }
    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'score',
    }

    with pytest.raises(BackendConfigurationError, match='task_mode mismatch'):
        backend.validate_runtime_spec(runtime_spec, runtime_context)


def test_serve_builder_rejects_unsupported_vllm_task(model_store: LocalModelStore) -> None:
    registry = ModelRegistry(
        ModelCatalogFile(
            models=[
                ModelConfig(
                    name='unsupported-vlm',
                    alias='unsupported-vlm',
                    task=TaskType.VLM,
                    backend='vllm',
                    model_path='Qwen/Qwen3-VL-7B-Instruct',
                    tensor_parallel_size=1,
                    cpu_per_replica=4,
                    gpu_per_replica=1,
                    min_replicas=1,
                    max_replicas=1,
                )
            ]
        )
    )
    builder = ServeApplicationBuilder(model_store=model_store)

    with pytest.raises(BackendConfigurationError, match="does not support model task 'vlm'"):
        builder.validate_registry_runtime_configs(registry)
