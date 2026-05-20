"""运行时构建与 vLLM 后端行为测试。"""

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
    """返回基于测试配置加载的模型注册表。"""
    return ModelRegistry(load_model_catalog('config/models.yaml'))


@pytest.fixture
def model_store() -> LocalModelStore:
    """返回测试用本地模型仓库实例。"""
    return LocalModelStore('models')


class FakeBoundDeployment:
    """模拟 Serve deployment.bind 产物。"""

    def __init__(self, kwargs: dict, replica_cls: type[ModelRuntimeReplica]) -> None:
        """记录 deployment 参数和副本类。"""
        self.kwargs = kwargs
        self.replica_cls = replica_cls

    def bind(self, *args, **kwargs) -> dict:
        """返回可断言的绑定信息。"""
        return {
            'deployment_kwargs': self.kwargs,
            'replica_cls': self.replica_cls,
            'args': args,
            'kwargs': kwargs,
        }


class FakeServe:
    """模拟 Serve 对象。"""

    def deployment(self, **kwargs):
        """模拟 `serve.deployment` 装饰器。"""
        def wrapper(replica_cls: type[ModelRuntimeReplica]) -> FakeBoundDeployment:
            return FakeBoundDeployment(kwargs, replica_cls)

        return wrapper


def test_serve_builder_plan_contains_all_registered_models(
    registry: ModelRegistry,
    model_store: LocalModelStore,
) -> None:
    """构建计划应覆盖目录中的全部模型。"""
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
    """本地开发摘要应返回模型、部署与资源汇总。"""
    builder = ServeApplicationBuilder(model_store=model_store)

    summary = builder.build_local_dev_summary(registry)

    assert summary['applications'] == [
        'infer-nexus-model-qwen3-32b-instruct',
        'infer-nexus-model-bge-large-zh-v1_5',
        'infer-nexus-model-bge-reranker-v2-m3',
    ]
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
    """运行时上下文应包含已解析模型路径和后端规格。"""
    builder = ServeApplicationBuilder(model_store=model_store)

    runtime_context = builder.build_runtime_context(registry, 'qwen3-chat')

    assert runtime_context['app_name'] == 'infer-nexus-model-qwen3-32b-instruct'
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
    """应基于 fake serve 正确生成各模型 binding。"""
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


def test_deployment_factory_applies_llmconfig_style_overrides() -> None:
    """Deployment config should be able to override autoscaling and actor options."""
    spec = ServeApplicationBuilder(model_store=LocalModelStore('models')).deployment_factory.build_spec(
        ModelConfig(
            name='mineru',
            alias='mineru',
            task=TaskType.CHAT,
            model_path='opendatalab/MinerU2.5-2509-1.2B',
            tensor_parallel_size=1,
            cpu_per_replica=4,
            gpu_per_replica=1,
            min_replicas=1,
            max_replicas=2,
            deployment_config={
                'autoscaling_config': {
                    'min_replicas': 7,
                    'max_replicas': 8,
                    'target_ongoing_requests': 20,
                },
                'ray_actor_options': {'num_cpus': 6},
            },
        )
    )

    assert spec.autoscaling_config == {
        'min_replicas': 7,
        'max_replicas': 8,
        'target_ongoing_requests': 20,
    }
    assert spec.ray_actor_options['num_cpus'] == 6
    assert spec.ray_actor_options['num_gpus'] == 1


def test_build_application_name_is_stable(
    registry: ModelRegistry,
    model_store: LocalModelStore,
) -> None:
    """Serve 应用根节点应包含全部模型 binding。"""
    builder = ServeApplicationBuilder(model_store=model_store)

    assert builder.build_application_name('qwen3-32b-instruct') == 'infer-nexus-model-qwen3-32b-instruct'


def test_require_ray_serve_raises_without_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺少 ray 依赖时 require_ray_serve 应报错。"""
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
    """存在注入模块时 require_ray_serve 应直接返回。"""
    builder = ServeApplicationBuilder(model_store=LocalModelStore('models'))
    fake_serve = object()
    fake_ray = types.SimpleNamespace(serve=fake_serve)
    monkeypatch.setitem(sys.modules, 'ray', fake_ray)

    assert builder.require_ray_serve() is fake_serve


def test_model_runtime_replica_calls_backend_for_chat(
    registry: ModelRegistry,
    model_store: LocalModelStore,
) -> None:
    """模型副本应调用后端完成 chat 请求。"""
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
    """模型副本应调用后端完成 embedding 请求。"""
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
    """embedding 请求指定 base64 时应返回字符串向量。"""
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
    """vLLM 后端应正确生成采样参数和消息格式。"""
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


def test_vllm_backend_accepts_multimodal_messages_for_vision_models() -> None:
    """Vision-capable models should pass multimodal chat blocks to vLLM."""
    backend = VLLMBackend({'capabilities': ['vision']})
    multimodal = ChatCompletionsRequest(
        model='qwen3-vl-chat-8b-instruct',
        messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'hello'},
                    {
                        'type': 'image_url',
                        'image_url': {'url': 'https://example.com/demo.png'},
                    },
                ],
            }
        ],
    )

    assert backend._build_chat_messages(multimodal) == [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'hello'},
                {
                    'type': 'image_url',
                    'image_url': {'url': 'https://example.com/demo.png'},
                },
            ],
        }
    ]


def test_vllm_backend_build_runtime_spec_preserves_engine_kwargs(tmp_path) -> None:
    """Model engine kwargs should be preserved in runtime spec for startup."""
    backend = VLLMBackend({})
    model = ModelConfig(
        name='mineru',
        alias='mineru',
        task=TaskType.CHAT,
        backend='vllm',
        model_path='opendatalab/MinerU2.5-2509-1.2B',
        dtype='auto',
        tensor_parallel_size=1,
        max_model_len=16384,
        cpu_per_replica=4,
        gpu_per_replica=1,
        min_replicas=1,
        max_replicas=1,
        capabilities=['vision'],
        engine_kwargs={
            'trust_remote_code': True,
            'limit_mm_per_prompt': {'image': 10},
        },
    )

    runtime_spec = backend.build_runtime_spec(model, str(tmp_path / 'mineru'))

    assert runtime_spec['engine_kwargs'] == {
        'trust_remote_code': True,
        'limit_mm_per_prompt': {'image': 10},
    }


def test_vllm_backend_build_runtime_spec_uses_served_model_name_and_loading_config() -> None:
    """Runtime spec should preserve official-style loading metadata."""
    backend = VLLMBackend({})
    model = ModelConfig(
        name='mineru',
        alias='mineru',
        served_model_name='mineru-chat',
        task=TaskType.CHAT,
        model_loading_config={'model_id': 'opendatalab/MinerU2.5-2509-1.2B'},
        tensor_parallel_size=1,
        cpu_per_replica=4,
        gpu_per_replica=1,
        min_replicas=1,
        max_replicas=1,
        require_local_artifacts=False,
    )

    runtime_spec = backend.build_runtime_spec(model, 'opendatalab/MinerU2.5-2509-1.2B')

    assert runtime_spec['model_path'] == 'opendatalab/MinerU2.5-2509-1.2B'
    assert runtime_spec['served_model_name'] == 'mineru-chat'
    assert runtime_spec['model_loading_config']['model_id'] == 'opendatalab/MinerU2.5-2509-1.2B'


def test_vllm_backend_startup_forwards_engine_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catalog-provided engine kwargs should be forwarded to vLLM LLM."""
    captured_kwargs: dict[str, object] = {}

    class FakeLLM:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)
            self.supported_tasks = ['generate']

    fake_vllm = types.SimpleNamespace(LLM=FakeLLM)
    monkeypatch.setitem(sys.modules, 'vllm', fake_vllm)

    backend = VLLMBackend(
        {
            'model_path': 'opendatalab/MinerU2.5-2509-1.2B',
            'tensor_parallel_size': 1,
            'dtype': 'auto',
            'max_model_len': 16384,
            'gpu_memory_utilization': 0.85,
            'task_mode': 'generate',
            'engine_kwargs': {
                'trust_remote_code': True,
                'limit_mm_per_prompt': {'image': 10},
            },
            'backend_init_mode': 'real',
        }
    )

    backend.startup()

    assert captured_kwargs['model'] == 'opendatalab/MinerU2.5-2509-1.2B'
    assert captured_kwargs['trust_remote_code'] is True
    assert captured_kwargs['limit_mm_per_prompt'] == {'image': 10}


def test_vllm_backend_sampling_params_include_official_compatible_fields() -> None:
    """Sampling params should preserve common OpenAI/vLLM request fields."""
    backend = VLLMBackend({})
    request = ChatCompletionsRequest(
        model='mineru',
        messages=[{'role': 'user', 'content': 'hello'}],
        temperature=0.2,
        top_p=0.9,
        max_tokens=256,
        presence_penalty=0.3,
        frequency_penalty=0.4,
        repetition_penalty=1.05,
        stop=['DONE'],
        n=2,
        seed=7,
        extra_body={'top_k': 20, 'min_p': 0.1},
    )

    sampling = backend._build_sampling_params(request)

    assert sampling == {
        'temperature': 0.2,
        'top_p': 0.9,
        'max_tokens': 256,
        'presence_penalty': 0.3,
        'frequency_penalty': 0.4,
        'repetition_penalty': 1.05,
        'stop': ['DONE'],
        'n': 2,
        'seed': 7,
        'top_k': 20,
        'min_p': 0.1,
    }


def test_vllm_backend_sampling_params_accept_mineru_http_client_fields_at_top_level() -> None:
    """MinerU http-client sends vLLM-compatible extras as top-level request fields."""
    backend = VLLMBackend({})
    request = ChatCompletionsRequest(
        model='mineru',
        messages=[{'role': 'user', 'content': 'hello'}],
        max_tokens=128,
        **{
            'top_k': 20,
            'skip_special_tokens': False,
            'vllm_xargs': {
                'no_repeat_ngram_size': 16,
                'debug': True,
            },
        },
    )

    sampling = backend._build_sampling_params(request)

    assert sampling == {
        'temperature': 0.7,
        'top_p': 1.0,
        'max_tokens': 128,
        'top_k': 20,
        'skip_special_tokens': False,
        'no_repeat_ngram_size': 16,
    }


def test_vllm_backend_filters_sampling_params_by_runtime_signature() -> None:
    """Unsupported SamplingParams kwargs should be dropped for older vLLM versions."""
    backend = VLLMBackend({})

    class FakeSamplingParams:
        def __init__(self, temperature=None, top_p=None, max_tokens=None, top_k=None):
            pass

    filtered = backend._filter_sampling_params_for_vllm(
        {
            'temperature': 0.7,
            'top_p': 1.0,
            'max_tokens': 128,
            'top_k': 20,
            'skip_special_tokens': False,
            'no_repeat_ngram_size': 16,
        },
        sampling_params_cls=FakeSamplingParams,
    )

    assert filtered == {
        'temperature': 0.7,
        'top_p': 1.0,
        'max_tokens': 128,
        'top_k': 20,
    }


def test_vllm_backend_builds_sampling_params_instance_by_retrying_unsupported_kwargs() -> None:
    """SamplingParams construction should retry after stripping unsupported kwargs from runtime errors."""
    backend = VLLMBackend({})

    class FakeSamplingParams:
        def __init__(self, temperature=None, top_p=None, max_tokens=None, no_repeat_ngram_size=None):
            if no_repeat_ngram_size is not None:
                raise TypeError("Unexpected keyword argument 'no_repeat_ngram_size'")
            self.temperature = temperature
            self.top_p = top_p
            self.max_tokens = max_tokens

    instance = backend._build_sampling_params_instance(
        {
            'temperature': 0.7,
            'top_p': 1.0,
            'max_tokens': 128,
            'no_repeat_ngram_size': 16,
        },
        sampling_params_cls=FakeSamplingParams,
    )

    assert instance.temperature == 0.7
    assert instance.top_p == 1.0
    assert instance.max_tokens == 128


def test_vllm_backend_rejects_streaming_and_multimodal_messages_for_text_only_models() -> None:
    """Streaming stays unsupported and text-only models still reject image blocks."""
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

    with pytest.raises(BackendRequestValidationError, match='does not support multimodal'):
        backend._build_chat_messages(multimodal)


def test_vllm_backend_normalizes_embedding_inputs_and_rejects_empty() -> None:
    """embedding 输入归一化应接受字符串并拒绝空输入。"""
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
    """模型副本应调用后端完成 rerank 请求。"""
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
    """rerank 文档归一化应接受字符串并拒绝空文档列表。"""
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
    """runtime_spec 与任务类型不匹配时应抛配置错误。"""
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
    """不支持的任务类型应在构建运行时上下文时被拒绝。"""
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
