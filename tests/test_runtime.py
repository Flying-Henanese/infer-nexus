"""运行时构建与 vLLM 后端行为测试。"""

import sys
import types
import asyncio

import pytest

from infer_nexus.catalog.loader import load_model_catalog
from infer_nexus.catalog.models import ModelCatalogFile, ModelConfig
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.backends.vllm import DynamicVLLMOpenAIChatServingAdapter
from infer_nexus.backends.vllm import DynamicVLLMOpenAIEmbeddingServingAdapter
from infer_nexus.backends.vllm import OpenAIServingEngineClientCompatProxy
from infer_nexus.core.errors import BackendConfigurationError, BackendRequestValidationError
from infer_nexus.core.enums import CompatibilityMode, TaskType
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest
from infer_nexus.backends.vllm import VLLMBackend
from infer_nexus.model_store import LocalModelStore
from infer_nexus.runtime.deployments import DeploymentFactory, ModelRuntimeReplica
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


class FakeOpenAIChatServingAdapter:
    """模拟 vLLM OpenAI serving adapter。"""

    def __init__(
        self,
        *,
        response: dict | None = None,
        stream_chunks: list[dict | bytes | str] | None = None,
    ) -> None:
        self.response = response or {}
        self.stream_chunks = stream_chunks or []
        self.requests: list[dict] = []

    async def chat_completion(self, request_payload: dict[str, object]) -> dict:
        self.requests.append(request_payload)
        return self.response

    async def chat_completion_stream(self, request_payload: dict[str, object]):
        self.requests.append(request_payload)
        for chunk in self.stream_chunks:
            yield chunk


class FakeOpenAIEmbeddingServingAdapter:
    """模拟 vLLM OpenAI embeddings serving adapter。"""

    def __init__(self, *, response: dict | None = None) -> None:
        self.response = response or {}
        self.requests: list[dict] = []

    async def embedding(self, request_payload: dict[str, object]) -> dict:
        self.requests.append(request_payload)
        return self.response


class FakeServingRequest:
    """模拟 vLLM 的 ChatCompletionRequest。"""

    def __init__(self, **payload) -> None:
        self.payload = payload
        self.stream = payload.get('stream')

    @classmethod
    def model_validate(cls, payload: dict[str, object]) -> 'FakeServingRequest':
        return cls(**payload)


class FakeServingResponse:
    """模拟 vLLM serving response Pydantic model。"""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def model_dump(self, *, mode: str = 'json', exclude_none: bool = True) -> dict[str, object]:
        return dict(self.payload)


class FakeJSONResponse:
    """模拟 FastAPI JSONResponse 风格响应。"""

    def __init__(self, payload: dict[str, object]) -> None:
        import json

        self.body = json.dumps(payload).encode('utf-8')


class FakeBaseModelPath:
    """模拟 vLLM BaseModelPath。"""

    def __init__(self, name: str, model_path: str) -> None:
        self.name = name
        self.model_path = model_path


class FakeOpenAIServingModelsNative:
    """模拟新布局的 OpenAIServingModels。"""

    def __init__(self, engine_client: object, base_model_paths: list[FakeBaseModelPath], *, lora_modules=None):
        self.engine_client = engine_client
        self.base_model_paths = base_model_paths
        self.registry = {'base_model_paths': base_model_paths}


class FakeOpenAIServingRender:
    """模拟新布局的 OpenAIServingRender。"""

    def __init__(
        self,
        model_config: object,
        renderer: object,
        io_processor: object,
        model_registry: object,
        *,
        request_logger: object,
        chat_template: object,
        chat_template_content_format: str,
        trust_request_chat_template: bool = False,
        default_chat_template_kwargs: dict | None = None,
        **_: object,
    ) -> None:
        self.model_config = model_config
        self.renderer = renderer
        self.io_processor = io_processor
        self.model_registry = model_registry
        self.request_logger = request_logger
        self.chat_template = chat_template
        self.chat_template_content_format = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template
        self.default_chat_template_kwargs = default_chat_template_kwargs


class FakeOpenAIServingChatNative:
    """模拟新布局的 OpenAIServingChat。"""

    def __init__(
        self,
        engine_client: object,
        models: FakeOpenAIServingModelsNative,
        response_role: str,
        *,
        openai_serving_render: FakeOpenAIServingRender,
        request_logger: object,
        chat_template: object,
        chat_template_content_format: str,
        enable_auto_tools: bool | None = None,
        tool_parser: str | None = None,
        reasoning_parser: str = '',
        default_chat_template_kwargs: dict | None = None,
        **_: object,
    ) -> None:
        self.engine_client = engine_client
        self.models = models
        self.response_role = response_role
        self.openai_serving_render = openai_serving_render
        self.request_logger = request_logger
        self.chat_template = chat_template
        self.chat_template_content_format = chat_template_content_format
        self.enable_auto_tools = enable_auto_tools
        self.tool_parser = tool_parser
        self.reasoning_parser = reasoning_parser
        self.default_chat_template_kwargs = default_chat_template_kwargs
        self.requests: list[FakeServingRequest] = []

    async def create_chat_completion(self, request: FakeServingRequest, raw_request=None):
        self.requests.append(request)
        return FakeServingResponse(
            {
                'id': 'chatcmpl-native',
                'object': 'chat.completion',
                'model': request.payload['model'],
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'native'}}],
            }
        )


class FakeServingEmbeddingNative:
    """模拟 vLLM native embedding serving。"""

    def __init__(
        self,
        engine_client: object,
        models: FakeOpenAIServingModelsNative,
        *,
        supported_tasks: tuple[str, ...] | None = None,
        request_logger: object = None,
        chat_template: object = None,
        chat_template_content_format: str = 'auto',
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        log_error_stack: bool = False,
        **_: object,
    ) -> None:
        self.engine_client = engine_client
        self.models = models
        self.kwargs = {
            'supported_tasks': supported_tasks,
            'request_logger': request_logger,
            'chat_template': chat_template,
            'chat_template_content_format': chat_template_content_format,
            'trust_request_chat_template': trust_request_chat_template,
            'return_tokens_as_token_ids': return_tokens_as_token_ids,
            'log_error_stack': log_error_stack,
        }
        self.requests: list[FakeServingRequest] = []

    async def create_embedding(self, request: FakeServingRequest, raw_request=None):
        self.requests.append(request)
        return FakeJSONResponse(
            {
                'object': 'list',
                'model': request.payload['model'],
                'data': [{'object': 'embedding', 'index': 0, 'embedding': [0.1, 0.2]}],
                'usage': {'prompt_tokens': 2, 'total_tokens': 2},
            }
        )


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


def test_deployment_factory_maps_npu_resources_to_custom_ray_resource() -> None:
    """NPU runtime should use Ray custom resources instead of CUDA num_gpus."""
    factory = DeploymentFactory(inference_device_type="npu")

    spec = factory.build_spec(
        ModelConfig(
            name='qwen3-embedding-8b',
            alias='qwen3-embedding-8b',
            task=TaskType.EMBEDDING,
            model_path='Qwen/Qwen3-Embedding-8B',
            tensor_parallel_size=1,
            cpu_per_replica=2,
            gpu_per_replica=0.3,
            min_replicas=1,
            max_replicas=1,
        )
    )

    assert spec.ray_actor_options['num_cpus'] == 2
    assert spec.ray_actor_options['resources'] == {'NPU': 0.3}
    assert 'num_gpus' not in spec.ray_actor_options


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


def test_model_config_merges_legacy_engine_kwargs_into_backend_scoped_vllm() -> None:
    """Legacy top-level engine kwargs should remain compatible with the new vllm block."""
    model = ModelConfig(
        name='qwen3-tools',
        alias='qwen3-tools',
        task=TaskType.CHAT,
        model_path='Qwen/Qwen3-8B',
        tensor_parallel_size=1,
        cpu_per_replica=4,
        gpu_per_replica=1,
        min_replicas=1,
        max_replicas=1,
        engine_kwargs={'trust_remote_code': True},
        vllm={
            'engine_kwargs': {'enable_auto_tool_choice': True},
            'request_defaults': {'temperature': 0.2, 'tool_choice': 'auto'},
            'request_policy': {'allow_tools': True},
        },
    )

    assert model.engine_kwargs == {
        'trust_remote_code': True,
        'enable_auto_tool_choice': True,
    }
    assert model.vllm.engine_kwargs == model.engine_kwargs
    assert model.vllm.request_defaults == {'temperature': 0.2, 'tool_choice': 'auto'}
    assert model.vllm.request_policy.allow_tools is True


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


def test_vllm_backend_build_runtime_spec_includes_request_defaults_and_policy() -> None:
    """Runtime spec should carry request defaults and passthrough policy for local chat handling."""
    backend = VLLMBackend({})
    model = ModelConfig(
        name='qwen3-tools',
        alias='qwen3-tools',
        task=TaskType.CHAT,
        model_path='Qwen/Qwen3-8B',
        tensor_parallel_size=1,
        cpu_per_replica=4,
        gpu_per_replica=1,
        min_replicas=1,
        max_replicas=1,
        vllm={
            'engine_kwargs': {'enable_auto_tool_choice': True},
            'request_defaults': {'temperature': 0.2, 'tool_choice': 'auto'},
            'request_policy': {
                'allow_tools': True,
                'allow_reasoning': True,
                'passthrough_unknown_openai_fields': False,
            },
        },
    )

    runtime_spec = backend.build_runtime_spec(model, 'Qwen/Qwen3-8B')

    assert runtime_spec['engine_kwargs'] == {'enable_auto_tool_choice': True}
    assert runtime_spec['request_defaults'] == {'temperature': 0.2, 'tool_choice': 'auto'}
    assert runtime_spec['request_policy'] == {
        'allow_tools': True,
        'allow_reasoning': True,
        'passthrough_unknown_openai_fields': False,
    }


def test_vllm_backend_build_runtime_spec_includes_openai_serving_reasoning_config() -> None:
    """OpenAI serving config should be explicit and mirrored into engine kwargs."""
    backend = VLLMBackend({})
    model = ModelConfig(
        name='qwen3-reasoning',
        alias='qwen3-reasoning',
        task=TaskType.CHAT,
        model_path='Qwen/Qwen3-32B',
        tensor_parallel_size=1,
        cpu_per_replica=4,
        gpu_per_replica=1,
        min_replicas=1,
        max_replicas=1,
        vllm={
            'openai_serving': {
                'enabled': True,
                'enable_reasoning': True,
                'reasoning_parser': 'qwen3',
            },
            'engine_kwargs': {
                'enable_auto_tool_choice': True,
                'tool_call_parser': 'qwen3_xml',
            },
            'request_policy': {'allow_tools': True, 'allow_reasoning': True},
        },
    )

    runtime_spec = backend.build_runtime_spec(model, 'Qwen/Qwen3-32B')

    assert runtime_spec['openai_serving'] == {
        'enabled': True,
        'enable_reasoning': True,
        'reasoning_parser': 'qwen3',
    }
    assert runtime_spec['engine_kwargs']['enable_reasoning'] is True
    assert runtime_spec['engine_kwargs']['reasoning_parser'] == 'qwen3'
    assert runtime_spec['engine_kwargs']['enable_auto_tool_choice'] is True
    assert runtime_spec['engine_kwargs']['tool_call_parser'] == 'qwen3_xml'
    assert runtime_spec['request_policy']['allow_tools'] is True


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


def test_vllm_backend_sampling_params_apply_request_defaults() -> None:
    """Per-model request defaults should seed chat sampling params without overriding explicit request fields."""
    backend = VLLMBackend(
        {
            'request_defaults': {
                'temperature': 0.2,
                'top_p': 0.8,
                'max_tokens': 256,
                'top_k': 20,
            }
        }
    )
    request = ChatCompletionsRequest(
        model='qwen3-tools',
        messages=[{'role': 'user', 'content': 'hello'}],
        top_p=0.95,
    )

    sampling = backend._build_sampling_params(request)

    assert sampling == {
        'temperature': 0.2,
        'top_p': 0.95,
        'max_tokens': 256,
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


def test_vllm_backend_accepts_streaming_messages_and_rejects_text_only_multimodal() -> None:
    """Streaming uses the same message normalization while text-only models still reject image blocks."""
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

    assert backend._build_chat_messages(streaming) == [{'role': 'user', 'content': 'hello'}]

    assert backend._build_chat_messages(multimodal) == [{'role': 'user', 'content': 'hello'}]


def test_vllm_backend_rejects_image_blocks_for_non_vision_models() -> None:
    """Non-vision models should still reject image content blocks."""
    backend = VLLMBackend({})
    multimodal = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'hello'},
                    {'type': 'image_url', 'image_url': {'url': 'https://example.com/demo.png'}},
                ],
            }
        ],
    )

    with pytest.raises(BackendRequestValidationError, match='does not support multimodal'):
        backend._build_chat_messages(multimodal)


def test_vllm_backend_rejects_tool_calling_when_policy_is_disabled() -> None:
    """Local vLLM backend should reject tool-call request fields unless explicitly enabled."""
    backend = VLLMBackend({'request_policy': {'allow_tools': False}})
    request = ChatCompletionsRequest(
        model='qwen3-tools',
        messages=[{'role': 'user', 'content': 'hello'}],
        tools=[{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}],
    )

    with pytest.raises(BackendRequestValidationError, match='tool-calling'):
        backend._build_chat_kwargs(request)


def test_vllm_backend_builds_chat_kwargs_for_tools_and_reasoning() -> None:
    """Enabled request policy should pass through tool-calling and reasoning request fields."""
    backend = VLLMBackend(
        {
            'request_defaults': {
                'tool_choice': 'auto',
                'parallel_tool_calls': True,
                'reasoning': {'enabled': True},
            },
            'request_policy': {
                'allow_tools': True,
                'allow_reasoning': True,
            },
        }
    )
    request = ChatCompletionsRequest(
        model='qwen3-tools',
        messages=[{'role': 'user', 'content': 'hello'}],
        tools=[{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}],
    )

    chat_kwargs = backend._build_chat_kwargs(request)

    assert chat_kwargs == {
        'tools': [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}],
        'tool_choice': 'auto',
        'parallel_tool_calls': True,
        'reasoning': {'enabled': True},
    }


def test_vllm_backend_chat_invocation_forwards_tool_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local engine.chat should receive tool-calling kwargs when the runtime policy enables them."""
    captured: dict[str, object] = {}

    class FakeSamplingParams:
        def __init__(self, temperature=None, top_p=None, max_tokens=None):
            self.temperature = temperature
            self.top_p = top_p
            self.max_tokens = max_tokens

    class FakeLLM:
        supported_tasks = ['generate']

        def chat(self, messages, sampling_params=None, tools=None, tool_choice=None, parallel_tool_calls=None):
            captured['messages'] = messages
            captured['sampling_params'] = sampling_params
            captured['tools'] = tools
            captured['tool_choice'] = tool_choice
            captured['parallel_tool_calls'] = parallel_tool_calls
            return [
                types.SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[
                        types.SimpleNamespace(
                            text='ok',
                            finish_reason='stop',
                            token_ids=[3],
                        )
                    ],
                )
            ]

    fake_vllm = types.SimpleNamespace(SamplingParams=FakeSamplingParams)
    monkeypatch.setitem(sys.modules, 'vllm', fake_vllm)

    backend = VLLMBackend(
        {
            'request_defaults': {'tool_choice': 'auto', 'parallel_tool_calls': True},
            'request_policy': {'allow_tools': True},
        }
    )
    backend.engine = FakeLLM()
    backend.engine_state = 'ready'
    request = ChatCompletionsRequest(
        model='qwen3-tools',
        messages=[{'role': 'user', 'content': 'hello'}],
        tools=[{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}],
    )

    response = asyncio.run(
        backend.chat_completion(
            {
                'backend': 'vllm',
                'request_defaults': {'tool_choice': 'auto', 'parallel_tool_calls': True},
                'request_policy': {'allow_tools': True},
            },
            request,
            {'deployment_name': 'model-qwen3-tools'},
        )
    )

    assert captured['messages'] == [{'role': 'user', 'content': 'hello'}]
    assert captured['tool_choice'] == 'auto'
    assert captured['parallel_tool_calls'] is True
    assert captured['tools'] == [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}]
    assert response['content'] == 'ok'


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


def test_vllm_backend_rejects_vllm_native_rerank_runtime_spec() -> None:
    """vllm_native 初始阶段不支持 rerank。"""
    backend = VLLMBackend({})
    runtime_context = {
        'model_name': 'bge-rerank',
        'task': TaskType.RERANK,
    }
    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'score',
        'compat_mode': CompatibilityMode.VLLM_NATIVE.value,
    }

    with pytest.raises(BackendConfigurationError, match='rerank'):
        backend.validate_runtime_spec(runtime_spec, runtime_context)


def test_vllm_backend_chat_completion_passthroughs_openai_serving_payload() -> None:
    """OpenAI serving adapter should receive a native OpenAI-style request payload."""
    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'generate',
        'request_defaults': {},
        'request_policy': {
            'allow_tools': True,
            'allow_reasoning': True,
            'passthrough_unknown_openai_fields': True,
        },
        'openai_serving': {'enabled': True},
        'served_model_name': 'qwen3-chat',
        'capabilities': [],
    }
    runtime_context = {
        'deployment_name': 'model-qwen3-32b-instruct',
        'served_model_name': 'qwen3-chat',
    }
    expected_response = {
        'id': 'chatcmpl-serving',
        'object': 'chat.completion',
        'created': 123,
        'model': 'qwen3-chat',
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': 'hello from serving'},
                'finish_reason': 'stop',
            }
        ],
        'usage': {'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7},
    }
    adapter = FakeOpenAIChatServingAdapter(response=expected_response)
    backend = VLLMBackend(runtime_spec)
    backend.openai_serving_chat_adapter = adapter

    response = asyncio.run(
        backend.chat_completion(
            runtime_spec,
            ChatCompletionsRequest(
                model='qwen3-chat',
                messages=[
                    {'role': 'system', 'content': 'be concise'},
                    {'role': 'user', 'content': 'hello'},
                ],
                max_completion_tokens=32,
                parallel_tool_calls=False,
                extra_body={'top_k': 50, 'min_p': 0.1},
            ),
            runtime_context,
        )
    )

    assert response == expected_response
    assert adapter.requests == [
        {
            'model': 'qwen3-chat',
            'messages': [
                {'role': 'system', 'content': 'be concise'},
                {'role': 'user', 'content': 'hello'},
            ],
            'max_completion_tokens': 32,
            'parallel_tool_calls': False,
            'stream': False,
            'top_k': 50,
            'min_p': 0.1,
        }
    ]


def test_vllm_backend_chat_completion_stream_passthroughs_openai_serving_chunks() -> None:
    """Streaming chat should pass through native OpenAI serving chunks when available."""
    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'generate',
        'request_defaults': {},
        'request_policy': {
            'allow_tools': True,
            'allow_reasoning': True,
            'passthrough_unknown_openai_fields': True,
        },
        'openai_serving': {'enabled': True},
        'served_model_name': 'qwen3-chat',
        'capabilities': [],
    }
    runtime_context = {
        'deployment_name': 'model-qwen3-32b-instruct',
        'served_model_name': 'qwen3-chat',
    }
    adapter = FakeOpenAIChatServingAdapter(stream_chunks=[b'data: [DONE]\n\n'])
    backend = VLLMBackend(runtime_spec)
    backend.openai_serving_chat_adapter = adapter

    async def collect() -> list[dict | bytes | str]:
        return [
            chunk
            async for chunk in backend.chat_completion_stream(
                runtime_spec,
                ChatCompletionsRequest(
                    model='qwen3-chat',
                    messages=[{'role': 'user', 'content': 'hello'}],
                    stream=True,
                    stream_options={'include_usage': True},
                ),
                runtime_context,
            )
        ]

    chunks = asyncio.run(collect())

    assert chunks == [b'data: [DONE]\n\n']
    assert adapter.requests == [
        {
            'model': 'qwen3-chat',
            'messages': [{'role': 'user', 'content': 'hello'}],
            'stream': True,
            'stream_options': {'include_usage': True},
        }
    ]


def test_vllm_native_chat_missing_adapter_fails_without_local_fallback() -> None:
    """vllm_native chat 缺少 native adapter 时不应退回本地协议重建。"""
    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'generate',
        'compat_mode': CompatibilityMode.VLLM_NATIVE.value,
        'request_defaults': {},
        'request_policy': {},
        'openai_serving': {'enabled': True},
        'served_model_name': 'qwen3-chat',
        'capabilities': [],
    }
    runtime_context = {
        'deployment_name': 'model-qwen3',
        'served_model_name': 'qwen3-chat',
    }
    backend = VLLMBackend(runtime_spec)
    backend.engine = None

    with pytest.raises(BackendConfigurationError, match='vllm_native'):
        asyncio.run(
            backend.chat_completion(
                runtime_spec,
                ChatCompletionsRequest(
                    model='qwen3-chat',
                    messages=[{'role': 'user', 'content': 'hello'}],
                ),
                runtime_context,
            )
        )


def test_vllm_native_chat_invocation_failure_does_not_fallback() -> None:
    """vllm_native chat 调用 native adapter 失败时应直接暴露错误。"""
    class FailingAdapter:
        async def chat_completion(self, request_payload):
            raise RuntimeError('native failed')

    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'generate',
        'compat_mode': CompatibilityMode.VLLM_NATIVE.value,
        'request_defaults': {},
        'request_policy': {},
        'openai_serving': {'enabled': True},
        'served_model_name': 'qwen3-chat',
        'capabilities': [],
    }
    runtime_context = {
        'deployment_name': 'model-qwen3',
        'served_model_name': 'qwen3-chat',
    }
    backend = VLLMBackend(runtime_spec)
    backend.openai_serving_chat_adapter = FailingAdapter()

    with pytest.raises(RuntimeError, match='native failed'):
        asyncio.run(
            backend.chat_completion(
                runtime_spec,
                ChatCompletionsRequest(
                    model='qwen3-chat',
                    messages=[{'role': 'user', 'content': 'hello'}],
                ),
                runtime_context,
            )
        )


def test_vllm_native_embedding_passthroughs_openai_serving_payload() -> None:
    """vllm_native embedding 应保留请求字段并只重写 model。"""
    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'embed',
        'compat_mode': CompatibilityMode.VLLM_NATIVE.value,
        'served_model_name': 'qwen3-embedding',
    }
    runtime_context = {
        'deployment_name': 'model-qwen3-embedding',
        'served_model_name': 'qwen3-embedding',
    }
    expected_response = {
        'object': 'list',
        'model': 'qwen3-embedding',
        'data': [{'object': 'embedding', 'index': 0, 'embedding': [0.1, 0.2]}],
        'usage': {'prompt_tokens': 2, 'total_tokens': 2},
    }
    adapter = FakeOpenAIEmbeddingServingAdapter(response=expected_response)
    backend = VLLMBackend(runtime_spec)
    backend.openai_serving_embedding_adapter = adapter

    response = asyncio.run(
        backend.embedding(
            runtime_spec,
            EmbeddingRequest(
                model='client-facing-name',
                input=['hello', 'world'],
                encoding_format='float',
                dimensions=2,
                user='u1',
            ),
            runtime_context,
        )
    )

    assert response == expected_response
    assert adapter.requests == [
        {
            'model': 'qwen3-embedding',
            'input': ['hello', 'world'],
            'encoding_format': 'float',
            'dimensions': 2,
            'user': 'u1',
        }
    ]


def test_vllm_native_embedding_missing_adapter_fails_without_local_fallback() -> None:
    """vllm_native embedding 缺少 native adapter 时不应返回本地 stub。"""
    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'embed',
        'compat_mode': CompatibilityMode.VLLM_NATIVE.value,
        'served_model_name': 'qwen3-embedding',
    }
    runtime_context = {'deployment_name': 'model-qwen3-embedding'}
    backend = VLLMBackend(runtime_spec)

    with pytest.raises(BackendConfigurationError, match='embedding'):
        asyncio.run(
            backend.embedding(
                runtime_spec,
                EmbeddingRequest(model='qwen3-embedding', input='hello'),
                runtime_context,
            )
        )


def test_vllm_backend_initializes_native_embedding_serving_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding native serving adapter should be constructed from vLLM serving symbols."""
    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'embed',
        'compat_mode': CompatibilityMode.VLLM_NATIVE.value,
        'model_path': '/models/Qwen/Qwen3-Embedding-8B',
        'request_defaults': {},
        'openai_serving': {'enabled': True},
        'served_model_name': 'qwen3-embedding',
    }
    backend = VLLMBackend(runtime_spec)
    engine_client = types.SimpleNamespace(
        model_config='model-config',
        renderer='renderer',
        vllm_config='vllm-config',
    )
    backend.engine = types.SimpleNamespace(llm_engine=engine_client)

    module_map = {
        'vllm.entrypoints.pooling.embed.protocol': types.SimpleNamespace(
            EmbeddingRequest=FakeServingRequest
        ),
        'vllm.entrypoints.pooling.embed.serving': types.SimpleNamespace(
            ServingEmbedding=FakeServingEmbeddingNative
        ),
        'vllm.entrypoints.openai.chat_completion.protocol': types.SimpleNamespace(
            ChatCompletionRequest=FakeServingRequest
        ),
        'vllm.entrypoints.openai.chat_completion.serving': types.SimpleNamespace(
            OpenAIServingChat=FakeOpenAIServingChatNative
        ),
        'vllm.entrypoints.openai.models.serving': types.SimpleNamespace(
            OpenAIServingModels=FakeOpenAIServingModelsNative
        ),
        'vllm.entrypoints.openai.models.protocol': types.SimpleNamespace(
            BaseModelPath=FakeBaseModelPath
        ),
        'vllm.entrypoints.serve.render.serving': types.SimpleNamespace(
            OpenAIServingRender=FakeOpenAIServingRender
        ),
    }

    def fake_import_module(name: str):
        if name not in module_map:
            raise ImportError(name)
        return module_map[name]

    monkeypatch.setattr('infer_nexus.backends.vllm.importlib.import_module', fake_import_module)

    adapter = backend._initialize_openai_serving_embedding_adapter()

    assert isinstance(adapter, DynamicVLLMOpenAIEmbeddingServingAdapter)
    assert backend.openai_serving_embedding_adapter_init_error is None

    response = asyncio.run(
        adapter.embedding(
            {
                'model': 'qwen3-embedding',
                'input': ['hello'],
                'encoding_format': 'float',
            }
        )
    )

    assert response == {
        'object': 'list',
        'model': 'qwen3-embedding',
        'data': [{'object': 'embedding', 'index': 0, 'embedding': [0.1, 0.2]}],
        'usage': {'prompt_tokens': 2, 'total_tokens': 2},
    }
    assert adapter.serving_embedding.requests[0].payload == {
        'model': 'qwen3-embedding',
        'input': ['hello'],
        'encoding_format': 'float',
    }
    assert adapter.serving_embedding.kwargs['supported_tasks'] == ('embed',)
    assert adapter.serving_embedding.kwargs['request_logger'] is None


def test_vllm_backend_startup_initializes_embedding_adapter_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embed runtimes should initialize the native embedding adapter during startup."""
    captured_kwargs = {}

    class FakeLLM:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)
            self.supported_tasks = ['embed']
            self.llm_engine = types.SimpleNamespace(
                model_config='model-config',
                renderer='renderer',
                vllm_config='vllm-config',
            )

    fake_vllm = types.SimpleNamespace(LLM=FakeLLM)
    monkeypatch.setitem(sys.modules, 'vllm', fake_vllm)
    monkeypatch.setattr(
        VLLMBackend,
        '_initialize_openai_serving_embedding_adapter',
        lambda self: FakeOpenAIEmbeddingServingAdapter(response={'ok': True}),
    )
    monkeypatch.setattr(
        VLLMBackend,
        '_initialize_openai_serving_chat_adapter',
        lambda self: None,
    )

    backend = VLLMBackend(
        {
            'backend': 'vllm',
            'model_path': 'Qwen/Qwen3-Embedding-8B',
            'tensor_parallel_size': 1,
            'dtype': 'auto',
            'task_mode': 'embed',
            'engine_kwargs': {},
            'openai_serving': {'enabled': True},
            'compat_mode': CompatibilityMode.VLLM_NATIVE.value,
            'backend_init_mode': 'real',
        }
    )

    backend.startup()

    assert captured_kwargs['model'] == 'Qwen/Qwen3-Embedding-8B'
    assert captured_kwargs['task'] == 'embed'
    assert backend.engine_kind == 'sync'
    assert backend.engine_state == 'ready'
    assert isinstance(backend.openai_serving_embedding_adapter, FakeOpenAIEmbeddingServingAdapter)


def test_vllm_backend_chat_completion_stream_uses_native_vllm_deltas() -> None:
    """Native vLLM streaming output should be normalized into text delta events."""
    captured = {}

    class FakeOutput:
        def __init__(self, text: str, finish_reason: str | None = None) -> None:
            self.text = text
            self.finish_reason = finish_reason

    class FakeRequestOutput:
        def __init__(self, text: str, finish_reason: str | None = None) -> None:
            self.outputs = [FakeOutput(text, finish_reason)]

    class FakeStreamingLLM:
        def chat(self, messages, *, stream: bool, temperature, top_p, max_tokens, **kwargs):
            captured['messages'] = messages
            captured['stream'] = stream
            captured['temperature'] = temperature
            captured['top_p'] = top_p
            captured['max_tokens'] = max_tokens

            async def iterator():
                yield FakeRequestOutput('hel')
                yield FakeRequestOutput('hello')
                yield FakeRequestOutput('hello', 'stop')

            return iterator()

    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'generate',
        'request_defaults': {},
        'request_policy': {},
        'served_model_name': 'qwen3-chat',
        'capabilities': [],
    }
    runtime_context = {
        'deployment_name': 'model-qwen3-32b-instruct',
        'served_model_name': 'qwen3-chat',
    }
    backend = VLLMBackend(runtime_spec)
    backend.engine = FakeStreamingLLM()
    backend.engine_state = 'ready'

    async def collect() -> list[dict | bytes | str]:
        return [
            chunk
            async for chunk in backend.chat_completion_stream(
                runtime_spec,
                ChatCompletionsRequest(
                    model='qwen3-chat',
                    messages=[{'role': 'user', 'content': 'hello'}],
                    stream=True,
                    max_tokens=16,
                    temperature=0.1,
                ),
                runtime_context,
            )
        ]

    chunks = asyncio.run(collect())

    assert captured == {
        'messages': [{'role': 'user', 'content': 'hello'}],
        'stream': True,
        'temperature': 0.1,
        'top_p': 1.0,
        'max_tokens': 16,
    }
    assert [chunk['delta_text'] for chunk in chunks] == ['hel', 'lo', '']
    assert [chunk['finish_reason'] for chunk in chunks] == [None, None, 'stop']
    assert {chunk['type'] for chunk in chunks} == {'chat_delta'}


def test_vllm_backend_chat_completion_stream_uses_async_engine_generate_deltas() -> None:
    """AsyncLLMEngine outputs should be normalized into incremental chat delta events."""
    captured = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            captured['template_messages'] = messages
            captured['template_kwargs'] = kwargs
            captured['tokenize'] = tokenize
            captured['add_generation_prompt'] = add_generation_prompt
            return '<chat prompt>'

    class FakeOutput:
        def __init__(self, text: str, finish_reason: str | None = None) -> None:
            self.text = text
            self.finish_reason = finish_reason

    class FakeRequestOutput:
        def __init__(self, text: str, finish_reason: str | None = None) -> None:
            self.outputs = [FakeOutput(text, finish_reason)]

    class FakeAsyncEngine:
        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompt, sampling_params, request_id):
            captured['prompt'] = prompt
            captured['sampling_params'] = sampling_params
            captured['request_id'] = request_id

            async def iterator():
                yield FakeRequestOutput('hel')
                yield FakeRequestOutput('hello')
                yield FakeRequestOutput('hello', 'stop')

            return iterator()

    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'generate',
        'request_defaults': {'chat_template_kwargs': {'enable_thinking': False}},
        'request_policy': {'allow_reasoning': True},
        'served_model_name': 'qwen3-chat',
        'capabilities': [],
    }
    runtime_context = {
        'deployment_name': 'model-qwen3-32b-instruct',
        'served_model_name': 'qwen3-chat',
    }
    backend = VLLMBackend(runtime_spec)
    backend.engine = FakeAsyncEngine()
    backend.engine_kind = 'async'
    backend.engine_state = 'ready'

    async def collect() -> list[dict | bytes | str]:
        return [
            chunk
            async for chunk in backend.chat_completion_stream(
                runtime_spec,
                ChatCompletionsRequest(
                    model='qwen3-chat',
                    messages=[{'role': 'user', 'content': 'hello'}],
                    stream=True,
                    max_tokens=16,
                    temperature=0.1,
                    extra_body={'enable_thinking': False},
                ),
                runtime_context,
            )
        ]

    chunks = asyncio.run(collect())

    assert captured['prompt'] == '<chat prompt>'
    assert captured['request_id'].startswith('chatcmpl-')
    assert captured['sampling_params'] == {'temperature': 0.1, 'top_p': 1.0, 'max_tokens': 16}
    assert captured['template_messages'] == [{'role': 'user', 'content': 'hello'}]
    assert captured['template_kwargs'] == {'enable_thinking': False}
    assert captured['tokenize'] is False
    assert captured['add_generation_prompt'] is True
    assert [chunk['delta_text'] for chunk in chunks] == ['hel', 'lo', '']
    assert [chunk['finish_reason'] for chunk in chunks] == [None, None, 'stop']


def test_vllm_backend_chat_completion_stream_passes_async_engine_multimodal_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async vision chat should pass decoded images to vLLM when generate accepts multi_modal_data."""
    captured = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            captured['template_messages'] = messages
            return '<vision prompt>'

    class FakeOutput:
        text = 'ok'
        finish_reason = 'stop'

    class FakeRequestOutput:
        outputs = [FakeOutput()]

    class FakeAsyncEngine:
        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompt, sampling_params, request_id, **kwargs):
            captured['prompt'] = prompt
            captured['generate_kwargs'] = kwargs

            async def iterator():
                yield FakeRequestOutput()

            return iterator()

    monkeypatch.setattr(
        VLLMBackend,
        '_load_async_engine_image_asset',
        lambda self, image_url: f'image:{image_url}',
    )
    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'generate',
        'request_defaults': {},
        'request_policy': {},
        'served_model_name': 'mineru',
        'capabilities': ['vision'],
    }
    backend = VLLMBackend(runtime_spec)
    backend.engine = FakeAsyncEngine()
    backend.engine_kind = 'async'
    backend.engine_state = 'ready'

    async def collect() -> list[dict | bytes | str]:
        return [
            chunk
            async for chunk in backend.chat_completion_stream(
                runtime_spec,
                ChatCompletionsRequest(
                    model='mineru',
                    messages=[
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': 'describe'},
                                {
                                    'type': 'image_url',
                                    'image_url': {'url': 'data:image/png;base64,AAAA'},
                                },
                            ],
                        }
                    ],
                    stream=True,
                ),
                {'deployment_name': 'model-mineru', 'served_model_name': 'mineru'},
            )
        ]

    chunks = asyncio.run(collect())

    assert captured['prompt'] == '<vision prompt>'
    assert captured['generate_kwargs'] == {
        'multi_modal_data': {'image': ['image:data:image/png;base64,AAAA']}
    }
    assert captured['template_messages'] == [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'describe'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}},
            ],
        }
    ]
    assert [chunk['delta_text'] for chunk in chunks] == ['ok', '']
    assert [chunk['finish_reason'] for chunk in chunks] == [None, 'stop']


def test_vllm_backend_chat_completion_stream_passes_async_engine_multimodal_prompt_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older AsyncLLMEngine signatures should receive multimodal data inside the prompt input."""
    captured = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            return '<vision prompt>'

    class FakeOutput:
        text = 'ok'
        finish_reason = 'stop'

    class FakeRequestOutput:
        outputs = [FakeOutput()]

    class FakeAsyncEngine:
        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompt, sampling_params, request_id):
            captured['prompt'] = prompt

            async def iterator():
                yield FakeRequestOutput()

            return iterator()

    monkeypatch.setattr(
        VLLMBackend,
        '_load_async_engine_image_asset',
        lambda self, image_url: f'image:{image_url}',
    )
    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'generate',
        'request_defaults': {},
        'request_policy': {},
        'served_model_name': 'mineru',
        'capabilities': ['vision'],
    }
    backend = VLLMBackend(runtime_spec)
    backend.engine = FakeAsyncEngine()
    backend.engine_kind = 'async'
    backend.engine_state = 'ready'

    async def collect() -> list[dict | bytes | str]:
        return [
            chunk
            async for chunk in backend.chat_completion_stream(
                runtime_spec,
                ChatCompletionsRequest(
                    model='mineru',
                    messages=[
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': 'describe'},
                                {'type': 'image_url', 'image_url': {'url': 'https://example.test/a.png'}},
                            ],
                        }
                    ],
                    stream=True,
                ),
                {'deployment_name': 'model-mineru', 'served_model_name': 'mineru'},
            )
        ]

    chunks = asyncio.run(collect())

    assert captured['prompt'] == {
        'prompt': '<vision prompt>',
        'multi_modal_data': {'image': ['image:https://example.test/a.png']},
    }
    assert [chunk['delta_text'] for chunk in chunks] == ['ok', '']
    assert [chunk['finish_reason'] for chunk in chunks] == [None, 'stop']


def test_vllm_backend_chat_completion_uses_async_engine_generate_result() -> None:
    """Non-streaming chat should aggregate the final AsyncLLMEngine output."""
    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            return '<chat prompt>'

    class FakeOutput:
        text = 'final answer'
        finish_reason = 'stop'
        token_ids = [1, 2, 3]

    class FakeRequestOutput:
        outputs = [FakeOutput()]
        prompt_token_ids = [4, 5]

    class FakeAsyncEngine:
        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompt, sampling_params, request_id):
            async def iterator():
                yield FakeRequestOutput()

            return iterator()

    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'generate',
        'request_defaults': {},
        'request_policy': {},
        'served_model_name': 'qwen3-chat',
        'capabilities': [],
    }
    backend = VLLMBackend(runtime_spec)
    backend.engine = FakeAsyncEngine()
    backend.engine_kind = 'async'
    backend.engine_state = 'ready'

    response = asyncio.run(
        backend.chat_completion(
            runtime_spec,
            ChatCompletionsRequest(
                model='qwen3-chat',
                messages=[{'role': 'user', 'content': 'hello'}],
                stream=False,
            ),
            {'deployment_name': 'model-qwen3-32b-instruct'},
        )
    )

    assert response['content'] == 'final answer'
    assert response['finish_reason'] == 'stop'
    assert response['usage'] == {
        'prompt_tokens': 2,
        'completion_tokens': 3,
        'total_tokens': 5,
    }


def test_vllm_backend_startup_prefers_async_engine_for_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real chat startup should prefer AsyncLLMEngine without requiring model config changes."""
    captured = {}

    class FakeAsyncEngineArgs:
        def __init__(
            self,
            *,
            model,
            tensor_parallel_size,
            dtype,
            task,
            gpu_memory_utilization=None,
            max_model_len=None,
        ):
            captured['engine_args'] = {
                'model': model,
                'tensor_parallel_size': tensor_parallel_size,
                'dtype': dtype,
                'task': task,
                'gpu_memory_utilization': gpu_memory_utilization,
                'max_model_len': max_model_len,
            }

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, engine_args):
            captured['from_engine_args'] = engine_args
            return cls()

    vllm_module = types.ModuleType('vllm')
    vllm_module.__path__ = []
    engine_module = types.ModuleType('vllm.engine')
    engine_module.__path__ = []
    arg_utils_module = types.ModuleType('vllm.engine.arg_utils')
    arg_utils_module.AsyncEngineArgs = FakeAsyncEngineArgs
    async_engine_module = types.ModuleType('vllm.engine.async_llm_engine')
    async_engine_module.AsyncLLMEngine = FakeAsyncLLMEngine
    monkeypatch.setitem(sys.modules, 'vllm', vllm_module)
    monkeypatch.setitem(sys.modules, 'vllm.engine', engine_module)
    monkeypatch.setitem(sys.modules, 'vllm.engine.arg_utils', arg_utils_module)
    monkeypatch.setitem(sys.modules, 'vllm.engine.async_llm_engine', async_engine_module)

    backend = VLLMBackend(
        {
            'backend_init_mode': 'real',
            'backend': 'vllm',
            'task_mode': 'generate',
            'model_path': '/models/qwen',
            'tensor_parallel_size': 1,
            'dtype': 'bfloat16',
            'gpu_memory_utilization': 0.35,
            'max_model_len': 4096,
            'engine_kwargs': {'trust_remote_code': True},
            'model_loading_config': {},
        }
    )

    backend.startup()

    assert backend.engine_kind == 'async'
    assert isinstance(backend.engine, FakeAsyncLLMEngine)
    assert backend.openai_serving_chat_adapter is None
    assert captured['engine_args'] == {
        'model': '/models/qwen',
        'tensor_parallel_size': 1,
        'dtype': 'bfloat16',
        'task': 'generate',
        'gpu_memory_utilization': 0.35,
        'max_model_len': 4096,
    }
    assert captured['from_engine_args'] is not None


def test_vllm_backend_chat_completion_stream_wraps_non_incremental_vllm_result() -> None:
    """Sync vLLM chat results should still be exposed as SSE-compatible delta events."""
    class FakeOutput:
        text = 'full response'
        finish_reason = 'stop'
        token_ids = [1, 2]

    class FakeRequestOutput:
        outputs = [FakeOutput()]
        prompt_token_ids = [3]

    class FakeNonStreamingLLM:
        def chat(self, messages, *, temperature, top_p, max_tokens):
            return [FakeRequestOutput()]

    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'generate',
        'request_defaults': {},
        'request_policy': {},
        'served_model_name': 'qwen3-chat',
        'capabilities': [],
    }
    backend = VLLMBackend(runtime_spec)
    backend.engine = FakeNonStreamingLLM()
    backend.engine_state = 'ready'

    async def collect() -> list[dict | bytes | str]:
        return [
            chunk
            async for chunk in backend.chat_completion_stream(
                runtime_spec,
                ChatCompletionsRequest(
                    model='qwen3-chat',
                    messages=[{'role': 'user', 'content': 'hello'}],
                    stream=True,
                ),
                {'deployment_name': 'model-qwen3-32b-instruct'},
            )
        ]

    chunks = asyncio.run(collect())

    assert [chunk['delta_text'] for chunk in chunks] == ['full response', '']
    assert [chunk['finish_reason'] for chunk in chunks] == [None, 'stop']


def test_vllm_backend_chat_completion_stream_prefers_openai_serving_for_sync_engines() -> None:
    """Sync engines should use OpenAI serving stream passthrough when the adapter is available."""
    class FakeOutput:
        text = 'fallback'
        finish_reason = 'stop'
        token_ids = [1]

    class FakeRequestOutput:
        outputs = [FakeOutput()]
        prompt_token_ids = [2]

    class FakeNonStreamingLLM:
        def chat(self, messages, *, temperature, top_p, max_tokens):
            return [FakeRequestOutput()]

    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'generate',
        'request_defaults': {},
        'request_policy': {
            'allow_tools': True,
            'allow_reasoning': True,
            'passthrough_unknown_openai_fields': True,
        },
        'openai_serving': {'enabled': True},
        'served_model_name': 'qwen3-chat',
        'capabilities': [],
    }
    runtime_context = {
        'deployment_name': 'model-qwen3-32b-instruct',
        'served_model_name': 'qwen3-chat',
    }
    adapter = FakeOpenAIChatServingAdapter(stream_chunks=[b'data: [DONE]\n\n'])
    backend = VLLMBackend(runtime_spec)
    backend.engine = FakeNonStreamingLLM()
    backend.engine_state = 'ready'
    backend.openai_serving_chat_adapter = adapter

    async def collect() -> list[dict | bytes | str]:
        return [
            chunk
            async for chunk in backend.chat_completion_stream(
                runtime_spec,
                ChatCompletionsRequest(
                    model='qwen3-chat',
                    messages=[{'role': 'user', 'content': 'hello'}],
                    stream=True,
                ),
                runtime_context,
            )
        ]

    chunks = asyncio.run(collect())

    assert chunks == [b'data: [DONE]\n\n']
    assert adapter.requests == [
        {
            'model': 'qwen3-chat',
            'messages': [{'role': 'user', 'content': 'hello'}],
            'stream': True,
        }
    ]


def test_vllm_backend_builds_openai_serving_stream_payload_for_legacy_adapter() -> None:
    """Legacy OpenAI serving payload construction remains available for explicit fallback paths."""
    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'generate',
        'request_defaults': {},
        'request_policy': {
            'allow_tools': True,
            'allow_reasoning': True,
            'passthrough_unknown_openai_fields': True,
        },
        'openai_serving': {'enabled': True},
        'served_model_name': 'qwen3-chat',
        'capabilities': [],
    }
    runtime_context = {
        'deployment_name': 'model-qwen3-32b-instruct',
        'served_model_name': 'qwen3-chat',
    }
    backend = VLLMBackend(runtime_spec)
    payload = backend._build_openai_serving_request_payload(
        ChatCompletionsRequest(
            model='qwen3-chat',
            messages=[
                {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': 'call_1',
                            'type': 'function',
                            'function': {'name': 'lookup', 'arguments': '{}'},
                        }
                    ],
                }
            ],
            stream=True,
            stream_options={'include_usage': True},
        ),
        runtime_spec=runtime_spec,
        runtime_context=runtime_context,
    )

    assert payload == {
        'model': 'qwen3-chat',
        'messages': [
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'type': 'function',
                        'function': {'name': 'lookup', 'arguments': '{}'},
                    }
                ],
            }
        ],
        'stream': True,
        'stream_options': {'include_usage': True},
    }


def test_openai_serving_engine_client_proxy_adapts_sync_llm_generate_signature() -> None:
    """Compat proxy should drop async-engine request_id args for sync LLM.generate."""
    captured = {}

    class FakeSyncLLM:
        model_config = 'model-config'

        def generate(self, prompts, sampling_params=None, *, use_tqdm=True):
            captured['prompts'] = prompts
            captured['sampling_params'] = sampling_params
            captured['use_tqdm'] = use_tqdm
            return ['output']

    proxy = OpenAIServingEngineClientCompatProxy(FakeSyncLLM(), [])
    result = proxy.generate('prompt', 'sampling', 'request-id', use_tqdm=False, trace_headers={})

    async def collect() -> list[str]:
        return [item async for item in result]

    assert asyncio.run(collect()) == ['output']
    assert captured == {
        'prompts': 'prompt',
        'sampling_params': 'sampling',
        'use_tqdm': False,
    }


def test_vllm_backend_initializes_openai_serving_adapter_via_dynamic_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamic import + constructor path should build a native serving adapter when symbols exist."""
    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'generate',
        'model_path': '/models/Qwen/Qwen3',
        'engine_kwargs': {
            'enable_auto_tool_choice': True,
            'tool_call_parser': 'qwen3_xml',
        },
        'request_defaults': {'chat_template_kwargs': {'enable_thinking': False}},
        'openai_serving': {
            'enabled': True,
            'reasoning_parser': 'qwen3',
        },
        'served_model_name': 'qwen3-chat',
    }
    backend = VLLMBackend(runtime_spec)
    engine_client = types.SimpleNamespace(
        model_config='model-config',
        renderer='renderer',
        io_processor='io-processor',
    )
    backend.engine = types.SimpleNamespace(llm_engine=engine_client)

    module_map = {
        'vllm.entrypoints.openai.chat_completion.protocol': types.SimpleNamespace(
            ChatCompletionRequest=FakeServingRequest
        ),
        'vllm.entrypoints.openai.chat_completion.serving': types.SimpleNamespace(
            OpenAIServingChat=FakeOpenAIServingChatNative
        ),
        'vllm.entrypoints.openai.models.serving': types.SimpleNamespace(
            OpenAIServingModels=FakeOpenAIServingModelsNative
        ),
        'vllm.entrypoints.openai.models.protocol': types.SimpleNamespace(
            BaseModelPath=FakeBaseModelPath
        ),
        'vllm.entrypoints.serve.render.serving': types.SimpleNamespace(
            OpenAIServingRender=FakeOpenAIServingRender
        ),
    }

    def fake_import_module(name: str):
        if name not in module_map:
            raise ImportError(name)
        return module_map[name]

    monkeypatch.setattr('infer_nexus.backends.vllm.importlib.import_module', fake_import_module)

    adapter = backend._initialize_openai_serving_chat_adapter()

    assert isinstance(adapter, DynamicVLLMOpenAIChatServingAdapter)
    assert backend.openai_serving_adapter_init_error is None
    serving_engine_client = adapter.serving_chat.engine_client
    assert getattr(serving_engine_client, "_client", serving_engine_client) is engine_client
    assert adapter.serving_chat.response_role == 'assistant'
    assert adapter.serving_chat.enable_auto_tools is True
    assert adapter.serving_chat.tool_parser == 'qwen3_xml'
    assert adapter.serving_chat.reasoning_parser == 'qwen3'
    assert adapter.serving_chat.default_chat_template_kwargs == {'enable_thinking': False}
    assert adapter.serving_chat.models.base_model_paths[0].name == 'qwen3-chat'
    assert adapter.serving_chat.models.base_model_paths[0].model_path == '/models/Qwen/Qwen3'
    assert adapter.serving_chat.openai_serving_render.chat_template_content_format == 'auto'

    response = asyncio.run(
        adapter.chat_completion(
            {
                'model': 'qwen3-chat',
                'messages': [{'role': 'user', 'content': 'hello'}],
                'stream': False,
            }
        )
    )

    assert response == {
        'id': 'chatcmpl-native',
        'object': 'chat.completion',
        'model': 'qwen3-chat',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'native'}}],
    }
    assert adapter.serving_chat.requests[0].payload['messages'] == [
        {'role': 'user', 'content': 'hello'}
    ]


def test_vllm_backend_openai_serving_adapter_init_reports_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing vLLM serving symbols should fall back cleanly and retain the failure reason."""
    runtime_spec = {
        'backend': 'vllm',
        'task_mode': 'generate',
        'model_path': '/models/Qwen/Qwen3',
        'request_defaults': {},
        'openai_serving': {'enabled': True},
    }
    backend = VLLMBackend(runtime_spec)
    backend.engine = object()

    def fake_import_module(name: str):
        raise ImportError(f'missing {name}')

    monkeypatch.setattr('infer_nexus.backends.vllm.importlib.import_module', fake_import_module)

    adapter = backend._initialize_openai_serving_chat_adapter()

    assert adapter is None
    assert backend.openai_serving_adapter_init_error is not None
    assert 'Unable to resolve supported vLLM OpenAI serving imports' in (
        backend.openai_serving_adapter_init_error
    )


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
