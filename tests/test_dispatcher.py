"""运行时分发器测试。"""

import asyncio
import json
from time import perf_counter

import httpx
import pytest
from starlette.responses import Response
from starlette.responses import StreamingResponse

from infer_nexus.catalog.models import ModelCatalogFile, ModelConfig
from infer_nexus.catalog.loader import load_model_catalog
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.core.config import Settings
from infer_nexus.core.enums import BackendType, TaskType
from infer_nexus.core.errors import AdmissionRejectedError, RuntimeNotConnectedError
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest
from infer_nexus.model_store import LocalModelStore
from infer_nexus.observability.metrics import render_prometheus_metrics
from infer_nexus.runtime.dispatcher import RuntimeDispatcher
from infer_nexus.runtime.executor import RuntimeExecutor
from infer_nexus.runtime.handles import ServeDeploymentHandleResolver
from infer_nexus.runtime.serve_app import ServeApplicationBuilder
from infer_nexus.runtime.types import RuntimeTarget


def make_dispatcher(executor: RuntimeExecutor | None = None) -> tuple[ModelRegistry, LocalModelStore, RuntimeDispatcher]:
    """构造带测试模型目录的运行时分发器。"""
    registry = ModelRegistry(load_model_catalog('config/models.yaml'))
    store = LocalModelStore('models')
    for relative_path in [
        'Qwen/Qwen3-32B-Instruct',
        'BAAI/bge-large-zh-v1.5',
        'BAAI/bge-reranker-v2-m3',
    ]:
        (store.root_dir / relative_path).mkdir(parents=True, exist_ok=True)
    builder = ServeApplicationBuilder(model_store=store)
    dispatcher = RuntimeDispatcher(
        registry=registry,
        serve_builder=builder,
        executor=executor or RuntimeExecutor(),
    )
    return registry, store, dispatcher


def test_dispatcher_resolves_target_with_runtime_context() -> None:
    """分发器应为模型解析出完整运行时目标。"""
    registry, _, dispatcher = make_dispatcher()

    target = dispatcher.resolve_target(registry.get('qwen3-chat'))

    assert target.model_name == 'qwen3-32b-instruct'
    assert target.model_alias == 'qwen3-chat'
    assert target.app_name == 'infer-nexus-model-qwen3-32b-instruct'
    assert target.deployment_name == 'model-qwen3-32b-instruct'
    assert target.runtime_context['resolved_model_path'].endswith('/models/Qwen/Qwen3-32B-Instruct')
    assert target.runtime_context['app_name'] == 'infer-nexus-model-qwen3-32b-instruct'
    assert target.runtime_context['deployment_name'] == 'model-qwen3-32b-instruct'


def test_dispatch_chat_returns_stub_chat_completion() -> None:
    """stub 模式下 chat 分发应返回占位响应。"""
    registry, _, dispatcher = make_dispatcher()
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
    )

    response = asyncio.run(dispatcher.dispatch_chat(registry.get('qwen3-chat'), request))

    assert response.object == 'chat.completion'
    assert response.model == 'qwen3-chat'
    assert response.choices[0].message.role == 'assistant'
    assert 'backend stub response from vllm' in str(response.choices[0].message.content)
    assert 'model-qwen3-32b-instruct' in str(response.choices[0].message.content)


def test_dispatch_chat_returns_sse_stream_when_requested() -> None:
    """stub 模式下 stream=True 应返回 OpenAI-style SSE 响应。"""
    registry, _, dispatcher = make_dispatcher()
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
        stream=True,
    )

    response = asyncio.run(dispatcher.dispatch_chat(registry.get('qwen3-chat'), request))

    assert isinstance(response, StreamingResponse)

    async def collect() -> bytes:
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return b''.join(chunks)

    body = asyncio.run(collect())
    assert response.media_type == 'text/event-stream'
    assert b'chat.completion.chunk' in body
    assert b'backend stub response from vllm' in body
    assert body.endswith(b'data: [DONE]\n\n')


def test_chat_stream_response_preserves_vllm_reasoning_deltas() -> None:
    """Streaming adapter should preserve vLLM/OpenAI reasoning delta chunks."""
    executor = RuntimeExecutor()
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
        stream=True,
    )
    target = RuntimeTarget(
        model_name='qwen3-32b-instruct',
        model_alias='qwen3-chat',
        backend=BackendType.VLLM,
        app_name='infer-nexus-model-qwen3-32b-instruct',
        deployment_name='model-qwen3-32b-instruct',
        runtime_context={'served_model_name': 'qwen3-chat'},
    )

    response = executor._build_chat_stream_response(
        request,
        target,
        {
            'stream_chunks': [
                {
                    'id': 'chatcmpl-reasoning',
                    'object': 'chat.completion.chunk',
                    'created': 123,
                    'model': 'qwen3-chat',
                    'choices': [
                        {
                            'index': 0,
                            'delta': {'role': 'assistant', 'reasoning_content': '先分析'},
                            'finish_reason': None,
                        }
                    ],
                },
                {
                    'id': 'chatcmpl-reasoning',
                    'object': 'chat.completion.chunk',
                    'created': 123,
                    'model': 'qwen3-chat',
                    'choices': [
                        {
                            'index': 0,
                            'delta': {'content': '答案'},
                            'finish_reason': None,
                        }
                    ],
                },
            ],
        },
        stream_start_time=perf_counter(),
    )

    async def collect() -> bytes:
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return b''.join(chunks)

    body = asyncio.run(collect())
    assert b'"reasoning_content":' in body
    assert '先分析'.encode() in body
    assert b'"content":' in body
    assert '答案'.encode() in body
    assert body.endswith(b'data: [DONE]\n\n')


def test_chat_stream_response_forwards_raw_sse_bytes_unchanged() -> None:
    """Raw SSE bytes from a backend should pass through without semantic rewriting."""
    executor = RuntimeExecutor()
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
        stream=True,
    )
    target = RuntimeTarget(
        model_name='qwen3-32b-instruct',
        model_alias='qwen3-chat',
        backend=BackendType.VLLM,
        app_name='infer-nexus-model-qwen3-32b-instruct',
        deployment_name='model-qwen3-32b-instruct',
        runtime_context={'served_model_name': 'qwen3-chat'},
    )
    raw = b'data: {"custom": true}\n\ndata: [DONE]\n\n'

    response = executor._build_chat_stream_response(
        request,
        target,
        {'stream_chunks': [raw]},
        stream_start_time=perf_counter(),
    )

    async def collect() -> bytes:
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return b''.join(chunks)

    assert asyncio.run(collect()) == raw


def test_chat_stream_response_maps_standard_delta_events_to_openai_sse() -> None:
    """Backend-standard delta events should be exposed as OpenAI-compatible SSE chunks."""
    executor = RuntimeExecutor()
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
        stream=True,
    )
    target = RuntimeTarget(
        model_name='qwen3-32b-instruct',
        model_alias='qwen3-chat',
        backend=BackendType.VLLM,
        app_name='infer-nexus-model-qwen3-32b-instruct',
        deployment_name='model-qwen3-32b-instruct',
        runtime_context={'served_model_name': 'qwen3-chat'},
    )

    async def chunks():
        yield {
            'type': 'chat_delta',
            'id': 'chatcmpl-native',
            'created': 123,
            'model': 'qwen3-chat',
            'delta_text': 'hel',
            'finish_reason': None,
        }
        yield {
            'type': 'chat_delta',
            'id': 'chatcmpl-native',
            'created': 123,
            'model': 'qwen3-chat',
            'delta_text': 'lo',
            'finish_reason': None,
        }
        yield {
            'type': 'chat_delta',
            'id': 'chatcmpl-native',
            'created': 123,
            'model': 'qwen3-chat',
            'delta_text': '',
            'finish_reason': 'stop',
        }

    response = executor._build_chat_stream_response(
        request,
        target,
        chunks(),
        stream_start_time=perf_counter(),
    )

    async def collect() -> list[dict | str]:
        payloads = []
        async for chunk in response.body_iterator:
            text = chunk.decode('utf-8')
            for line in text.splitlines():
                if not line.startswith('data: '):
                    continue
                payload = line.removeprefix('data: ')
                payloads.append('[DONE]' if payload == '[DONE]' else json.loads(payload))
        return payloads

    payloads = asyncio.run(collect())

    assert payloads[0]['choices'][0]['delta'] == {'role': 'assistant'}
    assert payloads[1]['choices'][0]['delta'] == {'content': 'hel'}
    assert payloads[2]['choices'][0]['delta'] == {'content': 'lo'}
    assert payloads[3]['choices'][0]['finish_reason'] == 'stop'
    assert payloads[4] == '[DONE]'


def test_dispatch_embedding_returns_stub_embedding_response() -> None:
    """stub 模式下 embedding 分发应返回占位向量。"""
    registry, _, dispatcher = make_dispatcher()
    request = EmbeddingRequest(
        model='bge-embedding',
        input='hello',
    )

    response = asyncio.run(dispatcher.dispatch_embedding(registry.get('bge-embedding'), request))

    assert response.object == 'list'
    assert response.model == 'bge-embedding'
    assert response.data[0].embedding == [5.0, 0.0, 4.0]


def test_dispatch_rerank_returns_stub_rerank_response() -> None:
    """stub 模式下 rerank 分发应返回相关度排序结果。"""
    registry, _, dispatcher = make_dispatcher()
    request = RerankRequest(
        model='bge-rerank',
        query='capital of france',
        documents=[
            'The capital of Brazil is Brasilia.',
            'The capital of France is Paris.',
            'Python is a programming language.',
        ],
        top_n=2,
    )

    response = asyncio.run(dispatcher.dispatch_rerank(registry.get('bge-rerank'), request))

    assert response.model == 'bge-rerank'
    assert len(response.results) == 2
    assert response.results[0].index == 1
    assert response.results[0].document.text == 'The capital of France is Paris.'


class FakeDeploymentMethod:
    """模拟 Serve 句柄上的远程方法。"""

    def __init__(self, payload: dict) -> None:
        """保存固定返回负载。"""
        self.payload = payload

    async def remote(self, request_payload: dict) -> dict:
        """回显请求负载并返回预设响应。"""
        result = dict(self.payload)
        result['payload'] = request_payload
        return result


class FakeKeywordOnlyDeploymentMethod:
    """模拟仅接受关键字载荷的 Serve 远程方法。"""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def remote(self, *, request_payload: dict) -> dict:
        result = dict(self.payload)
        result['payload'] = request_payload
        return result


class FakeDeploymentHandle:
    """模拟单个模型部署句柄。"""

    def __init__(self, deployment_name: str) -> None:
        """初始化 chat/embedding/rerank 三类远程方法。"""
        self.chat_completion = FakeDeploymentMethod(
            {
                'status': 'ok',
                'deployment': deployment_name,
                'model': 'qwen3-chat',
            }
        )
        self.embedding = FakeDeploymentMethod(
            {
                'status': 'ok',
                'deployment': deployment_name,
                'model': 'bge-embedding',
                'data': [{'index': 0, 'embedding': [5.0, 0.0, 4.0]}],
                'usage': {
                    'prompt_tokens': 1,
                    'completion_tokens': 0,
                    'total_tokens': 1,
                },
            }
        )
        self.rerank = FakeDeploymentMethod(
            {
                'status': 'ok',
                'deployment': deployment_name,
                'model': 'bge-rerank',
                'usage': {'total_tokens': 4},
                'results': [
                    {
                        'index': 1,
                        'document': {'text': 'The capital of France is Paris.'},
                        'relevance_score': 0.99,
                    }
                ],
            }
        )


class FakeKeywordOnlyDeploymentHandle(FakeDeploymentHandle):
    """模拟仅接受关键字请求载荷的部署句柄。"""

    def __init__(self, deployment_name: str) -> None:
        super().__init__(deployment_name)
        self.chat_completion = FakeKeywordOnlyDeploymentMethod(
            {
                'status': 'ok',
                'deployment': deployment_name,
                'model': 'qwen3-chat',
            }
        )


class FakeStreamingDeploymentMethod:
    """模拟 Serve 句柄上的 streaming 远程方法。"""

    def __init__(self, captured_payloads: list[dict]) -> None:
        self.captured_payloads = captured_payloads

    def remote(self, request_payload: dict):
        self.captured_payloads.append(request_payload)

        async def iterator():
            yield {
                'id': 'chatcmpl-stream',
                'object': 'chat.completion.chunk',
                'created': 123,
                'model': request_payload['model'],
                'choices': [
                    {
                        'index': 0,
                        'delta': {'role': 'assistant', 'content': 'hello'},
                        'finish_reason': None,
                    }
                ],
            }
            yield {
                'id': 'chatcmpl-stream',
                'object': 'chat.completion.chunk',
                'created': 123,
                'model': request_payload['model'],
                'choices': [
                    {
                        'index': 0,
                        'delta': {},
                        'finish_reason': 'stop',
                    }
                ],
            }

        return iterator()


class FakeStreamingDeploymentHandle(FakeDeploymentHandle):
    """模拟支持 streaming chat 的部署句柄。"""

    def __init__(self, deployment_name: str, captured_payloads: list[dict]) -> None:
        super().__init__(deployment_name)
        self.chat_completion_stream = FakeStreamingDeploymentMethod(captured_payloads)
        self.options_calls: list[dict] = []

    def options(self, **kwargs):
        self.options_calls.append(kwargs)
        return self


class FakeStreamingServe:
    """模拟支持 streaming 方法的 Serve runtime。"""

    def __init__(self) -> None:
        self.captured_payloads: list[dict] = []
        self.handles: list[FakeStreamingDeploymentHandle] = []

    def get_deployment_handle(self, deployment_name: str, app_name: str) -> FakeStreamingDeploymentHandle:
        assert app_name.startswith('infer-nexus-model-')
        handle = FakeStreamingDeploymentHandle(deployment_name, self.captured_payloads)
        self.handles.append(handle)
        return handle


class FakeOpenAIChatDeploymentHandle:
    """模拟返回完整 OpenAI chat payload 的部署句柄。"""

    def __init__(self) -> None:
        self.chat_completion = FakeDeploymentMethod(
            {
                'object': 'chat.completion',
                'id': 'chatcmpl-native',
                'created': 123,
                'model': 'qwen3-chat',
                'choices': [
                    {
                        'index': 0,
                        'message': {'role': 'assistant', 'content': 'first'},
                        'finish_reason': 'stop',
                    },
                    {
                        'index': 1,
                        'message': {'role': 'assistant', 'content': 'second'},
                        'finish_reason': 'stop',
                    },
                ],
                'usage': {
                    'prompt_tokens': 1,
                    'completion_tokens': 2,
                    'total_tokens': 3,
                    'completion_tokens_details': {'reasoning_tokens': 1},
                },
            }
        )


class FakeOpenAIChatServe:
    """模拟返回完整 OpenAI chat payload 的 Serve runtime。"""

    def get_deployment_handle(self, deployment_name: str, app_name: str) -> FakeOpenAIChatDeploymentHandle:
        assert app_name.startswith('infer-nexus-model-')
        return FakeOpenAIChatDeploymentHandle()


class FakeServe:
    """模拟 Serve runtime。"""

    def get_deployment_handle(self, deployment_name: str, app_name: str) -> FakeDeploymentHandle:
        """返回模拟部署句柄。"""
        assert app_name.startswith('infer-nexus-model-')
        return FakeDeploymentHandle(deployment_name)


class FakeKeywordOnlyServe:
    """模拟只接受关键字 payload 的 Serve runtime。"""

    def get_deployment_handle(self, deployment_name: str, app_name: str) -> FakeKeywordOnlyDeploymentHandle:
        assert app_name.startswith('infer-nexus-model-')
        return FakeKeywordOnlyDeploymentHandle(deployment_name)


def test_dispatch_chat_uses_serve_handle_in_serve_mode() -> None:
    """serve 模式下 chat 分发应通过 deployment handle 执行。"""
    resolver = ServeDeploymentHandleResolver(serve=FakeServe())
    executor = RuntimeExecutor(mode='serve', handle_resolver=resolver)
    registry, _, dispatcher = make_dispatcher(executor)
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
    )

    response = asyncio.run(dispatcher.dispatch_chat(registry.get('qwen3-chat'), request))

    assert response.object == 'chat.completion'
    assert response.model == 'qwen3-chat'
    assert 'serve chat response from deployment' in str(response.choices[0].message.content)
    assert 'model-qwen3-32b-instruct' in str(response.choices[0].message.content)


def test_dispatcher_reuses_precomputed_catalog_target() -> None:
    """Request routing never derives Ray application names from client input."""
    registry, _, dispatcher = make_dispatcher()
    model = registry.get('qwen3.5-27b')

    first = dispatcher.resolve_target(model)
    second = dispatcher.resolve_target(model)

    assert first is second
    assert first.app_name == 'infer-nexus-model-Qwen3.5-27B'
    assert first.deployment_name == 'model-Qwen3.5-27B'


def test_dispatcher_uses_the_deployed_gateway_target_snapshot() -> None:
    """A restarted ingress cannot reconstruct a different Serve address from mutable config."""
    registry, store, _ = make_dispatcher()
    builder = ServeApplicationBuilder(model_store=store)
    snapshot = builder.build_gateway_spec(registry, settings=Settings()).model_targets
    model = registry.list_models()[0]
    snapshot[model.name] = {
        "application_name": "locked-model-application",
        "deployment_name": "locked-model-deployment",
    }
    dispatcher = RuntimeDispatcher(
        registry=registry,
        serve_builder=builder,
        executor=RuntimeExecutor(),
        model_targets=snapshot,
    )

    target = dispatcher.resolve_target(model)

    assert target.app_name == "locked-model-application"
    assert target.deployment_name == "locked-model-deployment"


def test_dispatch_chat_uses_keyword_request_payload_when_invoking_serve_handle() -> None:
    """Serve chat dispatch should send request payload as an explicit keyword argument."""
    resolver = ServeDeploymentHandleResolver(serve=FakeKeywordOnlyServe())
    executor = RuntimeExecutor(mode='serve', handle_resolver=resolver)
    registry, _, dispatcher = make_dispatcher(executor)
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
    )

    response = asyncio.run(dispatcher.dispatch_chat(registry.get('qwen3-chat'), request))

    assert response.object == 'chat.completion'
    assert response.model == 'qwen3-chat'
    assert 'serve chat response from deployment' in str(response.choices[0].message.content)


def test_dispatch_chat_passthrough_openai_payload_without_collapsing_choices() -> None:
    """完整 OpenAI chat payload 应原样透传，避免 choices 或 usage 扩展字段被裁剪。"""
    resolver = ServeDeploymentHandleResolver(serve=FakeOpenAIChatServe())
    executor = RuntimeExecutor(mode='serve', handle_resolver=resolver)
    registry, _, dispatcher = make_dispatcher(executor)
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
    )

    response = asyncio.run(dispatcher.dispatch_chat(registry.get('qwen3-chat'), request))

    assert isinstance(response, Response)
    body = json.loads(response.body)
    assert body['object'] == 'chat.completion'
    assert len(body['choices']) == 2
    assert body['choices'][1]['message']['content'] == 'second'
    assert body['usage']['completion_tokens_details'] == {'reasoning_tokens': 1}


def test_dispatch_chat_stream_uses_serve_streaming_handle_without_rewriting_request() -> None:
    """serve 模式下 stream=True 应调用 deployment streaming 方法并保留 stream 字段。"""
    fake_serve = FakeStreamingServe()
    resolver = ServeDeploymentHandleResolver(serve=fake_serve)
    executor = RuntimeExecutor(mode='serve', handle_resolver=resolver)
    registry, _, dispatcher = make_dispatcher(executor)
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
        stream=True,
        extra_body={'top_k': 50},
    )

    response = asyncio.run(dispatcher.dispatch_chat(registry.get('qwen3-chat'), request))

    assert isinstance(response, StreamingResponse)
    assert fake_serve.captured_payloads[0]['stream'] is True
    assert fake_serve.captured_payloads[0]['extra_body'] == {'top_k': 50}
    assert fake_serve.handles[0].options_calls[0] == {'stream': True}

    async def collect() -> bytes:
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return b''.join(chunks)

    body = asyncio.run(collect())
    assert b'chat.completion.chunk' in body
    assert b'"content": "hello"' in body
    assert body.endswith(b'data: [DONE]\n\n')


def test_streaming_handle_capability_failure_never_falls_back_to_unary_handle() -> None:
    """A Ray Client-style stream capability error must fail before invoking the model method."""

    class StreamMethod:
        def __init__(self) -> None:
            self.calls = 0

        async def remote(self, *, request_payload: dict) -> dict:
            self.calls += 1
            return {"status": "ok", "payload": request_payload}

    class UnsupportedStreamHandle:
        def __init__(self) -> None:
            self.chat_completion_stream = StreamMethod()

        def options(self, *, stream: bool) -> "UnsupportedStreamHandle":
            assert stream is True
            raise RuntimeError("stream=True is not supported over Ray Client")

    class UnsupportedStreamServe:
        def __init__(self) -> None:
            self.handle = UnsupportedStreamHandle()

        def get_deployment_handle(self, deployment_name: str, app_name: str) -> UnsupportedStreamHandle:
            return self.handle

    serve = UnsupportedStreamServe()
    executor = RuntimeExecutor(
        mode='serve',
        handle_resolver=ServeDeploymentHandleResolver(serve=serve),
    )
    target = RuntimeTarget(
        model_name='registered-model',
        model_alias='registered-model',
        backend=BackendType.VLLM,
        app_name='infer-nexus-model-registered-model',
        deployment_name='model-registered-model',
        runtime_context={'served_model_name': 'registered-model'},
    )

    with pytest.raises(RuntimeNotConnectedError) as exc_info:
        asyncio.run(
            executor._invoke_handle_stream(
                target=target,
                method_name='chat_completion_stream',
                payload={'stream': True},
            )
        )

    assert exc_info.value.code == 'streaming_unavailable'
    assert serve.handle.chat_completion_stream.calls == 0


def test_dispatch_embedding_uses_serve_handle_in_serve_mode() -> None:
    """serve 模式下 embedding 分发应通过 deployment handle 执行。"""
    resolver = ServeDeploymentHandleResolver(serve=FakeServe())
    executor = RuntimeExecutor(mode='serve', handle_resolver=resolver)
    registry, _, dispatcher = make_dispatcher(executor)
    request = EmbeddingRequest(
        model='bge-embedding',
        input='hello',
    )

    response = asyncio.run(dispatcher.dispatch_embedding(registry.get('bge-embedding'), request))

    assert response.object == 'list'
    assert response.model == 'bge-embedding'
    assert response.data[0].embedding == [5.0, 0.0, 4.0]


def test_dispatch_rerank_uses_serve_handle_in_serve_mode() -> None:
    """serve 模式下 rerank 分发应通过 deployment handle 执行。"""
    resolver = ServeDeploymentHandleResolver(serve=FakeServe())
    executor = RuntimeExecutor(mode='serve', handle_resolver=resolver)
    registry, _, dispatcher = make_dispatcher(executor)
    request = RerankRequest(
        model='bge-rerank',
        query='capital of france',
        documents=[
            'The capital of Brazil is Brasilia.',
            'The capital of France is Paris.',
        ],
        top_n=1,
    )

    response = asyncio.run(dispatcher.dispatch_rerank(registry.get('bge-rerank'), request))

    assert response.model == 'bge-rerank'
    assert len(response.results) == 1
    assert response.results[0].index == 1


def test_dispatch_raises_when_serve_mode_has_no_resolver() -> None:
    """serve 模式未提供 resolver 时应抛出运行时未连接错误。"""
    executor = RuntimeExecutor(mode='serve')
    registry, _, dispatcher = make_dispatcher(executor)
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
    )

    with pytest.raises(RuntimeNotConnectedError, match='no handle resolver'):
        asyncio.run(dispatcher.dispatch_chat(registry.get('qwen3-chat'), request))


def test_serve_handle_resolver_caches_handles_by_app_and_deployment() -> None:
    """Serve handles should be reused instead of fetched for every request."""

    class CountingServe:
        def __init__(self) -> None:
            self.calls = 0

        def get_deployment_handle(self, deployment_name: str, app_name: str) -> object:
            self.calls += 1
            return {"deployment_name": deployment_name, "app_name": app_name}

    serve = CountingServe()
    resolver = ServeDeploymentHandleResolver(serve=serve)

    first = resolver.get_handle("model-qwen3", app_name="infer-nexus-model-qwen3")
    second = resolver.get_handle("model-qwen3", app_name="infer-nexus-model-qwen3")

    assert first is second
    assert serve.calls == 1


def test_dispatch_chat_times_out_slow_serve_handle() -> None:
    """Serve handle calls should fail fast when the router or replica stalls."""

    class SlowDeploymentMethod:
        async def remote(self, *, request_payload: dict) -> dict:
            await asyncio.sleep(1)
            return {"status": "ok", "payload": request_payload}

    class SlowDeploymentHandle:
        def __init__(self) -> None:
            self.chat_completion = SlowDeploymentMethod()

    class SlowServe:
        def get_deployment_handle(self, deployment_name: str, app_name: str) -> SlowDeploymentHandle:
            return SlowDeploymentHandle()

    resolver = ServeDeploymentHandleResolver(serve=SlowServe())
    executor = RuntimeExecutor(
        mode='serve',
        handle_resolver=resolver,
        serve_request_timeout_seconds=0.01,
    )
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
    )
    target = RuntimeTarget(
        model_name='qwen3-32b-instruct',
        model_alias='qwen3-chat',
        backend=BackendType.VLLM,
        app_name='infer-nexus-model-qwen3-32b-instruct',
        deployment_name='model-qwen3-32b-instruct',
        runtime_context={'served_model_name': 'qwen3-chat'},
    )

    with pytest.raises(RuntimeNotConnectedError) as exc_info:
        asyncio.run(executor.execute_chat(target=target, request=request))

    assert exc_info.value.code == 'upstream_timeout'


def test_dispatch_chat_rejects_when_gateway_inflight_limit_is_reached() -> None:
    """Gateway-side inflight limits should reject excess requests before Serve routing."""

    async def run_case() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class BlockingDeploymentMethod:
            async def remote(self, *, request_payload: dict) -> dict:
                started.set()
                await release.wait()
                return {"status": "ok", "payload": request_payload}

        class BlockingDeploymentHandle:
            def __init__(self) -> None:
                self.chat_completion = BlockingDeploymentMethod()

        class BlockingServe:
            def get_deployment_handle(self, deployment_name: str, app_name: str) -> BlockingDeploymentHandle:
                return BlockingDeploymentHandle()

        resolver = ServeDeploymentHandleResolver(serve=BlockingServe())
        executor = RuntimeExecutor(
            mode='serve',
            handle_resolver=resolver,
            max_inflight_per_model=1,
        )
        request = ChatCompletionsRequest(
            model='qwen3-chat',
            messages=[{'role': 'user', 'content': 'hello'}],
        )
        target = RuntimeTarget(
            model_name='qwen3-32b-instruct',
            model_alias='qwen3-chat',
            backend=BackendType.VLLM,
            app_name='infer-nexus-model-qwen3-32b-instruct',
            deployment_name='model-qwen3-32b-instruct',
            runtime_context={'served_model_name': 'qwen3-chat'},
        )

        first_task = asyncio.create_task(executor.execute_chat(target=target, request=request))
        await started.wait()
        with pytest.raises(AdmissionRejectedError) as exc_info:
            await executor.execute_chat(target=target, request=request)
        release.set()
        await first_task

        assert exc_info.value.code == 'gateway_overloaded'

    asyncio.run(run_case())


def test_dispatch_chat_opens_circuit_after_repeated_serve_timeouts() -> None:
    """Repeated Serve timeouts should open the deployment circuit breaker."""

    class SlowDeploymentMethod:
        async def remote(self, *, request_payload: dict) -> dict:
            await asyncio.sleep(1)
            return {"status": "ok", "payload": request_payload}

    class SlowDeploymentHandle:
        def __init__(self) -> None:
            self.chat_completion = SlowDeploymentMethod()

    class SlowServe:
        def get_deployment_handle(self, deployment_name: str, app_name: str) -> SlowDeploymentHandle:
            return SlowDeploymentHandle()

    async def run_case() -> None:
        resolver = ServeDeploymentHandleResolver(serve=SlowServe())
        executor = RuntimeExecutor(
            mode='serve',
            handle_resolver=resolver,
            serve_request_timeout_seconds=0.01,
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=1,
            circuit_breaker_cooldown_seconds=60,
        )
        request = ChatCompletionsRequest(
            model='qwen3-chat',
            messages=[{'role': 'user', 'content': 'hello'}],
        )
        target = RuntimeTarget(
            model_name='qwen3-32b-instruct',
            model_alias='qwen3-chat',
            backend=BackendType.VLLM,
            app_name='infer-nexus-model-qwen3-32b-instruct',
            deployment_name='model-qwen3-32b-instruct',
            runtime_context={'served_model_name': 'qwen3-chat'},
        )

        with pytest.raises(RuntimeNotConnectedError) as timeout_info:
            await executor.execute_chat(target=target, request=request)
        with pytest.raises(RuntimeNotConnectedError) as circuit_info:
            await executor.execute_chat(target=target, request=request)

        assert timeout_info.value.code == 'upstream_timeout'
        assert circuit_info.value.code == 'runtime_circuit_open'

    asyncio.run(run_case())


def test_dispatch_chat_uses_bounded_gateway_queue_before_serve_routing() -> None:
    """Gateway-local model queues should absorb only a bounded request burst."""

    async def run_case() -> None:
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        call_count = 0

        class BlockingDeploymentMethod:
            async def remote(self, *, request_payload: dict) -> dict:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    first_started.set()
                    await release_first.wait()
                return {"status": "ok", "payload": request_payload}

        class BlockingDeploymentHandle:
            def __init__(self) -> None:
                self.chat_completion = BlockingDeploymentMethod()

        class BlockingServe:
            def __init__(self) -> None:
                self.handle = BlockingDeploymentHandle()

            def get_deployment_handle(self, deployment_name: str, app_name: str) -> BlockingDeploymentHandle:
                return self.handle

        resolver = ServeDeploymentHandleResolver(serve=BlockingServe())
        executor = RuntimeExecutor(
            mode='serve',
            handle_resolver=resolver,
            max_inflight_per_model=1,
            max_queued_per_model=1,
            admission_queue_timeout_seconds=1,
        )
        request = ChatCompletionsRequest(
            model='qwen3-chat',
            messages=[{'role': 'user', 'content': 'hello'}],
        )
        target = RuntimeTarget(
            model_name='qwen3-32b-instruct',
            model_alias='qwen3-chat',
            backend=BackendType.VLLM,
            app_name='infer-nexus-model-qwen3-32b-instruct',
            deployment_name='model-qwen3-32b-instruct',
            runtime_context={'served_model_name': 'qwen3-chat'},
        )

        first_task = asyncio.create_task(executor.execute_chat(target=target, request=request))
        await first_started.wait()
        queued_task = asyncio.create_task(executor.execute_chat(target=target, request=request))
        await asyncio.sleep(0)

        with pytest.raises(AdmissionRejectedError) as exc_info:
            await executor.execute_chat(target=target, request=request)

        release_first.set()
        await first_task
        await queued_task

        assert exc_info.value.code == 'gateway_queue_full'

    asyncio.run(run_case())


def test_dispatch_chat_stream_uses_idle_timeout_instead_of_total_timeout() -> None:
    """Streaming Serve calls should fail only when chunks stop arriving."""

    class SlowStreamMethod:
        def remote(self, *, request_payload: dict):
            async def iterator():
                await asyncio.sleep(1)
                yield {"type": "chat_delta", "delta_text": "late"}

            return iterator()

    class SlowStreamHandle:
        def __init__(self) -> None:
            self.chat_completion_stream = SlowStreamMethod()

        def options(self, *, stream: bool) -> "SlowStreamHandle":
            assert stream is True
            return self

    class SlowStreamServe:
        def get_deployment_handle(self, deployment_name: str, app_name: str) -> SlowStreamHandle:
            return SlowStreamHandle()

    async def run_case() -> None:
        resolver = ServeDeploymentHandleResolver(serve=SlowStreamServe())
        executor = RuntimeExecutor(
            mode='serve',
            handle_resolver=resolver,
            serve_request_timeout_seconds=30,
            serve_stream_idle_timeout_seconds=0.01,
        )
        request = ChatCompletionsRequest(
            model='qwen3-chat',
            messages=[{'role': 'user', 'content': 'hello'}],
            stream=True,
        )
        target = RuntimeTarget(
            model_name='qwen3-32b-instruct',
            model_alias='qwen3-chat',
            backend=BackendType.VLLM,
            app_name='infer-nexus-model-qwen3-32b-instruct',
            deployment_name='model-qwen3-32b-instruct',
            runtime_context={'served_model_name': 'qwen3-chat'},
        )

        with pytest.raises(RuntimeNotConnectedError) as exc_info:
            await executor.execute_chat(target=target, request=request)

        assert exc_info.value.code == 'upstream_timeout'

    asyncio.run(run_case())


def test_runtime_worker_client_delegates_serve_chat_without_direct_handle() -> None:
    """Configured runtime workers should own Serve execution instead of the HTTP executor."""

    class FakeWorkerClient:
        def __init__(self) -> None:
            self.calls = []

        async def chat_completion(self, *, target: RuntimeTarget, request: ChatCompletionsRequest):
            self.calls.append((target.deployment_name, request.model))
            return Response(content=b'worker-ok', media_type='text/plain')

    worker = FakeWorkerClient()
    executor = RuntimeExecutor(mode='serve', runtime_worker_client=worker)
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
    )
    target = RuntimeTarget(
        model_name='qwen3-32b-instruct',
        model_alias='qwen3-chat',
        backend=BackendType.VLLM,
        app_name='infer-nexus-model-qwen3-32b-instruct',
        deployment_name='model-qwen3-32b-instruct',
        runtime_context={'served_model_name': 'qwen3-chat'},
    )

    response = asyncio.run(executor.execute_chat(target=target, request=request))

    assert isinstance(response, Response)
    assert response.body == b'worker-ok'
    assert worker.calls == [('model-qwen3-32b-instruct', 'qwen3-chat')]


def test_gateway_runtime_metrics_render_after_serve_call() -> None:
    """Serve handle health metrics should be visible through Prometheus rendering."""

    class FastDeploymentMethod:
        async def remote(self, *, request_payload: dict) -> dict:
            return {"status": "ok", "payload": request_payload}

    class FastDeploymentHandle:
        def __init__(self) -> None:
            self.chat_completion = FastDeploymentMethod()

    class FastServe:
        def get_deployment_handle(self, deployment_name: str, app_name: str) -> FastDeploymentHandle:
            return FastDeploymentHandle()

    resolver = ServeDeploymentHandleResolver(serve=FastServe())
    executor = RuntimeExecutor(mode='serve', handle_resolver=resolver, max_inflight_per_model=1)
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
    )
    target = RuntimeTarget(
        model_name='qwen3-32b-instruct',
        model_alias='qwen3-chat',
        backend=BackendType.VLLM,
        app_name='infer-nexus-model-qwen3-32b-instruct',
        deployment_name='model-qwen3-32b-instruct',
        runtime_context={'served_model_name': 'qwen3-chat'},
    )

    asyncio.run(executor.execute_chat(target=target, request=request))

    metrics = render_prometheus_metrics()[0].decode("utf-8")
    assert 'infer_nexus_serve_handle_calls_total{model="qwen3-chat",method="chat_completion",status="success"}' in metrics
    assert 'infer_nexus_model_inflight{model="qwen3-chat"}' in metrics


def _build_proxy_dispatcher() -> tuple[ModelRegistry, RuntimeDispatcher]:
    model = ModelConfig(
        name="mineru-proxy",
        alias="mineru",
        task=TaskType.CHAT,
        backend=BackendType.VLLM_OPENAI_PROXY,
        tensor_parallel_size=1,
        cpu_per_replica=1,
        gpu_per_replica=0,
        min_replicas=1,
        max_replicas=1,
        proxy_config={
            "upstream_base_url": "http://upstream.local/v1",
            "upstream_model_name": "opendatalab/MinerU2.5-2509-1.2B",
            "auth": {"mode": "none"},
        },
    )
    embedding = ModelConfig(
        name="bge-embedding-proxy",
        alias="bge-embedding-proxy",
        task=TaskType.EMBEDDING,
        backend=BackendType.VLLM_OPENAI_PROXY,
        tensor_parallel_size=1,
        cpu_per_replica=1,
        gpu_per_replica=0,
        min_replicas=1,
        max_replicas=1,
        proxy_config={
            "upstream_base_url": "http://upstream.local/v1",
            "upstream_model_name": "BAAI/bge-large-zh-v1.5",
            "auth": {"mode": "none"},
        },
    )
    rerank = ModelConfig(
        name="bge-rerank-proxy",
        alias="bge-rerank-proxy",
        task=TaskType.RERANK,
        backend=BackendType.VLLM_OPENAI_PROXY,
        tensor_parallel_size=1,
        cpu_per_replica=1,
        gpu_per_replica=0,
        min_replicas=1,
        max_replicas=1,
        proxy_config={
            "upstream_base_url": "http://upstream.local/v1",
            "upstream_model_name": "BAAI/bge-reranker-v2-m3",
            "auth": {"mode": "none"},
        },
    )
    registry = ModelRegistry(ModelCatalogFile(models=[model, embedding, rerank]))
    store = LocalModelStore("models")
    builder = ServeApplicationBuilder(model_store=store)
    dispatcher = RuntimeDispatcher(
        registry=registry,
        serve_builder=builder,
        executor=RuntimeExecutor(),
    )
    return registry, dispatcher


def test_proxy_chat_dispatch_rewrites_only_model_and_passthrough_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, dispatcher = _build_proxy_dispatcher()
    captured: dict[str, object] = {}

    async def fake_request_proxy(self, *, proxy_config, path, payload, request_id):  # type: ignore[no-untyped-def]
        captured["path"] = path
        captured["payload"] = payload
        captured["request_id"] = request_id
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "chatcmpl-proxy",
                "object": "chat.completion",
                "created": 123,
                "model": "opendatalab/MinerU2.5-2509-1.2B",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    monkeypatch.setattr(RuntimeExecutor, "_request_proxy", fake_request_proxy)

    request = ChatCompletionsRequest(
        model="mineru",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
        extra_body={"no_repeat_ngram_size": 16},
    )
    response = asyncio.run(dispatcher.dispatch_chat(registry.get("mineru"), request))

    assert isinstance(response, Response)
    body = json.loads(response.body)
    assert body["model"] == "opendatalab/MinerU2.5-2509-1.2B"
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.headers["X-Infer-Nexus-Request-ID"] == captured["request_id"]
    assert captured["path"] == "/chat/completions"
    assert isinstance(captured["payload"], dict)
    assert captured["payload"]["model"] == "opendatalab/MinerU2.5-2509-1.2B"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["payload"]["temperature"] == 0.2
    assert captured["payload"]["extra_body"] == {"no_repeat_ngram_size": 16}


def test_proxy_embedding_dispatch_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    registry, dispatcher = _build_proxy_dispatcher()

    async def fake_request_proxy(self, *, proxy_config, path, payload, request_id):  # type: ignore[no-untyped-def]
        assert path == "/embeddings"
        assert payload["model"] == "BAAI/bge-large-zh-v1.5"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "BAAI/bge-large-zh-v1.5",
                "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            },
        )

    monkeypatch.setattr(RuntimeExecutor, "_request_proxy", fake_request_proxy)
    request = EmbeddingRequest(model="bge-embedding-proxy", input="hello")
    response = asyncio.run(dispatcher.dispatch_embedding(registry.get("bge-embedding-proxy"), request))
    assert isinstance(response, Response)
    body = json.loads(response.body)
    assert body["model"] == "BAAI/bge-large-zh-v1.5"
    assert body["data"][0]["embedding"] == [0.1, 0.2]


def test_proxy_rerank_dispatch_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    registry, dispatcher = _build_proxy_dispatcher()

    async def fake_request_proxy(self, *, proxy_config, path, payload, request_id):  # type: ignore[no-untyped-def]
        assert path == "/rerank"
        assert payload["model"] == "BAAI/bge-reranker-v2-m3"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "rerank-1",
                "model": "BAAI/bge-reranker-v2-m3",
                "usage": {"total_tokens": 5},
                "results": [{"index": 0, "document": {"text": "hello"}, "relevance_score": 0.9}],
            },
        )

    monkeypatch.setattr(RuntimeExecutor, "_request_proxy", fake_request_proxy)
    request = RerankRequest(model="bge-rerank-proxy", query="hello", documents=["hello"])
    response = asyncio.run(dispatcher.dispatch_rerank(registry.get("bge-rerank-proxy"), request))
    assert isinstance(response, Response)
    body = json.loads(response.body)
    assert body["model"] == "BAAI/bge-reranker-v2-m3"
    assert body["results"][0]["relevance_score"] == 0.9


def test_proxy_upstream_error_is_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    registry, dispatcher = _build_proxy_dispatcher()
    upstream_error = b'{"error":{"message":"bad upstream request","code":"bad_request"}}'

    async def fake_request_proxy(self, *, proxy_config, path, payload, request_id):  # type: ignore[no-untyped-def]
        return httpx.Response(
            400,
            content=upstream_error,
            headers={"content-type": "application/json"},
        )

    monkeypatch.setattr(RuntimeExecutor, "_request_proxy", fake_request_proxy)
    request = ChatCompletionsRequest(model="mineru", messages=[{"role": "user", "content": "hello"}])
    response = asyncio.run(dispatcher.dispatch_chat(registry.get("mineru"), request))

    assert isinstance(response, Response)
    assert response.status_code == 400
    assert response.body == upstream_error
    assert response.headers["content-type"] == "application/json"
