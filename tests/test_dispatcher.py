"""运行时分发器测试。"""

import asyncio
import json

import httpx
import pytest
from starlette.responses import Response

from infer_nexus.catalog.models import ModelCatalogFile, ModelConfig
from infer_nexus.catalog.loader import load_model_catalog
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.core.enums import BackendType, TaskType
from infer_nexus.core.errors import RuntimeNotConnectedError
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest
from infer_nexus.model_store import LocalModelStore
from infer_nexus.runtime.dispatcher import RuntimeDispatcher
from infer_nexus.runtime.executor import RuntimeExecutor
from infer_nexus.runtime.handles import ServeDeploymentHandleResolver
from infer_nexus.runtime.serve_app import ServeApplicationBuilder


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


class FakeServe:
    """模拟 Serve runtime。"""

    def get_deployment_handle(self, deployment_name: str, app_name: str) -> FakeDeploymentHandle:
        """返回模拟部署句柄。"""
        assert app_name.startswith('infer-nexus-model-')
        return FakeDeploymentHandle(deployment_name)


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
