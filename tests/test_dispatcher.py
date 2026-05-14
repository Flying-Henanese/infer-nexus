import asyncio

import pytest

from infer_nexus.catalog.loader import load_model_catalog
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.core.errors import RuntimeNotConnectedError
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest
from infer_nexus.model_store import LocalModelStore
from infer_nexus.runtime.dispatcher import RuntimeDispatcher
from infer_nexus.runtime.executor import RuntimeExecutor
from infer_nexus.runtime.handles import ServeDeploymentHandleResolver
from infer_nexus.runtime.serve_app import ServeApplicationBuilder


def make_dispatcher(executor: RuntimeExecutor | None = None) -> tuple[ModelRegistry, LocalModelStore, RuntimeDispatcher]:
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
    registry, _, dispatcher = make_dispatcher()

    target = dispatcher.resolve_target(registry.get('qwen3-chat'))

    assert target.model_name == 'qwen3-32b-instruct'
    assert target.model_alias == 'qwen3-chat'
    assert target.deployment_name == 'model-qwen3-32b-instruct'
    assert target.runtime_context['resolved_model_path'].endswith('/models/Qwen/Qwen3-32B-Instruct')
    assert target.runtime_context['deployment_name'] == 'model-qwen3-32b-instruct'


def test_dispatch_chat_returns_stub_chat_completion() -> None:
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
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def remote(self, request_payload: dict) -> dict:
        result = dict(self.payload)
        result['payload'] = request_payload
        return result


class FakeDeploymentHandle:
    def __init__(self, deployment_name: str) -> None:
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
    def get_deployment_handle(self, deployment_name: str, app_name: str) -> FakeDeploymentHandle:
        assert app_name == 'infer-nexus'
        return FakeDeploymentHandle(deployment_name)


def test_dispatch_chat_uses_serve_handle_in_serve_mode() -> None:
    resolver = ServeDeploymentHandleResolver(app_name='infer-nexus', serve=FakeServe())
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
    resolver = ServeDeploymentHandleResolver(app_name='infer-nexus', serve=FakeServe())
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
    resolver = ServeDeploymentHandleResolver(app_name='infer-nexus', serve=FakeServe())
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
    executor = RuntimeExecutor(mode='serve')
    registry, _, dispatcher = make_dispatcher(executor)
    request = ChatCompletionsRequest(
        model='qwen3-chat',
        messages=[{'role': 'user', 'content': 'hello'}],
    )

    with pytest.raises(RuntimeNotConnectedError, match='no handle resolver'):
        asyncio.run(dispatcher.dispatch_chat(registry.get('qwen3-chat'), request))
