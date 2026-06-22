"""API 路由集成测试。"""

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from infer_nexus.core.errors import AdmissionRejectedError, RuntimeExecutionError, RuntimeNotConnectedError
from infer_nexus.main import create_app
from infer_nexus.observability.metrics import render_prometheus_metrics
from infer_nexus.runtime.executor import RuntimeExecutor


def test_health_and_ready_endpoints(prepared_model_store: Path) -> None:
    """健康与就绪接口应返回 200 和标准响应体。"""
    app = create_app()

    with TestClient(app) as client:
        health = client.get('/healthz')
        ready = client.get('/readyz')

    assert health.status_code == 200
    assert health.json() == {'status': 'ok', 'service': 'infer-nexus'}
    assert ready.status_code == 200
    assert ready.json() == {'status': 'ok', 'service': 'infer-nexus'}


def test_catalog_and_openai_model_endpoints(prepared_model_store: Path) -> None:
    """模型列表与目录接口应返回预期模型集合。"""
    app = create_app()

    with TestClient(app) as client:
        models = client.get('/v1/models')
        catalog = client.get('/api/catalog/models')
        load = client.get('/api/cluster/load')

    assert models.status_code == 200
    assert catalog.status_code == 200
    assert load.status_code == 200

    model_ids = [item['id'] for item in models.json()['data']]
    catalog_names = [item['name'] for item in catalog.json()]

    assert model_ids == ['qwen3-chat', 'bge-embedding', 'bge-rerank']
    assert catalog_names == [
        'qwen3-32b-instruct',
        'bge-large-zh-v1_5',
        'bge-reranker-v2-m3',
    ]
    assert load.json()['active_models'] == 3


def test_metrics_endpoint_exposes_prometheus_text(prepared_model_store: Path) -> None:
    """metrics 接口应暴露 Prometheus 文本格式指标。"""
    app = create_app()

    with TestClient(app) as client:
        response = client.get('/metrics')

    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/plain')
    assert 'infer_nexus_requests_total' in response.text
    assert 'infer_nexus_stream_tpot_seconds' in response.text
    assert 'infer_nexus_stream_completions_total' in response.text
    assert 'infer_nexus_request_queue_seconds' in response.text
    assert 'infer_nexus_serve_handle_calls_total' in response.text
    assert 'infer_nexus_runtime_guard_rejections_total' in response.text


def test_catalog_model_lookup_by_alias(prepared_model_store: Path) -> None:
    """目录接口应支持通过 alias 查询模型。"""
    app = create_app()

    with TestClient(app) as client:
        response = client.get('/api/catalog/models/qwen3-chat')

    assert response.status_code == 200
    assert response.json()['name'] == 'qwen3-32b-instruct'


def test_catalog_model_lookup_returns_404_for_unknown_model(prepared_model_store: Path) -> None:
    """未知模型查询应返回 404。"""
    app = create_app()

    with TestClient(app) as client:
        response = client.get('/api/catalog/models/does-not-exist')

    assert response.status_code == 404
    assert 'does-not-exist' in response.json()['detail']


def test_model_status_reports_present_local_path(prepared_model_store: Path) -> None:
    """模型文件存在时状态接口应返回 unknown 且包含路径提示。"""
    app = create_app()

    with TestClient(app) as client:
        response = client.get('/api/models/qwen3-chat/status')

    assert response.status_code == 200
    assert response.json()['status'] == 'unknown'
    assert 'local model path is present' in response.json()['message']


def test_model_status_reports_missing_local_artifact(prepared_model_store: Path) -> None:
    """模型文件缺失时状态接口应返回 degraded。"""
    missing_path = prepared_model_store / 'Qwen' / 'Qwen3-32B-Instruct'
    missing_path.rmdir()

    app = create_app()

    with TestClient(app) as client:
        response = client.get('/api/models/qwen3-chat/status')

    assert response.status_code == 200
    assert response.json()['status'] == 'degraded'
    assert 'missing from local storage' in response.json()['message']


def test_model_status_returns_404_for_unknown_model(prepared_model_store: Path) -> None:
    """未知模型状态查询应返回 404。"""
    app = create_app()

    with TestClient(app) as client:
        response = client.get('/api/models/does-not-exist/status')

    assert response.status_code == 404
    assert 'does-not-exist' in response.json()['detail']


def test_chat_completions_returns_stub_chat_completion_for_chat_model(prepared_model_store: Path) -> None:
    """chat 接口应为 chat 模型返回 stub completion。"""
    app = create_app()

    payload = {
        'model': 'qwen3-chat',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body['object'] == 'chat.completion'
    assert body['model'] == 'qwen3-chat'
    assert body['choices'][0]['message']['role'] == 'assistant'
    assert 'backend stub response from vllm' in body['choices'][0]['message']['content']
    assert 'model-qwen3-32b-instruct' in body['choices'][0]['message']['content']


def test_chat_completions_updates_gateway_metrics(prepared_model_store: Path) -> None:
    """成功 chat 请求应更新网关 Prometheus 指标。"""
    app = create_app()

    payload = {
        'model': 'qwen3-chat',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=payload)
        metrics = client.get('/metrics')

    assert response.status_code == 200
    assert metrics.status_code == 200
    assert 'infer_nexus_requests_total{' in metrics.text
    assert 'endpoint="/v1/chat/completions"' in metrics.text
    assert 'model="qwen3-chat"' in metrics.text
    assert 'status="success"' in metrics.text
    assert 'task="chat"' in metrics.text
    assert 'infer_nexus_input_tokens_total{' in metrics.text


def test_chat_completions_reports_unknown_model(prepared_model_store: Path) -> None:
    """chat 接口请求未知模型应返回 model_not_found。"""
    app = create_app()

    payload = {
        'model': 'does-not-exist',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 404
    assert response.json()['error']['code'] == 'model_not_found'


def test_chat_completions_error_updates_gateway_metrics(prepared_model_store: Path) -> None:
    """失败请求应记录稳定错误码且未知模型不使用动态标签。"""
    app = create_app()

    payload = {
        'model': 'does-not-exist',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=payload)
        metrics = client.get('/metrics')

    assert response.status_code == 404
    assert 'infer_nexus_errors_total{' in metrics.text
    assert 'code="model_not_found"' in metrics.text
    assert 'model="unknown"' in metrics.text
    assert 'task="chat"' in metrics.text


def test_chat_completions_reports_missing_local_artifact(prepared_model_store: Path) -> None:
    """chat 接口在模型文件缺失时应返回 503。"""
    missing_path = prepared_model_store / 'Qwen' / 'Qwen3-32B-Instruct'
    missing_path.rmdir()

    app = create_app()

    payload = {
        'model': 'qwen3-chat',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 503
    assert response.json()['error']['code'] == 'model_artifact_missing'
    assert 'missing from local storage' in response.json()['error']['message']


def test_chat_completions_rejects_embedding_model(prepared_model_store: Path) -> None:
    """chat 接口不应接受 embedding 模型。"""
    app = create_app()

    payload = {
        'model': 'bge-embedding',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 400
    assert response.json()['error']['code'] == 'unsupported_task_type'


def test_chat_completions_returns_sse_stream_for_chat_model(prepared_model_store: Path) -> None:
    """chat 接口应支持 OpenAI-style SSE 流式响应。"""
    app = create_app()

    payload = {
        'model': 'qwen3-8b',
        'messages': [{'role': 'user', 'content': 'hello'}],
        'stream': True,
    }

    with TestClient(app) as client:
        client.app.state.model_store.require_model_path = lambda _model: None
        with client.stream('POST', '/v1/chat/completions', json=payload) as response:
            body = b''.join(response.iter_bytes())
            headers = dict(response.headers)
            status_code = response.status_code

    assert status_code == 200
    assert headers['content-type'].startswith('text/event-stream')
    assert b'chat.completion.chunk' in body
    assert b'backend stub response from vllm' in body
    assert b'data: [DONE]\n\n' in body


def test_streaming_chat_updates_stream_metrics(prepared_model_store: Path) -> None:
    """流式输出包装器应记录 TTFT、TPOT 和完成状态指标。"""
    executor = RuntimeExecutor()

    async def chunks():
        yield b"data: first\n\n"
        yield b"data: second\n\n"

    async def collect() -> None:
        async for _chunk in executor._observe_stream_chunks(
            chunks(),
            model_label="qwen3-32b",
            stream_start_time=0.0,
        ):
            pass

    asyncio.run(collect())
    metrics = render_prometheus_metrics()[0].decode("utf-8")

    assert 'infer_nexus_stream_ttft_seconds_count{model="qwen3-32b"}' in metrics
    assert 'infer_nexus_stream_tpot_seconds_count{model="qwen3-32b"}' in metrics
    assert 'infer_nexus_stream_completions_total{model="qwen3-32b",status="success"}' in metrics


def test_streaming_chat_records_error_completion_status(prepared_model_store: Path) -> None:
    """流式迭代异常应记录 error 终止状态。"""
    executor = RuntimeExecutor()

    async def chunks():
        yield b"data: first\n\n"
        raise RuntimeError("stream failed")

    async def collect() -> None:
        async for _chunk in executor._observe_stream_chunks(
            chunks(),
            model_label="qwen3-32b-error",
            stream_start_time=0.0,
        ):
            pass

    try:
        asyncio.run(collect())
    except RuntimeError as exc:
        assert str(exc) == "stream failed"
    else:
        raise AssertionError("stream error was not propagated")

    metrics = render_prometheus_metrics()[0].decode("utf-8")

    assert 'infer_nexus_stream_ttft_seconds_count{model="qwen3-32b-error"}' in metrics
    assert 'infer_nexus_stream_completions_total{model="qwen3-32b-error",status="error"}' in metrics


def test_chat_completions_accepts_multimodal_message_content_for_vision_model(
    prepared_model_store: Path,
) -> None:
    """Vision-capable chat models should accept multimodal message content."""
    app = create_app()

    payload = {
        'model': 'qwen3-vl-chat-8b-instruct',
        'messages': [
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
    }

    with TestClient(app) as client:
        client.app.state.model_store.require_model_path = lambda _model: None
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 200
    assert response.json()['choices'][0]['message']['role'] == 'assistant'


def test_chat_completions_rejects_multimodal_message_content_for_text_only_model(
    prepared_model_store: Path,
) -> None:
    """Text-only chat models should reject multimodal message content."""
    app = create_app()

    payload = {
        'model': 'qwen3-8b',
        'messages': [
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
    }

    with TestClient(app) as client:
        client.app.state.model_store.require_model_path = lambda _model: None
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 400
    assert response.json()['error']['code'] == 'unsupported_message_content'


def test_chat_completions_returns_429_when_admission_rejects(prepared_model_store: Path) -> None:
    """准入控制拒绝时 chat 接口应返回 429。"""
    app = create_app()
    payload = {
        'model': 'qwen3-chat',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        def reject(_model):
            raise AdmissionRejectedError('overloaded', code='queue_full')

        client.app.state.admission.check_model_request = reject
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 429
    assert response.json()['error']['code'] == 'queue_full'


def test_chat_completions_returns_429_when_gateway_admission_rejects(
    prepared_model_store: Path,
) -> None:
    """Gateway-local runtime admission rejects should map to rate limiting."""
    app = create_app()
    payload = {
        'model': 'qwen3-chat',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        async def reject_gateway_overload(*, target, request):
            raise AdmissionRejectedError('gateway overloaded', code='gateway_overloaded')

        client.app.state.runtime_dispatcher.executor.execute_chat = reject_gateway_overload
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 429
    assert response.json()['error']['type'] == 'rate_limit_error'
    assert response.json()['error']['code'] == 'gateway_overloaded'


def test_chat_completions_returns_501_when_serve_handle_is_unavailable(
    prepared_model_store: Path,
) -> None:
    """serve 句柄不可用时 chat 接口应返回 501。"""
    app = create_app()
    payload = {
        'model': 'qwen3-chat',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        client.app.state.runtime_executor.mode = 'serve'
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 501
    assert response.json()['error']['code'] == 'runtime_not_connected'


def test_chat_completions_maps_proxy_upstream_timeout_to_504(
    prepared_model_store: Path,
) -> None:
    """Gateway-stage upstream timeouts should use a stable 504 error response."""
    app = create_app()
    payload = {
        'model': 'qwen3-chat',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        async def raise_upstream_timeout(*, target, request):
            raise RuntimeNotConnectedError('upstream timed out', code='upstream_timeout')

        client.app.state.runtime_dispatcher.executor.execute_chat = raise_upstream_timeout
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 504
    assert response.json()['error']['type'] == 'service_unavailable_error'
    assert response.json()['error']['code'] == 'upstream_timeout'


def test_chat_completions_returns_500_when_serve_execution_fails(
    prepared_model_store: Path,
) -> None:
    """serve 远端执行失败时 chat 接口应返回 500 而不是 501。"""
    app = create_app()
    payload = {
        'model': 'qwen3-chat',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        async def raise_runtime_execution_error(*, target, request):
            raise RuntimeExecutionError(
                f"Serve execution failed for deployment '{target.deployment_name}' in app '{target.app_name}'."
            )

        client.app.state.runtime_dispatcher.executor.execute_chat = raise_runtime_execution_error
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 500
    assert response.json()['error']['code'] == 'runtime_execution_failed'


def test_chat_completions_returns_400_when_runtime_validation_fails(
    prepared_model_store: Path,
    monkeypatch,
) -> None:
    """运行时参数校验错误应映射为 400 invalid_request_error。"""
    app = create_app()
    payload = {
        'model': 'qwen3-8b',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        async def raise_runtime_validation_error(*, target, request):
            raise RuntimeExecutionError(
                'This model does not allow reasoning request fields.',
                code='unsupported_parameter',
            )

        client.app.state.model_store.require_model_path = lambda _model: None
        monkeypatch.setattr(
            RuntimeExecutor,
            'execute_chat',
            raise_runtime_validation_error,
        )
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 400
    assert response.json()['error']['type'] == 'invalid_request_error'
    assert response.json()['error']['code'] == 'unsupported_parameter'


def test_embeddings_returns_stub_embedding_response_for_embedding_model(prepared_model_store: Path) -> None:
    """embedding 接口应返回 stub 向量响应。"""
    app = create_app()

    payload = {
        'model': 'bge-embedding',
        'input': 'hello',
    }

    with TestClient(app) as client:
        response = client.post('/v1/embeddings', json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body['object'] == 'list'
    assert body['model'] == 'bge-embedding'
    assert body['data'][0]['embedding'] == [5.0, 0.0, 4.0]


def test_embeddings_returns_base64_embedding_when_requested(prepared_model_store: Path) -> None:
    """请求 base64 编码时 embedding 输出应为字符串。"""
    app = create_app()

    payload = {
        'model': 'bge-embedding',
        'input': 'hello',
        'encoding_format': 'base64',
    }

    with TestClient(app) as client:
        response = client.post('/v1/embeddings', json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body['object'] == 'list'
    assert body['model'] == 'bge-embedding'
    assert isinstance(body['data'][0]['embedding'], str)


def test_embeddings_report_unknown_model(prepared_model_store: Path) -> None:
    """embedding 接口请求未知模型应返回 model_not_found。"""
    app = create_app()

    payload = {
        'model': 'does-not-exist',
        'input': 'hello',
    }

    with TestClient(app) as client:
        response = client.post('/v1/embeddings', json=payload)

    assert response.status_code == 404
    assert response.json()['error']['code'] == 'model_not_found'


def test_embeddings_reports_missing_local_artifact(prepared_model_store: Path) -> None:
    """embedding 接口在模型文件缺失时应返回 503。"""
    missing_path = prepared_model_store / 'BAAI' / 'bge-large-zh-v1.5'
    missing_path.rmdir()

    app = create_app()

    payload = {
        'model': 'bge-embedding',
        'input': 'hello',
    }

    with TestClient(app) as client:
        response = client.post('/v1/embeddings', json=payload)

    assert response.status_code == 503
    assert response.json()['error']['code'] == 'model_artifact_missing'
    assert 'missing from local storage' in response.json()['error']['message']


def test_embeddings_rejects_chat_model(prepared_model_store: Path) -> None:
    """embedding 接口不应接受 chat 模型。"""
    app = create_app()

    payload = {
        'model': 'qwen3-chat',
        'input': 'hello',
    }

    with TestClient(app) as client:
        response = client.post('/v1/embeddings', json=payload)

    assert response.status_code == 400
    assert response.json()['error']['code'] == 'unsupported_task_type'


def test_embeddings_reject_empty_input(prepared_model_store: Path) -> None:
    """embedding 接口应拒绝空输入。"""
    app = create_app()

    payload = {
        'model': 'bge-embedding',
        'input': [],
    }

    with TestClient(app) as client:
        response = client.post('/v1/embeddings', json=payload)

    assert response.status_code == 400
    assert response.json()['error']['code'] == 'invalid_input'


def test_rerank_returns_stub_rerank_response_for_rerank_model(prepared_model_store: Path) -> None:
    """rerank 接口应返回 stub 排序结果。"""
    app = create_app()

    payload = {
        'model': 'bge-rerank',
        'query': 'capital of france',
        'documents': [
            'The capital of Brazil is Brasilia.',
            'The capital of France is Paris.',
            'Python is a programming language.',
        ],
        'top_n': 2,
    }

    with TestClient(app) as client:
        response = client.post('/v1/rerank', json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body['model'] == 'bge-rerank'
    assert len(body['results']) == 2
    assert body['results'][0]['index'] == 1
    assert body['results'][0]['document']['text'] == 'The capital of France is Paris.'


def test_rerank_root_path_is_compatible(prepared_model_store: Path) -> None:
    """兼容路径 rerank 入口应可正常返回结果。"""
    app = create_app()

    payload = {
        'model': 'bge-rerank',
        'query': 'capital of france',
        'documents': [
            'The capital of Brazil is Brasilia.',
            'The capital of France is Paris.',
        ],
    }

    with TestClient(app) as client:
        response = client.post('/rerank', json=payload)

    assert response.status_code == 200
    assert response.json()['model'] == 'bge-rerank'


def test_rerank_reports_unknown_model(prepared_model_store: Path) -> None:
    """rerank 接口请求未知模型应返回 model_not_found。"""
    app = create_app()

    payload = {
        'model': 'does-not-exist',
        'query': 'capital of france',
        'documents': ['The capital of France is Paris.'],
    }

    with TestClient(app) as client:
        response = client.post('/v1/rerank', json=payload)

    assert response.status_code == 404
    assert response.json()['error']['code'] == 'model_not_found'


def test_rerank_reports_missing_local_artifact(prepared_model_store: Path) -> None:
    """rerank 接口在模型文件缺失时应返回 503。"""
    missing_path = prepared_model_store / 'BAAI' / 'bge-reranker-v2-m3'
    missing_path.rmdir()

    app = create_app()

    payload = {
        'model': 'bge-rerank',
        'query': 'capital of france',
        'documents': ['The capital of France is Paris.'],
    }

    with TestClient(app) as client:
        response = client.post('/v1/rerank', json=payload)

    assert response.status_code == 503
    assert response.json()['error']['code'] == 'model_artifact_missing'


def test_rerank_rejects_embedding_model(prepared_model_store: Path) -> None:
    """rerank 接口不应接受 embedding 模型。"""
    app = create_app()

    payload = {
        'model': 'bge-embedding',
        'query': 'capital of france',
        'documents': ['The capital of France is Paris.'],
    }

    with TestClient(app) as client:
        response = client.post('/v1/rerank', json=payload)

    assert response.status_code == 400
    assert response.json()['error']['code'] == 'unsupported_task_type'


def test_rerank_rejects_empty_documents(prepared_model_store: Path) -> None:
    """rerank 接口应拒绝空文档列表。"""
    app = create_app()

    payload = {
        'model': 'bge-rerank',
        'query': 'capital of france',
        'documents': [],
    }

    with TestClient(app) as client:
        response = client.post('/v1/rerank', json=payload)

    assert response.status_code == 400
    assert response.json()['error']['code'] == 'invalid_input'
