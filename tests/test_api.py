from pathlib import Path

from fastapi.testclient import TestClient

from infer_nexus.core.errors import AdmissionRejectedError
from infer_nexus.main import create_app


def test_health_and_ready_endpoints(prepared_model_store: Path) -> None:
    app = create_app()

    with TestClient(app) as client:
        health = client.get('/healthz')
        ready = client.get('/readyz')

    assert health.status_code == 200
    assert health.json() == {'status': 'ok', 'service': 'infer-nexus'}
    assert ready.status_code == 200
    assert ready.json() == {'status': 'ok', 'service': 'infer-nexus'}


def test_catalog_and_openai_model_endpoints(prepared_model_store: Path) -> None:
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


def test_catalog_model_lookup_by_alias(prepared_model_store: Path) -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get('/api/catalog/models/qwen3-chat')

    assert response.status_code == 200
    assert response.json()['name'] == 'qwen3-32b-instruct'


def test_catalog_model_lookup_returns_404_for_unknown_model(prepared_model_store: Path) -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get('/api/catalog/models/does-not-exist')

    assert response.status_code == 404
    assert 'does-not-exist' in response.json()['detail']


def test_model_status_reports_present_local_path(prepared_model_store: Path) -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get('/api/models/qwen3-chat/status')

    assert response.status_code == 200
    assert response.json()['status'] == 'unknown'
    assert 'local model path is present' in response.json()['message']


def test_model_status_reports_missing_local_artifact(prepared_model_store: Path) -> None:
    missing_path = prepared_model_store / 'Qwen' / 'Qwen3-32B-Instruct'
    missing_path.rmdir()

    app = create_app()

    with TestClient(app) as client:
        response = client.get('/api/models/qwen3-chat/status')

    assert response.status_code == 200
    assert response.json()['status'] == 'degraded'
    assert 'missing from local storage' in response.json()['message']


def test_model_status_returns_404_for_unknown_model(prepared_model_store: Path) -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get('/api/models/does-not-exist/status')

    assert response.status_code == 404
    assert 'does-not-exist' in response.json()['detail']


def test_chat_completions_returns_stub_chat_completion_for_chat_model(prepared_model_store: Path) -> None:
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


def test_chat_completions_reports_unknown_model(prepared_model_store: Path) -> None:
    app = create_app()

    payload = {
        'model': 'does-not-exist',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 404
    assert response.json()['error']['code'] == 'model_not_found'


def test_chat_completions_reports_missing_local_artifact(prepared_model_store: Path) -> None:
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
    app = create_app()

    payload = {
        'model': 'bge-embedding',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 400
    assert response.json()['error']['code'] == 'unsupported_task_type'


def test_chat_completions_rejects_streaming_in_phase1(prepared_model_store: Path) -> None:
    app = create_app()

    payload = {
        'model': 'qwen3-chat',
        'messages': [{'role': 'user', 'content': 'hello'}],
        'stream': True,
    }

    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 400
    assert response.json()['error']['code'] == 'unsupported_parameter'


def test_chat_completions_rejects_multimodal_message_content(prepared_model_store: Path) -> None:
    app = create_app()

    payload = {
        'model': 'qwen3-chat',
        'messages': [
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'hello'}],
            }
        ],
    }

    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 400
    assert response.json()['error']['code'] == 'unsupported_message_content'


def test_chat_completions_returns_429_when_admission_rejects(prepared_model_store: Path) -> None:
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


def test_chat_completions_returns_501_when_serve_handle_is_unavailable(
    prepared_model_store: Path,
) -> None:
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


def test_embeddings_returns_stub_embedding_response_for_embedding_model(prepared_model_store: Path) -> None:
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
