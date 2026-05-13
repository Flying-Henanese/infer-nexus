from fastapi.testclient import TestClient

from infer_nexus.main import create_app


def test_health_and_ready_endpoints() -> None:
    app = create_app()

    with TestClient(app) as client:
        health = client.get('/healthz')
        ready = client.get('/readyz')

    assert health.status_code == 200
    assert health.json() == {'status': 'ok', 'service': 'infer-nexus'}
    assert ready.status_code == 200
    assert ready.json() == {'status': 'ok', 'service': 'infer-nexus'}


def test_catalog_and_openai_model_endpoints() -> None:
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


def test_catalog_model_lookup_by_alias() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get('/api/catalog/models/qwen3-chat')

    assert response.status_code == 200
    assert response.json()['name'] == 'qwen3-32b-instruct'


def test_chat_completions_returns_not_connected_error_for_chat_model() -> None:
    app = create_app()

    payload = {
        'model': 'qwen3-chat',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 501
    assert response.json() == {
        'error': {
            'message': 'Chat completions runtime is not connected yet.',
            'type': 'not_implemented_error',
            'param': None,
            'code': 'runtime_not_connected',
        }
    }


def test_chat_completions_rejects_embedding_model() -> None:
    app = create_app()

    payload = {
        'model': 'bge-embedding',
        'messages': [{'role': 'user', 'content': 'hello'}],
    }

    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 400
    assert response.json()['error']['code'] == 'unsupported_task_type'


def test_embeddings_returns_not_connected_error_for_embedding_model() -> None:
    app = create_app()

    payload = {
        'model': 'bge-embedding',
        'input': 'hello',
    }

    with TestClient(app) as client:
        response = client.post('/v1/embeddings', json=payload)

    assert response.status_code == 501
    assert response.json() == {
        'error': {
            'message': 'Embeddings runtime is not connected yet.',
            'type': 'not_implemented_error',
            'param': None,
            'code': 'runtime_not_connected',
        }
    }


def test_embeddings_rejects_chat_model() -> None:
    app = create_app()

    payload = {
        'model': 'qwen3-chat',
        'input': 'hello',
    }

    with TestClient(app) as client:
        response = client.post('/v1/embeddings', json=payload)

    assert response.status_code == 400
    assert response.json()['error']['code'] == 'unsupported_task_type'
