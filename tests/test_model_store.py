"""模型仓库与下载片段构建测试。"""

from pathlib import Path

import pytest

from infer_nexus.artifacts.download import build_model_registration_snippet
from infer_nexus.catalog.models import ModelConfig
from infer_nexus.core.enums import TaskType
from infer_nexus.core.errors import ModelArtifactMissingError
from infer_nexus.model_store import LocalModelStore


def test_local_model_store_resolves_relative_model_path_under_root() -> None:
    """相对模型路径应解析到模型仓库根目录下。"""
    store = LocalModelStore('models')

    resolved = store.resolve_model_path('Qwen/Qwen3-32B-Instruct')

    assert resolved.as_posix().endswith('/models/Qwen/Qwen3-32B-Instruct')


def test_local_model_store_builds_repo_target_dir() -> None:
    """repo id 应映射为分层目录。"""
    store = LocalModelStore('models')

    target_dir = store.build_repo_target_dir('BAAI/bge-reranker-v2-m3')

    assert target_dir.as_posix().endswith('/models/BAAI/bge-reranker-v2-m3')


def test_local_model_store_requires_existing_artifact(tmp_path: Path) -> None:
    """模型目录存在时 require_model_path 应返回解析路径。"""
    store = LocalModelStore(tmp_path)
    model = ModelConfig(
        name='qwen3-32b-instruct',
        alias='qwen3-chat',
        task=TaskType.CHAT,
        model_path='Qwen/Qwen3-32B-Instruct',
        tensor_parallel_size=4,
        cpu_per_replica=8,
        gpu_per_replica=4,
        min_replicas=1,
        max_replicas=2,
    )
    expected_dir = tmp_path / 'Qwen' / 'Qwen3-32B-Instruct'
    expected_dir.mkdir(parents=True)

    resolved_path = store.require_model_path(model)

    assert resolved_path.as_posix().endswith('/Qwen/Qwen3-32B-Instruct')


def test_local_model_store_raises_for_missing_artifact(tmp_path: Path) -> None:
    """模型目录缺失时应抛出 ModelArtifactMissingError。"""
    store = LocalModelStore(tmp_path)
    model = ModelConfig(
        name='qwen3-32b-instruct',
        alias='qwen3-chat',
        task=TaskType.CHAT,
        model_path='Qwen/Qwen3-32B-Instruct',
        tensor_parallel_size=4,
        cpu_per_replica=8,
        gpu_per_replica=4,
        min_replicas=1,
        max_replicas=2,
    )

    with pytest.raises(ModelArtifactMissingError, match='missing from local storage'):
        store.require_model_path(model)


def test_build_model_registration_snippet_uses_model_path_field() -> None:
    """生成的注册片段应包含 model_path、dtype 和 alias 字段。"""
    snippet = build_model_registration_snippet(
        name='qwen3-32b-instruct',
        alias='qwen3-chat',
        task='chat',
        backend='vllm',
        model_path='Qwen/Qwen3-32B-Instruct',
        dtype='bfloat16',
        tensor_parallel_size=4,
        cpu_per_replica=8,
        gpu_per_replica=4,
        min_replicas=1,
        max_replicas=2,
    )

    assert 'model_path: Qwen/Qwen3-32B-Instruct' in snippet
    assert 'dtype: bfloat16' in snippet
    assert 'alias: qwen3-chat' in snippet
