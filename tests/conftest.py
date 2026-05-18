"""测试共享 fixture。"""

from pathlib import Path

import pytest

from infer_nexus.core.config import Settings, load_settings


@pytest.fixture
def configured_model_paths() -> list[str]:
    """返回测试用模型相对路径列表。"""
    settings = load_settings()
    return [
        'Qwen/Qwen3-32B-Instruct',
        'BAAI/bge-large-zh-v1.5',
        'BAAI/bge-reranker-v2-m3',
    ]


@pytest.fixture
def prepared_model_store(configured_model_paths: list[str]) -> Path:
    """在本地创建测试模型目录并返回绝对路径。"""
    settings = load_settings()
    root = Path(settings.model_store.root_dir)
    root.mkdir(parents=True, exist_ok=True)
    for relative_path in configured_model_paths:
        (root / relative_path).mkdir(parents=True, exist_ok=True)
    return root.resolve()
