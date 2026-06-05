"""模型目录文件加载工具。

本模块负责从磁盘读取 YAML 模型目录，并使用 ``infer_nexus.catalog.models`` 中的
Pydantic 结构进行校验。调用方会获得类型化的 ``ModelCatalogFile``，下游运行时
和路由代码可以依赖其中已标准化的配置值。
"""

from pathlib import Path

import yaml

from infer_nexus.catalog.models import ModelCatalogFile
from infer_nexus.core.errors import ConfigError


def load_model_catalog(path: str | Path) -> ModelCatalogFile:
    """加载 YAML 模型目录文件，并返回校验后的类型化对象。

    Args:
        path: YAML 模型目录文件的路径。

    Raises:
        ConfigError: 当目录文件不存在时抛出。
        pydantic.ValidationError: 当 YAML 内容不符合目录结构，或违反模型级校验
            规则时抛出。

    Returns:
        包含类型化模型定义的 ``ModelCatalogFile``。
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"model catalog file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return ModelCatalogFile.model_validate(raw)
