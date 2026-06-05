"""内存模型目录索引和查询工具。

注册表会把校验后的 ``ModelCatalogFile`` 转换成快速查询映射，供 API 路由、运行时
分发和平台模型发现接口使用。它支持把规范模型名、别名和 served model name 解析
到同一个 ``ModelConfig`` 对象。
"""

from infer_nexus.catalog.models import ModelCatalogFile, ModelConfig
from infer_nexus.core.errors import ModelNotFoundError


class ModelRegistry:
    """按规范名称和别名查询模型的内存注册表。

    注册表不会修改模型定义，只为已加载的目录建立索引，方便调用方把请求里的模型
    字符串解析为对应的 ``ModelConfig``。
    """

    def __init__(self, catalog: ModelCatalogFile) -> None:
        """根据模型目录构建规范名称和别名索引。

        Args:
            catalog: 已校验的模型目录，其中的模型会被注册到索引中。

        Notes:
            ``alias`` 和 ``served_model_name`` 都会作为模型规范 ``name`` 的查询键。
            如果存在重复别名，后加载的目录条目会覆盖先前的别名映射。
        """
        self._by_name: dict[str, ModelConfig] = {}
        self._alias_to_name: dict[str, str] = {}

        for model in catalog.models:
            self._by_name[model.name] = model
            if model.alias:
                self._alias_to_name[model.alias] = model.name
            if model.served_model_name:
                self._alias_to_name[model.served_model_name] = model.name

    def list_models(self) -> list[ModelConfig]:
        """按目录加载顺序返回所有已注册的模型配置。"""
        return list(self._by_name.values())

    def get(self, model_name: str) -> ModelConfig:
        """把规范名称或别名解析为模型配置。

        Args:
            model_name: 客户端或内部调用方提供的模型标识，可以是规范 ``name``、
                ``alias`` 或 ``served_model_name``。

        Raises:
            ModelNotFoundError: 当没有任何已注册模型匹配 ``model_name`` 时抛出。

        Returns:
            匹配到的 ``ModelConfig`` 实例。
        """
        canonical = self._alias_to_name.get(model_name, model_name)
        model = self._by_name.get(canonical)
        if model is None:
            raise ModelNotFoundError(f"model not found: {model_name}")
        return model
