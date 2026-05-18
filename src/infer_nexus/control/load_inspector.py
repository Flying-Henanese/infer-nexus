"""负载观测入口。"""

from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.observability.metrics import MetricsSnapshot


class LoadInspector:
    """根据注册模型和运行态信息生成负载快照。"""

    def snapshot(self, registry: ModelRegistry) -> MetricsSnapshot:
        """返回当前负载快照。"""
        return MetricsSnapshot(active_models=len(registry.list_models()))
