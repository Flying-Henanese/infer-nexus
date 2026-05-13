from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.observability.metrics import MetricsSnapshot


class LoadInspector:
    def snapshot(self, registry: ModelRegistry) -> MetricsSnapshot:
        return MetricsSnapshot(active_models=len(registry.list_models()))
