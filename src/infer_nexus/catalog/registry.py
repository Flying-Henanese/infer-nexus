from infer_nexus.catalog.models import ModelCatalogFile, ModelConfig
from infer_nexus.core.errors import ModelNotFoundError


class ModelRegistry:
    def __init__(self, catalog: ModelCatalogFile) -> None:
        self._by_name: dict[str, ModelConfig] = {}
        self._alias_to_name: dict[str, str] = {}

        for model in catalog.models:
            self._by_name[model.name] = model
            if model.alias:
                self._alias_to_name[model.alias] = model.name

    def list_models(self) -> list[ModelConfig]:
        return list(self._by_name.values())

    def get(self, model_name: str) -> ModelConfig:
        canonical = self._alias_to_name.get(model_name, model_name)
        model = self._by_name.get(canonical)
        if model is None:
            raise ModelNotFoundError(f"model not found: {model_name}")
        return model
