from abc import ABC, abstractmethod
from typing import Any

from infer_nexus.catalog.models import ModelConfig


class InferenceBackend(ABC):
    @abstractmethod
    def build_runtime_spec(self, model: ModelConfig) -> dict[str, Any]:
        raise NotImplementedError
