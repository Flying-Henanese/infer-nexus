from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from infer_nexus.catalog.models import ModelConfig
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest


class InferenceBackend(ABC):
    @abstractmethod
    def validate_runtime_spec(self, runtime_spec: dict[str, Any], runtime_context: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def startup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_runtime_spec(self, model: ModelConfig, resolved_model_path: Path) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def chat_completion(
        self,
        runtime_spec: dict[str, Any],
        request: ChatCompletionsRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def embedding(
        self,
        runtime_spec: dict[str, Any],
        request: EmbeddingRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def rerank(
        self,
        runtime_spec: dict[str, Any],
        request: RerankRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError
