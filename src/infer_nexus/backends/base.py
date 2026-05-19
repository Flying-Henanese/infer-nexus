"""后端适配器抽象接口。"""

from abc import ABC, abstractmethod
from typing import Any

from infer_nexus.catalog.models import ModelConfig
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest


class InferenceBackend(ABC):
    """统一后端生命周期与推理调用协议。"""

    @abstractmethod
    def validate_runtime_spec(self, runtime_spec: dict[str, Any], runtime_context: dict[str, Any]) -> None:
        """校验运行时规格与上下文是否匹配当前后端要求。"""
        raise NotImplementedError

    @abstractmethod
    def startup(self) -> None:
        """启动后端资源（例如加载引擎、建立连接）。"""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """关闭后端资源。"""
        raise NotImplementedError

    @abstractmethod
    def build_runtime_spec(self, model: ModelConfig, resolved_model_reference: str) -> dict[str, Any]:
        """从模型声明构建后端可执行的 runtime spec。"""
        raise NotImplementedError

    @abstractmethod
    async def chat_completion(
        self,
        runtime_spec: dict[str, Any],
        request: ChatCompletionsRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """执行 chat completion 请求。"""
        raise NotImplementedError

    @abstractmethod
    async def embedding(
        self,
        runtime_spec: dict[str, Any],
        request: EmbeddingRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """执行 embedding 请求。"""
        raise NotImplementedError

    @abstractmethod
    async def rerank(
        self,
        runtime_spec: dict[str, Any],
        request: RerankRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """执行 rerank 请求。"""
        raise NotImplementedError
