"""后端适配器抽象接口。

本模块定义 infer-nexus 运行时与具体推理引擎之间的稳定边界。
运行时层负责解析模型目录、组装 runtime context、管理部署副本和路由请求；
具体后端实现负责把这些通用输入转换为引擎可执行的配置与调用。

新增后端时应继承 :class:`InferenceBackend`，并实现生命周期、runtime spec
构建、请求校验和各类推理任务方法。调用方只依赖这个抽象接口，因此后端实现
可以替换具体引擎、远程服务或本地执行策略，而不影响运行时调度代码。
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from infer_nexus.catalog.models import ModelConfig
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest


class InferenceBackend(ABC):
    """统一后端生命周期与推理调用协议。

    后端实现需要把 catalog 中的模型声明转换为可执行 runtime spec，并在
    模型副本启动时完成资源初始化。推理请求进入运行时后，会先被解析为核心
    schema 对象，再连同 runtime spec 和 runtime context 传给对应后端方法。
    """

    @abstractmethod
    def validate_runtime_spec(self, runtime_spec: dict[str, Any], runtime_context: dict[str, Any]) -> None:
        """校验 runtime spec 和 runtime context 是否满足当前后端要求。

        该方法在副本启动前调用，用于尽早发现后端类型、任务模式、模型能力、
        引擎初始化模式等配置不一致的问题。校验失败时，具体实现应抛出后端
        配置或请求校验相关异常，而不是静默修正输入。
        """
        raise NotImplementedError

    @abstractmethod
    def startup(self) -> None:
        """启动后端运行所需资源。

        典型实现会在这里创建本地推理引擎、初始化 serving adapter、建立远程
        客户端连接或准备缓存。该方法由运行时副本构造流程调用，应该可以在
        `validate_runtime_spec` 成功后安全执行。
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """关闭或释放后端持有的运行资源。

        典型实现应清理引擎、网络连接、后台任务或临时状态。该方法可能在对象
        析构或运行时停止阶段被调用，因此实现应尽量做到幂等，并避免因重复
        关闭而抛出无意义异常。
        """
        raise NotImplementedError

    @abstractmethod
    def build_runtime_spec(self, model: ModelConfig, resolved_model_reference: str) -> dict[str, Any]:
        """从模型声明构建后端可执行的 runtime spec。

        Args:
            model: catalog 中的模型配置。
            resolved_model_reference: 已解析的模型路径或引用，可能指向本地模型
                目录、模型仓库名称或后端可识别的其他引用格式。

        Returns:
            后端启动和请求处理所需的结构化配置。调用方会把该配置放入
            runtime context，并在后续生命周期和推理方法中传回后端。
        """
        raise NotImplementedError

    @abstractmethod
    async def chat_completion(
        self,
        runtime_spec: dict[str, Any],
        request: ChatCompletionsRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """执行非流式 chat completion 请求。

        Args:
            runtime_spec: `build_runtime_spec` 生成并经过运行时补充的后端配置。
            request: 已通过核心 schema 校验的聊天补全请求。
            runtime_context: 当前模型副本的运行上下文，包含模型名、任务类型、
                served model name、能力声明和部署信息等。

        Returns:
            推理响应字典。实现可以返回 OpenAI 兼容完整响应，也可以返回由
            运行时包装的后端标准响应字段。
        """
        raise NotImplementedError

    @abstractmethod
    def chat_completion_stream(
        self,
        runtime_spec: dict[str, Any],
        request: ChatCompletionsRequest,
        runtime_context: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
        """执行流式 chat completion 请求。

        Args:
            runtime_spec: 当前模型副本使用的后端运行配置。
            request: 已通过核心 schema 校验的聊天补全请求。
            runtime_context: 当前模型副本的运行上下文。

        Returns:
            异步迭代器，逐块产出后端标准事件、OpenAI 兼容 SSE 字节串或文本。
            运行时会把这些 chunk 继续交给 API 层或调用方消费。
        """
        raise NotImplementedError

    @abstractmethod
    async def embedding(
        self,
        runtime_spec: dict[str, Any],
        request: EmbeddingRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """执行 embedding 请求。

        Args:
            runtime_spec: 当前模型副本使用的后端运行配置。
            request: 已通过核心 schema 校验的 embedding 请求。
            runtime_context: 当前模型副本的运行上下文。

        Returns:
            包含向量结果和相关元数据的响应字典。运行时会在需要时追加统一状态
            字段。
        """
        raise NotImplementedError

    @abstractmethod
    async def rerank(
        self,
        runtime_spec: dict[str, Any],
        request: RerankRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """执行 rerank 请求。

        Args:
            runtime_spec: 当前模型副本使用的后端运行配置。
            request: 已通过核心 schema 校验的 rerank 请求。
            runtime_context: 当前模型副本的运行上下文。

        Returns:
            包含重排序分数、排序结果和相关元数据的响应字典。具体字段由后端
            实现和 API 兼容层约定。
        """
        raise NotImplementedError
