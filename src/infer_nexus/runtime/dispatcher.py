"""Runtime dispatch orchestration from API-facing model selection to execution target."""

from starlette.responses import Response

from infer_nexus.catalog.models import ModelConfig
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.core.enums import BackendType
from infer_nexus.core.schemas import (
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
)
from infer_nexus.runtime.executor import RuntimeExecutor
from infer_nexus.runtime.serve_app import ServeApplicationBuilder
from infer_nexus.runtime.types import RuntimeTarget


class RuntimeDispatcher:
    """Resolve model runtime targets and delegate execution to RuntimeExecutor."""

    def __init__(
        self,
        registry: ModelRegistry,
        serve_builder: ServeApplicationBuilder,
        executor: RuntimeExecutor,
    ) -> None:
        """初始化分发器，绑定注册表、Serve 构建器与执行器。"""
        self.registry = registry
        self.serve_builder = serve_builder
        self.executor = executor

    def resolve_target(self, model: ModelConfig) -> RuntimeTarget:
        """Build deployment target metadata from catalog + runtime builder."""
        # 核心职责：把“目录模型声明”转换成“可执行目标”，包含部署名和运行时上下文。
        deployment_name = self.serve_builder.deployment_factory.build_deployment_name(model)
        if model.backend == BackendType.VLLM_OPENAI_PROXY:
            runtime_context = {
                "model_name": model.name,
                "model_alias": model.alias,
                "served_model_name": model.served_model_name or model.alias or model.name,
                "task": model.task,
                "deployment_name": deployment_name,
                "backend": model.backend,
                "compat_mode": model.compat_mode.value,
                "proxy_config": model.proxy_config.model_dump(mode="json") if model.proxy_config else {},
            }
        else:
            runtime_context = self.serve_builder.build_runtime_context(self.registry, model.name)
        return RuntimeTarget(
            model_name=model.name,
            model_alias=model.alias,
            backend=model.backend,
            app_name=runtime_context.get("app_name"),
            deployment_name=deployment_name,
            runtime_context=runtime_context,
        )

    async def dispatch_chat(
        self,
        model: ModelConfig,
        request: ChatCompletionsRequest,
    ) -> ChatCompletionsResponse | Response:
        """分发 chat 请求到目标执行器。"""
        target = self.resolve_target(model)
        return await self.executor.execute_chat(target=target, request=request)

    async def dispatch_embedding(
        self,
        model: ModelConfig,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse | Response:
        """分发 embedding 请求到目标执行器。"""
        target = self.resolve_target(model)
        return await self.executor.execute_embedding(target=target, request=request)

    async def dispatch_rerank(
        self,
        model: ModelConfig,
        request: RerankRequest,
    ) -> RerankResponse | Response:
        """分发 rerank 请求到目标执行器。"""
        target = self.resolve_target(model)
        return await self.executor.execute_rerank(target=target, request=request)
