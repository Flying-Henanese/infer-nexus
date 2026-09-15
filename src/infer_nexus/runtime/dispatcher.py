"""编排 API 请求到运行时目标的分发。"""

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
    """描述运行时组件的数据或行为。"""

    def __init__(
        self,
        registry: ModelRegistry,
        serve_builder: ServeApplicationBuilder,
        executor: RuntimeExecutor,
        model_targets: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """初始化对象并保存运行时依赖。"""
        self.registry = registry
        self.serve_builder = serve_builder
        self.executor = executor
        self._configured_model_targets = (
            {public_name: dict(target) for public_name, target in model_targets.items()}
            if model_targets is not None
            else None
        )
        self._targets = {
            model.name: self._build_target(model)
            for model in registry.list_models()
        }

    def _build_target(self, model: ModelConfig) -> RuntimeTarget:
        """Build one config-derived target during gateway startup."""
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
            if self._configured_model_targets is not None:
                configured_target = self._configured_model_targets.get(model.name)
                if configured_target is None:
                    raise ValueError(
                        f"Model '{model.name}' is not present in the deployed Gateway target table."
                    )
                deployment_name = configured_target["deployment_name"]
                runtime_context = {
                    **runtime_context,
                    "app_name": configured_target["application_name"],
                    "deployment_name": deployment_name,
                }
        return RuntimeTarget(
            model_name=model.name,
            model_alias=model.alias,
            backend=model.backend,
            app_name=runtime_context.get("app_name"),
            deployment_name=deployment_name,
            runtime_context=runtime_context,
        )

    def resolve_target(self, model: ModelConfig) -> RuntimeTarget:
        """Return the precomputed target for a catalog-resolved model."""
        return self._targets[model.name]

    def model_target_table(self) -> dict[str, RuntimeTarget]:
        """Return a copy of the immutable-at-startup model target mapping."""
        return dict(self._targets)

    async def dispatch_chat(
        self,
        model: ModelConfig,
        request: ChatCompletionsRequest,
    ) -> ChatCompletionsResponse | Response:
        """将聊天请求分发到目标执行器。"""
        target = self.resolve_target(model)
        return await self.executor.execute_chat(target=target, request=request)

    async def dispatch_embedding(
        self,
        model: ModelConfig,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse | Response:
        """将向量化请求分发到目标执行器。"""
        target = self.resolve_target(model)
        return await self.executor.execute_embedding(target=target, request=request)

    async def dispatch_rerank(
        self,
        model: ModelConfig,
        request: RerankRequest,
    ) -> RerankResponse | Response:
        """将 rerank 请求分发到目标执行器。"""
        target = self.resolve_target(model)
        return await self.executor.execute_rerank(target=target, request=request)
