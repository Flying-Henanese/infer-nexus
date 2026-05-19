"""Ray Serve deployment building blocks and per-model deployment specs."""

from dataclasses import dataclass, field
from typing import Any

from infer_nexus.backends.base import InferenceBackend
from infer_nexus.backends.vllm import VLLMBackend
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest
from infer_nexus.catalog.models import ModelConfig


@dataclass(slots=True)
class DeploymentSpec:
    """Deployment declaration derived from one catalog model."""

    model_name: str
    model_alias: str | None
    deployment_name: str
    backend: str
    num_cpus: int | float
    num_gpus: int | float
    tensor_parallel_size: int
    max_model_len: int | None
    autoscaling_config: dict[str, int] = field(default_factory=dict)
    ray_actor_options: dict[str, Any] = field(default_factory=dict)


class ModelRuntimeReplica:
    """Serve replica that dispatches task requests into a backend adapter."""

    def __init__(
        self,
        runtime_context: dict[str, Any],
        backend: InferenceBackend | None = None,
    ) -> None:
        """初始化模型副本并启动后端。"""
        self.runtime_context = runtime_context
        self.backend = backend or self._build_backend(runtime_context["runtime_spec"]["backend"])
        self.backend.validate_runtime_spec(
            self.runtime_context["runtime_spec"],
            self.runtime_context,
        )
        self.backend.startup()

    def _build_backend(self, backend_name: str) -> InferenceBackend:
        """Instantiate backend adapter from runtime spec."""
        if backend_name == "vllm":
            return VLLMBackend(self.runtime_context["runtime_spec"])
        raise ValueError(f"unsupported backend '{backend_name}'")

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """处理 chat completion 负载。"""
        request = ChatCompletionsRequest.model_validate(payload)
        response = await self.backend.chat_completion(
            self.runtime_context["runtime_spec"],
            request,
            self.runtime_context,
        )
        return {"status": "ok", **response}

    async def embedding(self, payload: dict[str, Any]) -> dict[str, Any]:
        """处理 embedding 负载。"""
        request = EmbeddingRequest.model_validate(payload)
        response = await self.backend.embedding(
            self.runtime_context["runtime_spec"],
            request,
            self.runtime_context,
        )
        return {"status": "ok", **response}

    async def rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        """处理 rerank 负载。"""
        request = RerankRequest.model_validate(payload)
        response = await self.backend.rerank(
            self.runtime_context["runtime_spec"],
            request,
            self.runtime_context,
        )
        return {"status": "ok", **response}

    async def __call__(self, request: Any) -> dict[str, Any]:
        """Serve HTTP ingress 占位处理。"""
        # Phase 1 keeps ingress minimal; typed task methods are used for inference dispatch.
        return {
            "status": "not_implemented",
            "message": "runtime request handling is not connected yet",
            "model": self.runtime_context["model_name"],
            "path": getattr(getattr(request, "url", None), "path", None),
        }

    def __del__(self) -> None:
        """析构时尝试关闭后端资源。"""
        backend = getattr(self, "backend", None)
        if backend is not None:
            try:
                backend.shutdown()
            except Exception:
                pass


class RuntimeApplicationRoot:
    """Synthetic ingress to keep all model deployments inside one Serve application."""

    def __init__(self, **model_deployments: Any) -> None:
        """初始化根 ingress，持有所有模型部署绑定。"""
        self.model_deployments = model_deployments

    async def __call__(self, request: Any | None = None) -> dict[str, Any]:
        """返回根应用健康状态与已挂载模型列表。"""
        return {
            "status": "ok",
            "message": "infer-nexus runtime root is active",
            "models": sorted(self.model_deployments.keys()),
            "path": getattr(getattr(request, "url", None), "path", None),
        }


class DeploymentFactory:
    """Translate model catalog entries into Ray Serve deployment parameters."""

    def build_deployment_name(self, model: ModelConfig) -> str:
        """Build stable per-model deployment names for one-model-per-deployment topology."""
        return f"model-{model.name}"

    def build_spec(self, model: ModelConfig) -> DeploymentSpec:
        """从模型配置生成部署规格。"""
        # Replica bounds are declarative inputs; Serve owns runtime autoscaling behavior.
        autoscaling_config = {
            "min_replicas": model.min_replicas,
            "max_replicas": model.max_replicas,
        }
        autoscaling_config.update(model.deployment_config.autoscaling_config)
        ray_actor_options = {
            "num_cpus": model.cpu_per_replica,
            "num_gpus": model.gpu_per_replica,
        }
        ray_actor_options.update(model.deployment_config.ray_actor_options)
        return DeploymentSpec(
            model_name=model.name,
            model_alias=model.alias,
            deployment_name=self.build_deployment_name(model),
            backend=model.backend,
            num_cpus=model.cpu_per_replica,
            num_gpus=model.gpu_per_replica,
            tensor_parallel_size=model.tensor_parallel_size,
            max_model_len=model.max_model_len,
            autoscaling_config=autoscaling_config,
            ray_actor_options=ray_actor_options,
        )

    def build_serve_deployment_kwargs(self, spec: DeploymentSpec) -> dict[str, Any]:
        """Produce kwargs passed to `serve.deployment(...)`."""
        return {
            "name": spec.deployment_name,
            "ray_actor_options": dict(spec.ray_actor_options),
            "autoscaling_config": dict(spec.autoscaling_config),
        }
