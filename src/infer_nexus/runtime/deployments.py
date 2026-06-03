"""Ray Serve deployment building blocks and per-model deployment specs."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from infer_nexus.backends.base import InferenceBackend
from infer_nexus.backends.vllm import VLLMBackend
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest
from infer_nexus.catalog.models import ModelConfig
from infer_nexus.core.errors import BackendConfigurationError, BackendRequestValidationError, RuntimeExecutionError


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
    request_router_config: dict[str, Any] = field(default_factory=dict)


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

    def _resolve_request_payload(
        self,
        request_payload: dict[str, Any] | None = None,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_payload = request_payload if request_payload is not None else payload
        if resolved_payload is None:
            raise TypeError("request_payload is required")
        return resolved_payload

    async def chat_completion(
        self,
        request_payload: dict[str, Any] | None = None,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """处理 chat completion 负载。"""
        request = ChatCompletionsRequest.model_validate(
            self._resolve_request_payload(request_payload, payload=payload)
        )
        try:
            response = await self.backend.chat_completion(
                self.runtime_context["runtime_spec"],
                request,
                self.runtime_context,
            )
        except BackendRequestValidationError as exc:
            raise RuntimeExecutionError(str(exc), code=exc.code) from exc
        except BackendConfigurationError as exc:
            raise RuntimeExecutionError(str(exc), code="backend_misconfigured") from exc
        if self._is_openai_chat_response(response):
            return response
        return {"status": "ok", **response}

    async def chat_completion_stream(
        self,
        request_payload: dict[str, Any] | None = None,
        *,
        payload: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
        """处理 streaming chat completion 负载。"""
        request = ChatCompletionsRequest.model_validate(
            self._resolve_request_payload(request_payload, payload=payload)
        )
        try:
            async for chunk in self.backend.chat_completion_stream(
                self.runtime_context["runtime_spec"],
                request,
                self.runtime_context,
            ):
                yield chunk
        except BackendRequestValidationError as exc:
            raise RuntimeExecutionError(str(exc), code=exc.code) from exc
        except BackendConfigurationError as exc:
            raise RuntimeExecutionError(str(exc), code="backend_misconfigured") from exc

    def _is_openai_chat_response(self, payload: dict[str, Any]) -> bool:
        """Detect full OpenAI chat responses that should pass through unchanged."""
        return payload.get("object") == "chat.completion" and isinstance(payload.get("choices"), list)

    async def embedding(
        self,
        request_payload: dict[str, Any] | None = None,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """处理 embedding 负载。"""
        request = EmbeddingRequest.model_validate(
            self._resolve_request_payload(request_payload, payload=payload)
        )
        try:
            response = await self.backend.embedding(
                self.runtime_context["runtime_spec"],
                request,
                self.runtime_context,
            )
        except BackendRequestValidationError as exc:
            raise RuntimeExecutionError(str(exc), code=exc.code) from exc
        except BackendConfigurationError as exc:
            raise RuntimeExecutionError(str(exc), code="backend_misconfigured") from exc
        return {"status": "ok", **response}

    async def rerank(
        self,
        request_payload: dict[str, Any] | None = None,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """处理 rerank 负载。"""
        request = RerankRequest.model_validate(
            self._resolve_request_payload(request_payload, payload=payload)
        )
        try:
            response = await self.backend.rerank(
                self.runtime_context["runtime_spec"],
                request,
                self.runtime_context,
            )
        except BackendRequestValidationError as exc:
            raise RuntimeExecutionError(str(exc), code=exc.code) from exc
        except BackendConfigurationError as exc:
            raise RuntimeExecutionError(str(exc), code="backend_misconfigured") from exc
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

    def __init__(self, inference_device_type: Literal["cuda", "npu"] = "cuda") -> None:
        """Initialize deployment resource mapping for the configured inference device."""
        self.inference_device_type = inference_device_type

    def build_deployment_name(self, model: ModelConfig) -> str:
        """Build stable per-model deployment names for one-model-per-deployment topology."""
        return f"model-{model.name}"

    def build_accelerator_actor_options(self, accelerator_per_replica: int | float) -> dict[str, Any]:
        """Map logical per-replica accelerator demand to Ray actor options."""
        if self.inference_device_type == "cuda":
            return {"num_gpus": accelerator_per_replica}
        if self.inference_device_type == "npu":
            return {"resources": {"NPU": accelerator_per_replica}}
        raise ValueError(f"unsupported inference_device_type '{self.inference_device_type}'")

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
        }
        ray_actor_options.update(self.build_accelerator_actor_options(model.gpu_per_replica))
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
            request_router_config=dict(model.deployment_config.request_router_config),
        )

    def build_request_router_config(self, request_router_config: dict[str, Any]) -> Any:
        """Materialize request router config when Ray Serve is available."""
        if not request_router_config:
            return None
        try:
            from ray.serve.config import RequestRouterConfig
        except ImportError:
            return dict(request_router_config)
        return RequestRouterConfig(**request_router_config)

    def build_serve_deployment_kwargs(self, spec: DeploymentSpec) -> dict[str, Any]:
        """Produce kwargs passed to `serve.deployment(...)`."""
        kwargs = {
            "name": spec.deployment_name,
            "ray_actor_options": dict(spec.ray_actor_options),
            "autoscaling_config": dict(spec.autoscaling_config),
        }
        request_router_config = self.build_request_router_config(spec.request_router_config)
        if request_router_config is not None:
            kwargs["request_router_config"] = request_router_config
        return kwargs
