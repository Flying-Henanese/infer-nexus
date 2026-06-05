"""定义 Ray Serve 部署规格、模型副本和部署参数生成逻辑。"""

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
    """保存单个模型部署到 Ray Serve 时需要的资源和路由配置。"""

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
    """Ray Serve 模型副本，负责校验运行时规格并调用后端适配器。"""

    def __init__(
        self,
        runtime_context: dict[str, Any],
        backend: InferenceBackend | None = None,
    ) -> None:
        """创建模型副本、构造后端适配器，并在副本启动时完成后端初始化。"""
        self.runtime_context = runtime_context
        self.backend = backend or self._build_backend(runtime_context["runtime_spec"]["backend"])
        self.backend.validate_runtime_spec(
            self.runtime_context["runtime_spec"],
            self.runtime_context,
        )
        self.backend.startup()

    def _build_backend(self, backend_name: str) -> InferenceBackend:
        """根据 runtime_spec 中声明的后端名称选择对应的推理后端适配器。"""
        if backend_name == "vllm":
            return VLLMBackend(self.runtime_context["runtime_spec"])
        raise ValueError(f"unsupported backend '{backend_name}'")

    def _resolve_request_payload(
        self,
        request_payload: dict[str, Any] | None = None,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """兼容新旧调用参数名，并保证副本方法一定拿到请求载荷。"""
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
        """校验 chat completion 请求并交给后端执行，统一包装运行时错误。"""
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
        """校验流式 chat completion 请求，并逐块转发后端生成的响应片段。"""
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
        """判断后端返回值是否已经是完整 OpenAI chat 响应，可直接透传。"""
        return payload.get("object") == "chat.completion" and isinstance(payload.get("choices"), list)

    async def embedding(
        self,
        request_payload: dict[str, Any] | None = None,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """校验 embedding 请求并交给后端执行，返回统一成功载荷。"""
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
        """校验 rerank 请求并交给后端执行，返回统一成功载荷。"""
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
        """提供占位 HTTP 入口，实际推理由具名任务方法处理。"""
        # Phase 1 keeps ingress minimal; typed task methods are used for inference dispatch.
        return {
            "status": "not_implemented",
            "message": "runtime request handling is not connected yet",
            "model": self.runtime_context["model_name"],
            "path": getattr(getattr(request, "url", None), "path", None),
        }

    def __del__(self) -> None:
        """副本销毁时尽力关闭后端资源，避免清理异常影响进程退出。"""
        backend = getattr(self, "backend", None)
        if backend is not None:
            try:
                backend.shutdown()
            except Exception:
                pass

class RuntimeApplicationRoot:
    """Ray Serve 根入口，用于把多个模型部署挂在同一个应用下。"""

    def __init__(self, **model_deployments: Any) -> None:
        """保存模型名到部署绑定的映射，供根入口健康检查展示。"""
        self.model_deployments = model_deployments

    async def __call__(self, request: Any | None = None) -> dict[str, Any]:
        """返回运行时根应用状态、请求路径和当前挂载的模型列表。"""
        return {
            "status": "ok",
            "message": "infer-nexus runtime root is active",
            "models": sorted(self.model_deployments.keys()),
            "path": getattr(getattr(request, "url", None), "path", None),
        }

class DeploymentFactory:
    """把模型目录配置转换成 Ray Serve 可消费的部署声明。"""

    def __init__(self, inference_device_type: Literal["cuda", "npu"] = "cuda") -> None:
        """记录推理设备类型，用于后续生成 CUDA 或 NPU 的 Ray 资源配置。"""
        self.inference_device_type = inference_device_type

    def build_deployment_name(self, model: ModelConfig) -> str:
        """为模型生成稳定部署名，保证 Serve 句柄查找和计划输出一致。"""
        return f"model-{model.name}"

    def build_accelerator_actor_options(self, accelerator_per_replica: int | float) -> dict[str, Any]:
        """把每副本加速器数量映射为 Ray actor 的 GPU 或自定义 NPU 资源。"""
        if self.inference_device_type == "cuda":
            return {"num_gpus": accelerator_per_replica}
        if self.inference_device_type == "npu":
            return {"resources": {"NPU": accelerator_per_replica}}
        raise ValueError(f"unsupported inference_device_type '{self.inference_device_type}'")

    def build_spec(self, model: ModelConfig) -> DeploymentSpec:
        """汇总模型资源、伸缩和路由配置，生成完整 DeploymentSpec。"""
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
        """在 Ray Serve 可用时创建 RequestRouterConfig，否则保留普通字典。"""
        if not request_router_config:
            return None
        try:
            from ray.serve.config import RequestRouterConfig
        except ImportError:
            return dict(request_router_config)
        return RequestRouterConfig(**request_router_config)

    def build_serve_deployment_kwargs(self, spec: DeploymentSpec) -> dict[str, Any]:
        """把 DeploymentSpec 转换成 serve.deployment 所需的关键字参数。"""
        kwargs = {
            "name": spec.deployment_name,
            "ray_actor_options": dict(spec.ray_actor_options),
            "autoscaling_config": dict(spec.autoscaling_config),
        }
        request_router_config = self.build_request_router_config(spec.request_router_config)
        if request_router_config is not None:
            kwargs["request_router_config"] = request_router_config
        return kwargs
