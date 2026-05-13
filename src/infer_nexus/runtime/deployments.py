from dataclasses import dataclass, field
from typing import Any

from infer_nexus.catalog.models import ModelConfig


@dataclass(slots=True)
class DeploymentSpec:
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
    """Placeholder Ray Serve replica until a real backend runtime is attached."""

    def __init__(self, runtime_context: dict[str, Any]) -> None:
        self.runtime_context = runtime_context

    async def __call__(self, request: Any) -> dict[str, Any]:
        return {
            "status": "not_implemented",
            "message": "runtime request handling is not connected yet",
            "model": self.runtime_context["model_name"],
            "path": getattr(getattr(request, "url", None), "path", None),
        }


class DeploymentFactory:
    def build_deployment_name(self, model: ModelConfig) -> str:
        return f"model-{model.name}"

    def build_spec(self, model: ModelConfig) -> DeploymentSpec:
        autoscaling_config = {
            "min_replicas": model.min_replicas,
            "max_replicas": model.max_replicas,
        }
        ray_actor_options = {
            "num_cpus": model.cpu_per_replica,
            "num_gpus": model.gpu_per_replica,
        }
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
        return {
            "name": spec.deployment_name,
            "ray_actor_options": dict(spec.ray_actor_options),
            "autoscaling_config": dict(spec.autoscaling_config),
        }
