from dataclasses import asdict
from typing import Any

from infer_nexus.backends.vllm import VLLMBackend
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.runtime.deployments import (
    DeploymentFactory,
    DeploymentSpec,
    ModelRuntimeReplica,
)


class ServeApplicationBuilder:
    def __init__(self, deployment_factory: DeploymentFactory | None = None) -> None:
        self.deployment_factory = deployment_factory or DeploymentFactory()
        self.backend = VLLMBackend()

    def build_specs(self, registry: ModelRegistry) -> list[DeploymentSpec]:
        return [self.deployment_factory.build_spec(model) for model in registry.list_models()]

    def build_plan(self, registry: ModelRegistry) -> dict[str, dict[str, Any]]:
        return {spec.model_name: asdict(spec) for spec in self.build_specs(registry)}

    def build_local_dev_summary(self, registry: ModelRegistry) -> dict[str, Any]:
        specs = self.build_specs(registry)
        return {
            "deployments": [spec.deployment_name for spec in specs],
            "models": [spec.model_alias or spec.model_name for spec in specs],
            "total_declared_gpu_per_minimum_pool": sum(
                spec.num_gpus * spec.autoscaling_config["min_replicas"] for spec in specs
            ),
        }

    def require_ray_serve(self) -> Any:
        try:
            from ray import serve
        except ImportError as exc:
            raise RuntimeError(
                "Ray Serve is not installed. Install the 'serve' extra to enable runtime integration."
            ) from exc
        return serve

    def build_runtime_context(self, registry: ModelRegistry, model_name: str) -> dict[str, Any]:
        model = registry.get(model_name)
        runtime_spec = self.backend.build_runtime_spec(model)
        return {
            "model_name": model.name,
            "model_alias": model.alias,
            "task": model.task,
            "runtime_spec": runtime_spec,
        }

    def build_serve_bindings(
        self,
        registry: ModelRegistry,
        serve: Any | None = None,
        replica_cls: type[ModelRuntimeReplica] = ModelRuntimeReplica,
    ) -> dict[str, Any]:
        serve_runtime = serve or self.require_ray_serve()
        bindings: dict[str, Any] = {}

        for spec in self.build_specs(registry):
            deployment = serve_runtime.deployment(
                **self.deployment_factory.build_serve_deployment_kwargs(spec)
            )(replica_cls)
            runtime_context = self.build_runtime_context(registry, spec.model_name)
            bindings[spec.model_name] = deployment.bind(runtime_context)

        return bindings
