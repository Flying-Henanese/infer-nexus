"""Ray Serve application assembly utilities for infer-nexus runtime."""

from dataclasses import asdict
from typing import Any

from infer_nexus.backends.vllm import VLLMBackend
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.model_store import LocalModelStore
from infer_nexus.runtime.deployments import (
    DeploymentFactory,
    DeploymentSpec,
    ModelRuntimeReplica,
    RuntimeApplicationRoot,
)


class ServeApplicationBuilder:
    """Build Ray Serve deployment/application graph from declarative model registry."""

    def __init__(
        self,
        model_store: LocalModelStore,
        backend_init_mode: str = "stub",
        deployment_factory: DeploymentFactory | None = None,
    ) -> None:
        """初始化 Serve 应用构建器。"""
        self.model_store = model_store
        self.backend_init_mode = backend_init_mode
        self.deployment_factory = deployment_factory or DeploymentFactory()
        self.backend = VLLMBackend({})

    def build_specs(self, registry: ModelRegistry) -> list[DeploymentSpec]:
        """Compile per-model deployment specs from the catalog registry."""
        return [self.deployment_factory.build_spec(model) for model in registry.list_models()]

    def build_plan(self, registry: ModelRegistry) -> dict[str, dict[str, Any]]:
        """Build serializable deployment plan for diagnostics and inspection."""
        return {spec.model_name: asdict(spec) for spec in self.build_specs(registry)}

    def build_local_dev_summary(self, registry: ModelRegistry) -> dict[str, Any]:
        """Summarize declared model/deployment layout for local bring-up checks."""
        specs = self.build_specs(registry)
        return {
            "deployments": [spec.deployment_name for spec in specs],
            "models": [spec.model_alias or spec.model_name for spec in specs],
            "model_store_root": str(self.model_store.root_dir),
            "total_declared_gpu_per_minimum_pool": sum(
                spec.num_gpus * spec.autoscaling_config["min_replicas"] for spec in specs
            ),
        }

    def require_ray_serve(self) -> Any:
        """Import Ray Serve runtime or fail with actionable dependency hint."""
        try:
            from ray import serve
        except ImportError as exc:
            raise RuntimeError(
                "Ray Serve is not installed. Install the 'serve' extra to enable runtime integration."
            ) from exc
        return serve

    def build_runtime_context(self, registry: ModelRegistry, model_name: str) -> dict[str, Any]:
        """Build validated runtime context passed into each model replica deployment."""
        model = registry.get(model_name)
        model_reference = self.model_store.resolve_model_reference(model)
        runtime_spec = self.backend.build_runtime_spec(model, model_reference)
        runtime_spec["backend_init_mode"] = self.backend_init_mode
        runtime_context = {
            "model_name": model.name,
            "model_alias": model.alias,
            "served_model_name": model.served_model_name or model.alias or model.name,
            "task": model.task,
            "capabilities": list(model.capabilities),
            "deployment_name": self.deployment_factory.build_deployment_name(model),
            "resolved_model_path": str(model_reference),
            "runtime_spec": runtime_spec,
        }
        self.backend.validate_runtime_spec(runtime_spec, runtime_context)
        return runtime_context

    def validate_registry_runtime_configs(self, registry: ModelRegistry) -> None:
        """Eagerly validate all model runtime contexts at startup."""
        for model in registry.list_models():
            self.build_runtime_context(registry, model.name)

    def build_serve_bindings(
        self,
        registry: ModelRegistry,
        serve: Any | None = None,
        replica_cls: type[ModelRuntimeReplica] = ModelRuntimeReplica,
    ) -> dict[str, Any]:
        """Create one Serve deployment binding per registered model."""
        serve_runtime = serve or self.require_ray_serve()
        bindings: dict[str, Any] = {}

        for spec in self.build_specs(registry):
            deployment = serve_runtime.deployment(
                **self.deployment_factory.build_serve_deployment_kwargs(spec)
            )(replica_cls)
            runtime_context = self.build_runtime_context(registry, spec.model_name)
            bindings[spec.model_name] = deployment.bind(runtime_context)

        return bindings

    def build_serve_application(
        self,
        registry: ModelRegistry,
        *,
        serve: Any | None = None,
        replica_cls: type[ModelRuntimeReplica] = ModelRuntimeReplica,
        root_cls: type[RuntimeApplicationRoot] = RuntimeApplicationRoot,
        root_name: str = "infer-nexus-root",
    ) -> Any:
        """Assemble a single Serve application that contains all model deployments."""
        serve_runtime = serve or self.require_ray_serve()
        bindings = self.build_serve_bindings(
            registry,
            serve=serve_runtime,
            replica_cls=replica_cls,
        )
        root = serve_runtime.deployment(name=root_name)(root_cls)
        return root.bind(**bindings)
