"""组装 infer-nexus 运行时的 Ray Serve 应用。"""

from dataclasses import asdict
from typing import Any

from infer_nexus.backends.vllm import VLLMBackend
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.core.config import Settings
from infer_nexus.core.enums import BackendType
from infer_nexus.model_store import LocalModelStore
from infer_nexus.runtime.deployments import (
    DeploymentFactory,
    DeploymentSpec,
    ModelRuntimeReplica,
)
from infer_nexus.runtime.gateway_ingress import (
    GatewayIngressSpec,
    build_gateway_ingress_binding,
    build_gateway_ingress_spec,
)

class ServeApplicationBuilder:
    """描述运行时组件的数据或行为。"""

    def __init__(
        self,
        model_store: LocalModelStore,
        backend_init_mode: str = "stub",
        service_name: str = "infer-nexus",
        deployment_factory: DeploymentFactory | None = None,
    ) -> None:
        """初始化对象并保存运行时依赖。"""
        self.model_store = model_store
        self.backend_init_mode = backend_init_mode
        self.service_name = service_name
        self.deployment_factory = deployment_factory or DeploymentFactory()
        self.backend = VLLMBackend({})

    def build_specs(self, registry: ModelRegistry) -> list[DeploymentSpec]:
        """执行运行时相关逻辑。"""
        return [
            self.deployment_factory.build_spec(model)
            for model in registry.list_models()
            if model.backend == BackendType.VLLM
        ]

    def build_plan(self, registry: ModelRegistry) -> dict[str, dict[str, Any]]:
        """执行运行时相关逻辑。"""
        return {spec.model_name: asdict(spec) for spec in self.build_specs(registry)}

    def build_local_dev_summary(self, registry: ModelRegistry) -> dict[str, Any]:
        """执行运行时相关逻辑。"""
        specs = self.build_specs(registry)
        return {
            "applications": [self.build_application_name(spec.model_name) for spec in specs],
            "deployments": [spec.deployment_name for spec in specs],
            "models": [spec.model_alias or spec.model_name for spec in specs],
            "model_store_root": str(self.model_store.root_dir),
            "total_declared_gpu_per_minimum_pool": sum(
                spec.num_gpus * spec.autoscaling_config["min_replicas"] for spec in specs
            ),
        }

    def build_application_name(self, model_name: str) -> str:
        """执行运行时相关逻辑。"""
        return f"{self.service_name}-model-{model_name}"

    def build_gateway_spec(
        self,
        registry: ModelRegistry,
        *,
        settings: Settings,
    ) -> GatewayIngressSpec:
        """Build the public ingress from the pre-registered model catalog only."""
        model_targets: dict[str, dict[str, str]] = {}
        for model in registry.list_models():
            if model.backend != BackendType.VLLM:
                continue
            target = {
                "application_name": self.build_application_name(model.name),
                "deployment_name": self.deployment_factory.build_deployment_name(model),
            }
            for public_name in (model.name, model.alias, model.served_model_name):
                if public_name:
                    model_targets[public_name] = target
        return build_gateway_ingress_spec(
            settings=settings,
            service_name=self.service_name,
            model_targets=model_targets,
        )

    def build_gateway_binding(
        self,
        registry: ModelRegistry,
        *,
        settings: Settings,
        serve: Any | None = None,
    ) -> Any:
        """Create the one CPU-only public Serve gateway binding."""
        return build_gateway_ingress_binding(
            serve=serve or self.require_ray_serve(),
            settings=settings,
            spec=self.build_gateway_spec(registry, settings=settings),
        )

    def require_ray_serve(self) -> Any:
        """执行运行时相关逻辑。"""
        try:
            from ray import serve
        except ImportError as exc:
            raise RuntimeError(
                "Ray Serve is not installed. Install the 'serve' extra to enable runtime integration."
            ) from exc
        return serve

    def build_runtime_context(self, registry: ModelRegistry, model_name: str) -> dict[str, Any]:
        """构建模型副本所需的运行时上下文。"""
        model = registry.get(model_name)
        model_reference = self.model_store.resolve_model_reference(model)
        runtime_spec = self.backend.build_runtime_spec(model, model_reference)
        runtime_spec["backend_init_mode"] = self.backend_init_mode
        runtime_context = {
            "model_name": model.name,
            "model_alias": model.alias,
            "app_name": self.build_application_name(model.name),
            "served_model_name": model.served_model_name or model.alias or model.name,
            "task": model.task,
            "compat_mode": model.compat_mode.value,
            "capabilities": list(model.capabilities),
            "deployment_name": self.deployment_factory.build_deployment_name(model),
            "resolved_model_path": str(model_reference),
            "runtime_spec": runtime_spec,
        }
        self.backend.validate_runtime_spec(runtime_spec, runtime_context)
        return runtime_context

    def validate_registry_runtime_configs(self, registry: ModelRegistry) -> None:
        """执行运行时相关逻辑。"""
        for model in registry.list_models():
            if model.backend != BackendType.VLLM:
                continue
            self.build_runtime_context(registry, model.name)

    def build_serve_bindings(
        self,
        registry: ModelRegistry,
        serve: Any | None = None,
        replica_cls: type[ModelRuntimeReplica] = ModelRuntimeReplica,
    ) -> dict[str, Any]:
        """为注册模型创建 Serve 部署绑定。"""
        serve_runtime = serve or self.require_ray_serve()
        bindings: dict[str, Any] = {}

        for spec in self.build_specs(registry):
            deployment = serve_runtime.deployment(
                **self.deployment_factory.build_serve_deployment_kwargs(spec)
            )(replica_cls)
            runtime_context = self.build_runtime_context(registry, spec.model_name)
            bindings[spec.model_name] = deployment.bind(runtime_context)

        return bindings
