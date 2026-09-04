"""应用配置模型和加载逻辑。"""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from infer_nexus.core.errors import ConfigError


class ServiceSettings(BaseModel):
    """网关服务的网络监听和请求头默认配置。"""

    name: str = "infer-nexus"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = Field(default=1, ge=1)
    log_level: str = "INFO"
    api_key_header: str = "X-API-Key"


class CatalogSettings(BaseModel):
    """模型目录文件位置配置。"""

    models_path: str = "config/models.yaml"


class ClusterSettings(BaseModel):
    """集群级运行环境假设。"""

    inference_device_type: Literal["cuda", "npu"] = "cuda"
    device_pool_boundary: str = "ray_runtime_visible_devices"
    default_platform: str = "cuda"


class SchedulerSettings(BaseModel):
    """准入控制和扩缩容策略阈值。"""

    enable_admission_control: bool = True
    scale_up_cooldown_sec: int = 30
    scale_down_cooldown_sec: int = 300
    queue_length_threshold: int = 8
    ttft_threshold_ms: int = 2500
    p95_latency_threshold_ms: int = 10000


class GatewayIngressSettings(BaseModel):
    """Bounded CPU-only Ray Serve settings for the public gateway ingress."""

    enabled: bool = True
    application_name: str | None = None
    route_prefix: str = "/"
    num_replicas: int = Field(default=1, ge=1)
    num_cpus: float = Field(default=0.5, gt=0)
    max_ongoing_requests: int = Field(default=16, ge=1)
    max_queued_requests: int = Field(default=32, ge=1)


class RuntimeSettings(BaseModel):
    """运行时后端和执行链路模式配置。"""

    device_env_strategy: str = "ray_managed"
    backend: str = "vllm"
    execution_mode: str = "stub"
    backend_init_mode: str = "stub"
    ray_address: str | None = None
    gateway_worker_max_inflight: int = Field(default=0, ge=0)
    gateway_worker_retry_after_seconds: int = Field(default=1, ge=1)
    serve_request_timeout_seconds: int | float = Field(default=120, gt=0)
    serve_stream_idle_timeout_seconds: int | float = Field(default=30, gt=0)
    serve_stream_max_lifetime_seconds: int | float = Field(default=900, gt=0)
    max_inflight_per_model: int = Field(default=0, ge=0)
    max_streaming_inflight_per_model: int = Field(default=0, ge=0)
    max_non_streaming_inflight_per_model: int = Field(default=0, ge=0)
    max_queued_per_model: int = Field(default=0, ge=0)
    admission_acquire_timeout_seconds: int | float = Field(default=0, ge=0)
    admission_queue_timeout_seconds: int | float = Field(default=0, ge=0)
    circuit_breaker_enabled: bool = False
    circuit_breaker_failure_threshold: int = Field(default=3, ge=1)
    circuit_breaker_cooldown_seconds: int | float = Field(default=60, gt=0)
    runtime_worker_enabled: bool = False
    runtime_worker_request_timeout_seconds: int | float = Field(default=60, gt=0)
    runtime_worker_start_timeout_seconds: int | float = Field(default=15, gt=0)
    gateway_ingress: GatewayIngressSettings = Field(default_factory=GatewayIngressSettings)


class ModelStoreSettings(BaseModel):
    """本地模型产物存储配置。"""

    root_dir: str = "models"
    huggingface_endpoint: str = "https://hf-mirror.com"


def _apply_env_overrides(settings: "Settings") -> "Settings":
    """Apply deployment-specific environment overrides after YAML loading."""
    ray_address = os.getenv("INFER_NEXUS_RAY_ADDRESS")
    if ray_address is None:
        return settings
    normalized = ray_address.strip()
    return settings.model_copy(
        update={
            "runtime": settings.runtime.model_copy(
                update={"ray_address": normalized or None}
            )
        }
    )


class Settings(BaseModel):
    """从 ``config/settings.yaml`` 加载的平台顶层配置。"""

    service: ServiceSettings = Field(default_factory=ServiceSettings)
    catalog: CatalogSettings = Field(default_factory=CatalogSettings)
    cluster: ClusterSettings = Field(default_factory=ClusterSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    model_store: ModelStoreSettings = Field(default_factory=ModelStoreSettings)


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    """加载 YAML 配置文件，并校验为类型化 ``Settings`` 对象。

    参数:
        path: 配置文件路径，默认指向 ``config/settings.yaml``。

    异常:
        ConfigError: 当配置文件不存在时抛出。
        pydantic.ValidationError: 当配置内容不符合 ``Settings`` 结构时抛出。

    返回:
        校验后的平台配置对象。
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"settings file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return _apply_env_overrides(Settings.model_validate(raw))
