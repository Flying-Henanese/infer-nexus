"""应用配置模型和加载逻辑。"""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from infer_nexus.core.errors import ConfigError


class ServiceSettings(BaseModel):
    """网关服务的网络监听和请求头默认配置。"""

    name: str = "infer-nexus"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = Field(default=1, ge=1)
    log_level: str = "INFO"
    api_key_header: str = "X-API-Key"


_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class LoggingSettings(BaseModel):
    """Structured application logging behavior shared by every process role."""

    format: Literal["console", "json"] = "console"
    level: str = "INFO"
    success_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    slow_request_ms: int = Field(default=10_000, ge=0)
    include_traceback: bool = True
    access_log: bool = True
    # Local development keeps the event sink disabled. Compose enables it with
    # the in-container directory mounted by each physical service.
    event_log_dir: str | None = None
    event_max_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    event_backup_count: int = Field(default=5, gt=0)
    event_writer_error_interval_seconds: float = Field(default=60.0, gt=0)
    named_levels: dict[str, str] = Field(
        default_factory=lambda: {
            "ray": "INFO",
            "ray.serve": "INFO",
            "uvicorn.access": "WARNING",
            "vllm": "INFO",
        }
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in _LOG_LEVELS:
            raise ValueError(f"unsupported logging level: {value}")
        return normalized

    @field_validator("named_levels")
    @classmethod
    def validate_named_levels(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for logger_name, level in value.items():
            normalized_level = level.upper()
            if normalized_level not in _LOG_LEVELS:
                raise ValueError(
                    f"unsupported logging level for '{logger_name}': {level}"
                )
            normalized[logger_name] = normalized_level
        return normalized


class ObservabilitySettings(BaseModel):
    """Settings for logs and other application observability interfaces."""

    logging: LoggingSettings = Field(default_factory=LoggingSettings)


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
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)


def _apply_logging_migration(raw: dict) -> dict:
    """Read the legacy service level only when the new setting is absent."""
    if not isinstance(raw, dict):
        return raw
    normalized = dict(raw)
    service = normalized.get("service") or {}
    if not isinstance(service, dict):
        service = {}
    observability = dict(normalized.get("observability") or {})
    logging_settings = dict(observability.get("logging") or {})
    if "level" not in logging_settings and "log_level" in service:
        logging_settings["level"] = service["log_level"]
    observability["logging"] = logging_settings
    normalized["observability"] = observability
    return normalized


def _apply_logging_env_overrides(raw: dict) -> dict:
    """Apply the intentionally small, explicitly named logging overrides."""
    overrides = {
        "format": os.getenv("INFER_NEXUS_LOG_FORMAT"),
        "level": os.getenv("INFER_NEXUS_LOG_LEVEL"),
        "event_log_dir": os.getenv("INFER_NEXUS_EVENT_LOG_DIR"),
    }
    if all(value is None for value in overrides.values()):
        return raw
    if not isinstance(raw, dict):
        return raw
    normalized = dict(raw)
    observability = dict(normalized.get("observability") or {})
    logging_settings = dict(observability.get("logging") or {})
    for key, value in overrides.items():
        if value is not None:
            logging_settings[key] = value or None
    level_override = overrides["level"]
    if level_override is not None:
        named_levels = dict(logging_settings.get("named_levels") or {})
        named_levels["infer_nexus"] = level_override
        for logger_name in named_levels:
            if logger_name == "infer_nexus" or logger_name.startswith("infer_nexus."):
                named_levels[logger_name] = level_override
        logging_settings["named_levels"] = named_levels
    observability["logging"] = logging_settings
    normalized["observability"] = observability
    return normalized


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

    raw = _apply_logging_migration(raw)
    raw = _apply_logging_env_overrides(raw)
    return _apply_env_overrides(Settings.model_validate(raw))
