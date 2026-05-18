"""应用配置模型与加载逻辑。"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from infer_nexus.core.errors import ConfigError


class ServiceSettings(BaseModel):
    """Gateway service networking and request-header defaults."""

    name: str = "infer-nexus"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    api_key_header: str = "X-API-Key"


class CatalogSettings(BaseModel):
    """Catalog file location settings."""

    models_path: str = "config/models.yaml"


class ClusterSettings(BaseModel):
    """Cluster-level runtime environment assumptions."""

    accelerator_type: str = "cuda"
    device_pool_boundary: str = "ray_runtime_visible_devices"
    default_platform: str = "cuda"


class SchedulerSettings(BaseModel):
    """Admission and scaling policy thresholds."""

    enable_admission_control: bool = True
    scale_up_cooldown_sec: int = 30
    scale_down_cooldown_sec: int = 300
    queue_length_threshold: int = 8
    ttft_threshold_ms: int = 2500
    p95_latency_threshold_ms: int = 10000


class RuntimeSettings(BaseModel):
    """Runtime backend and execution wiring modes."""

    device_env_strategy: str = "ray_managed"
    backend: str = "vllm"
    execution_mode: str = "stub"
    backend_init_mode: str = "stub"


class ModelStoreSettings(BaseModel):
    """Local model artifact store settings."""

    root_dir: str = "models"
    huggingface_endpoint: str = "https://hf-mirror.com"


class Settings(BaseModel):
    """Top-level platform settings loaded from config/settings.yaml."""

    service: ServiceSettings = Field(default_factory=ServiceSettings)
    catalog: CatalogSettings = Field(default_factory=CatalogSettings)
    cluster: ClusterSettings = Field(default_factory=ClusterSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    model_store: ModelStoreSettings = Field(default_factory=ModelStoreSettings)


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    """Load and validate YAML settings into typed config objects."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"settings file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return Settings.model_validate(raw)
