"""Typed schemas for declarative model catalog entries."""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from infer_nexus.core.enums import BackendType, ModelStatus, TaskType


class ModelLoadingConfig(BaseModel):
    """Ray/vLLM-style model loading options."""

    model_id: str | None = None
    revision: str | None = None


class DeploymentConfig(BaseModel):
    """Ray Serve deployment overrides aligned with LLMConfig semantics."""

    autoscaling_config: dict[str, Any] = Field(default_factory=dict)
    ray_actor_options: dict[str, Any] = Field(default_factory=dict)


class ProxyAuthConfig(BaseModel):
    """Optional upstream auth config for proxy backends."""

    mode: str = "none"
    env_var: str | None = None
    token: str | None = None


class ProxyTimeoutConfig(BaseModel):
    """Proxy timeout settings in seconds."""

    connect_seconds: int | float = Field(default=3, gt=0)
    read_seconds: int | float = Field(default=180, gt=0)
    write_seconds: int | float = Field(default=30, gt=0)
    pool_seconds: int | float = Field(default=5, gt=0)


class ProxyRetryConfig(BaseModel):
    """Proxy retry settings."""

    max_attempts: int = Field(default=1, ge=1)
    backoff_ms: int = Field(default=0, ge=0)
    retry_on_status: list[int] = Field(default_factory=lambda: [502, 503, 504])


class ProxyStreamingConfig(BaseModel):
    """Streaming passthrough toggles."""

    enabled: bool = True
    passthrough_sse: bool = True


class ProxyHeadersPolicy(BaseModel):
    """Header forwarding policy for proxy backend."""

    pass_request_id: bool = True
    forward_authorization: bool = False


class ProxyConfig(BaseModel):
    """Per-model upstream config for OpenAI-compatible proxy mode."""

    upstream_base_url: str
    upstream_model_name: str | None = None
    auth: ProxyAuthConfig = Field(default_factory=ProxyAuthConfig)
    timeout: ProxyTimeoutConfig = Field(default_factory=ProxyTimeoutConfig)
    retry: ProxyRetryConfig = Field(default_factory=ProxyRetryConfig)
    streaming: ProxyStreamingConfig = Field(default_factory=ProxyStreamingConfig)
    headers_policy: ProxyHeadersPolicy = Field(default_factory=ProxyHeadersPolicy)


class VLLMRequestPolicy(BaseModel):
    """Per-model request passthrough policy for local vLLM backends."""

    allow_tools: bool = False
    allow_reasoning: bool = False
    passthrough_unknown_openai_fields: bool = False


class VLLMConfig(BaseModel):
    """Backend-scoped local vLLM configuration."""

    engine_kwargs: dict[str, Any] = Field(default_factory=dict)
    request_defaults: dict[str, Any] = Field(default_factory=dict)
    request_policy: VLLMRequestPolicy = Field(default_factory=VLLMRequestPolicy)


class ModelConfig(BaseModel):
    """One declarative model entry loaded from ``config/models.yaml``."""

    name: str
    alias: str | None = None
    task: TaskType
    backend: BackendType = BackendType.VLLM
    model_path: str | None = None
    model_loading_config: ModelLoadingConfig = Field(default_factory=ModelLoadingConfig)
    dtype: str | None = None
    tensor_parallel_size: int = Field(ge=1)
    max_model_len: int | None = None
    cpu_per_replica: int | float = Field(gt=0)
    gpu_per_replica: int | float = Field(ge=0)
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    min_replicas: int = Field(ge=0)
    max_replicas: int = Field(ge=1)
    capabilities: list[str] = Field(default_factory=list)
    engine_kwargs: dict[str, Any] = Field(default_factory=dict)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    deployment_config: DeploymentConfig = Field(default_factory=DeploymentConfig)
    proxy_config: ProxyConfig | None = None
    served_model_name: str | None = None
    require_local_artifacts: bool = True
    labels: list[str] = Field(default_factory=list)
    status: ModelStatus = ModelStatus.UNKNOWN

    @model_validator(mode="after")
    def validate_replica_bounds(self) -> "ModelConfig":
        """Ensure replica bounds are internally consistent."""
        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")

        merged_engine_kwargs = dict(self.engine_kwargs)
        merged_engine_kwargs.update(self.vllm.engine_kwargs)
        self.vllm.engine_kwargs = merged_engine_kwargs
        self.engine_kwargs = dict(merged_engine_kwargs)

        if self.backend == BackendType.VLLM_OPENAI_PROXY:
            if self.proxy_config is None:
                raise ValueError("proxy_config is required for vllm_openai_proxy backend")
            return self

        if not self.model_path and not self.model_loading_config.model_id:
            raise ValueError("either model_path or model_loading_config.model_id must be set")
        return self


class ModelCatalogFile(BaseModel):
    """Top-level model catalog file."""

    models: list[ModelConfig]
