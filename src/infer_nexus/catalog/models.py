"""声明式模型目录条目的类型化结构。

目录结构描述 ``infer-nexus`` 可以本地服务或代理转发的所有模型。这些 Pydantic
模型会把 YAML 配置标准化为类型化 Python 对象，执行跨字段约束校验，并为运行时、
路由和后端层提供默认值。
"""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from infer_nexus.core.enums import BackendType, CompatibilityMode, ModelStatus, TaskType


class ModelLoadingConfig(BaseModel):
    """远程模型产物加载所需的可选模型标识和版本。

    ``model_id`` 通常是 Hugging Face 风格的仓库标识。设置后，运行时启动前可以把
    它解析为本地产物路径。``revision`` 用于在产物提供方支持版本化引用时固定分支、
    标签或提交。
    """

    model_id: str | None = None
    revision: str | None = None


class DeploymentConfig(BaseModel):
    """与 LLMConfig 语义对齐的 Ray Serve 部署覆盖项。

    这些字段刻意保留为自由结构的字典，因为 Ray Serve 和 vLLM 的部署选项会独立于
    目录结构演进。运行时构建器会把它们传递给部署构造层。
    """

    autoscaling_config: dict[str, Any] = Field(default_factory=dict)
    ray_actor_options: dict[str, Any] = Field(default_factory=dict)
    request_router_config: dict[str, Any] = Field(default_factory=dict)
    serve_deployment_kwargs: dict[str, Any] = Field(default_factory=dict)


class ProxyAuthConfig(BaseModel):
    """OpenAI 兼容上游代理目标的认证配置。"""

    mode: str = "none"
    env_var: str | None = None
    token: str | None = None


class ProxyTimeoutConfig(BaseModel):
    """上游代理 HTTP 请求的超时配置，单位为秒。"""

    connect_seconds: int | float = Field(default=3, gt=0)
    read_seconds: int | float = Field(default=180, gt=0)
    write_seconds: int | float = Field(default=30, gt=0)
    pool_seconds: int | float = Field(default=5, gt=0)


class ProxyRetryConfig(BaseModel):
    """上游代理瞬时失败的重试策略。"""

    max_attempts: int = Field(default=1, ge=1)
    backoff_ms: int = Field(default=0, ge=0)
    retry_on_status: list[int] = Field(default_factory=lambda: [502, 503, 504])


class ProxyStreamingConfig(BaseModel):
    """控制代理后端的流式透传行为。"""

    enabled: bool = True
    passthrough_sse: bool = True


class ProxyHeadersPolicy(BaseModel):
    """OpenAI 兼容代理请求的请求头转发策略。"""

    pass_request_id: bool = True
    forward_authorization: bool = False


class ProxyConfig(BaseModel):
    """OpenAI 兼容代理模式下的单模型上游配置。

    代理模型不会启动本地 vLLM engine。请求会按照本模型的认证、超时、重试、流式和
    请求头策略转发到 ``upstream_base_url``。
    """

    upstream_base_url: str
    upstream_model_name: str | None = None
    auth: ProxyAuthConfig = Field(default_factory=ProxyAuthConfig)
    timeout: ProxyTimeoutConfig = Field(default_factory=ProxyTimeoutConfig)
    retry: ProxyRetryConfig = Field(default_factory=ProxyRetryConfig)
    streaming: ProxyStreamingConfig = Field(default_factory=ProxyStreamingConfig)
    headers_policy: ProxyHeadersPolicy = Field(default_factory=ProxyHeadersPolicy)


class VLLMRequestPolicy(BaseModel):
    """本地 vLLM 后端的单模型请求透传策略。

    该策略控制哪些 OpenAI 风格请求特性可以进入本地 vLLM 执行路径。除非显式开启，
    否则会阻止不受支持的工具调用、reasoning 字段或未知请求扩展进入模型。
    """

    allow_tools: bool = False
    allow_reasoning: bool = False
    passthrough_unknown_openai_fields: bool = False


class VLLMOpenAIServingConfig(BaseModel):
    """Ray replica 内 vLLM OpenAI 兼容 serving 的配置。

    启用后，strict/native 执行路径可以在 replica 内构造 vLLM 的 OpenAI serving
    对象，并按其语义处理请求，而不是走本地 best-effort 的 engine 翻译路径。
    """

    enabled: bool = False
    enable_reasoning: bool = False
    reasoning_parser: str | None = None


class VLLMConfig(BaseModel):
    """本地 vLLM 模型部署的后端级配置。"""

    engine_kwargs: dict[str, Any] = Field(default_factory=dict)
    request_defaults: dict[str, Any] = Field(default_factory=dict)
    request_policy: VLLMRequestPolicy = Field(default_factory=VLLMRequestPolicy)
    openai_serving: VLLMOpenAIServingConfig = Field(default_factory=VLLMOpenAIServingConfig)


class ModelConfig(BaseModel):
    """从模型目录加载的一条声明式模型配置。

    ``ModelConfig`` 汇总单个已注册模型的用户侧标识、任务类型、后端类型、本地或代理
    运行时配置、资源需求、部署覆盖项和运维元数据。
    """

    name: str
    alias: str | None = None
    task: TaskType
    backend: BackendType = BackendType.VLLM
    compat_mode: CompatibilityMode = CompatibilityMode.LOCAL_BEST_EFFORT
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
        """校验跨字段约束，并标准化 engine kwargs。

        该校验器负责处理单字段约束无法表达的关系：

        - ``max_replicas`` 不能小于 ``min_replicas``。
        - 顶层 ``engine_kwargs`` 会与 ``vllm.engine_kwargs`` 合并。
        - 代理后端必须定义 ``proxy_config``。
        - native/strict 本地 vLLM 模式只能用于受支持的任务，并在需要时要求启用
          OpenAI serving。
        - request router 配置只接受本地 vLLM chat 模型使用。
        - 本地非代理模型必须提供 ``model_path`` 或可加载的
          ``model_loading_config.model_id``。

        Returns:
            标准化并校验后的模型配置。

        Raises:
            ValueError: 当配置包含不支持的字段组合时抛出。
        """
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

        if self.compat_mode in {
            CompatibilityMode.VLLM_NATIVE,
            CompatibilityMode.STRICT_OPENAI,
        } and self.backend == BackendType.VLLM:
            if self.task == TaskType.RERANK:
                raise ValueError("vllm_native is not supported for local vllm rerank models")
            if self.compat_mode == CompatibilityMode.STRICT_OPENAI and self.task != TaskType.CHAT:
                raise ValueError("strict_openai is only supported as an alias for local vllm chat models")
            if self.task in {TaskType.CHAT, TaskType.EMBEDDING} and not self.vllm.openai_serving.enabled:
                raise ValueError(
                    "vllm_native local vllm chat and embedding models require "
                    "vllm.openai_serving.enabled=true"
                )

        if self.deployment_config.request_router_config and (
            self.backend != BackendType.VLLM or self.task != TaskType.CHAT
        ):
            raise ValueError(
                "deployment_config.request_router_config is currently only supported for "
                "local vllm chat models"
            )

        if not self.model_path and not self.model_loading_config.model_id:
            raise ValueError("either model_path or model_loading_config.model_id must be set")
        return self


class ModelCatalogFile(BaseModel):
    """模型目录文件的顶层结构。"""

    models: list[ModelConfig]
