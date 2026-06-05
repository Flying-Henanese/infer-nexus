"""API 请求、响应和内部传输使用的 Pydantic 数据模型。"""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from infer_nexus.core.enums import BackendType, CompatibilityMode, ModelStatus, TaskType


class HealthResponse(BaseModel):
    """服务存活检查接口的响应体。"""

    status: Literal["ok"] = "ok"
    service: str = "infer-nexus"


class ModelSummary(BaseModel):
    """OpenAI 风格模型列表中的单个模型条目。"""

    id: str
    object: Literal["model"] = "model"
    owned_by: str = "infer-nexus"
    task: TaskType
    backend: BackendType
    compat_mode: CompatibilityMode = CompatibilityMode.LOCAL_BEST_EFFORT
    status: ModelStatus
    alias: str | None = None


class ModelListResponse(BaseModel):
    """OpenAI 兼容的模型列表响应外壳。"""

    object: Literal["list"] = "list"
    data: list[ModelSummary]


class CatalogModelResponse(BaseModel):
    """平台原生 API 返回的模型元数据。"""

    name: str
    alias: str | None = None
    task: TaskType
    backend: BackendType
    compat_mode: CompatibilityMode = CompatibilityMode.LOCAL_BEST_EFFORT
    model_path: str | None = None
    model_loading_config: dict[str, Any] = Field(default_factory=dict)
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
    vllm: dict[str, Any] = Field(default_factory=dict)
    deployment_config: dict[str, Any] = Field(default_factory=dict)
    served_model_name: str | None = None
    require_local_artifacts: bool = True
    labels: list[str] = Field(default_factory=list)
    status: ModelStatus = ModelStatus.UNKNOWN


class ClusterLoadResponse(BaseModel):
    """平台原生 API 返回的集群负载摘要。"""

    status: str
    message: str
    active_models: int


class ModelStatusResponse(BaseModel):
    """单个模型运行状态响应。"""

    name: str
    status: ModelStatus
    message: str


class OpenAIErrorDetail(BaseModel):
    """OpenAI 风格错误响应中的错误详情对象。"""

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    """OpenAI 风格错误响应外壳。"""

    error: OpenAIErrorDetail


class ChatTextContentPart(BaseModel):
    """OpenAI 兼容的文本内容块。"""

    type: Literal["text"]
    text: str


class ChatImageURL(BaseModel):
    """OpenAI 兼容图片 URL 内容块的负载。"""

    url: str
    detail: Literal["auto", "low", "high"] | None = None


class ChatImageContentPart(BaseModel):
    """OpenAI 兼容的图片内容块。"""

    type: Literal["image_url"]
    image_url: ChatImageURL


ChatContentPart: TypeAlias = Annotated[
    ChatTextContentPart | ChatImageContentPart,
    Field(discriminator="type"),
]


class ChatMessage(BaseModel):
    """OpenAI 兼容 chat API 使用的单条消息。"""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ChatContentPart] | None = None
    reasoning_content: str | None = None
    reasoning: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    function_call: dict[str, Any] | None = None


class ChatCompletionsRequest(BaseModel):
    """OpenAI 兼容的 chat completions 请求。"""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    stop: str | list[str] | None = None
    n: int | None = Field(default=None, ge=1)
    seed: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0)
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None
    user: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    parallel_tool_calls: bool | None = None
    stream_options: dict[str, Any] | None = None
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    """chat completion 返回的单个候选回复。"""

    index: int
    message: ChatMessage
    finish_reason: str | None = None


class TokenUsage(BaseModel):
    """各类响应共用的 token 用量对象。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionsResponse(BaseModel):
    """OpenAI 兼容的 chat completions 响应。"""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: TokenUsage


class EmbeddingInputItem(BaseModel):
    """结构化 embedding 输入项包装。"""

    text: str


class EmbeddingRequest(BaseModel):
    """OpenAI 兼容的 embeddings 请求。"""

    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] | None = "float"
    dimensions: int | None = Field(default=None, ge=1)
    user: str | None = None


class EmbeddingData(BaseModel):
    """embeddings 响应中的单个向量条目。"""

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | str


class EmbeddingResponse(BaseModel):
    """OpenAI 兼容的 embeddings 响应。"""

    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: TokenUsage


class RerankDocument(BaseModel):
    """rerank 输入文档包装。"""

    text: str


class RerankRequest(BaseModel):
    """平台原生 rerank 请求模型。"""

    model: str
    query: str
    documents: str | list[str]
    top_n: int = Field(default=0, ge=0)
    user: str | None = None


class RerankUsage(BaseModel):
    """rerank 响应的 token 用量。"""

    total_tokens: int


class RerankResult(BaseModel):
    """单个 rerank 打分结果。"""

    index: int
    document: RerankDocument
    relevance_score: float


class RerankResponse(BaseModel):
    """平台原生 rerank 响应模型。"""

    id: str
    model: str
    usage: RerankUsage
    results: list[RerankResult]
