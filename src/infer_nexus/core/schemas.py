"""API 请求/响应与内部传输的 Pydantic 数据模型。"""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from infer_nexus.core.enums import BackendType, ModelStatus, TaskType


class HealthResponse(BaseModel):
    """Health-check payload for service liveness endpoints."""

    status: Literal["ok"] = "ok"
    service: str = "infer-nexus"


class ModelSummary(BaseModel):
    """OpenAI-style model listing item."""

    id: str
    object: Literal["model"] = "model"
    owned_by: str = "infer-nexus"
    task: TaskType
    backend: BackendType
    status: ModelStatus
    alias: str | None = None


class ModelListResponse(BaseModel):
    """OpenAI-compatible model list response envelope."""

    object: Literal["list"] = "list"
    data: list[ModelSummary]


class CatalogModelResponse(BaseModel):
    """Native API model metadata response."""

    name: str
    alias: str | None = None
    task: TaskType
    backend: BackendType
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
    """Native API cluster load summary."""

    status: str
    message: str
    active_models: int


class ModelStatusResponse(BaseModel):
    """Per-model runtime status response."""

    name: str
    status: ModelStatus
    message: str


class OpenAIErrorDetail(BaseModel):
    """OpenAI-style error object body."""

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    """OpenAI-style error response envelope."""

    error: OpenAIErrorDetail


class ChatTextContentPart(BaseModel):
    """OpenAI-compatible text content block."""

    type: Literal["text"]
    text: str


class ChatImageURL(BaseModel):
    """OpenAI-compatible image URL block payload."""

    url: str
    detail: Literal["auto", "low", "high"] | None = None


class ChatImageContentPart(BaseModel):
    """OpenAI-compatible image content block."""

    type: Literal["image_url"]
    image_url: ChatImageURL


ChatContentPart: TypeAlias = Annotated[
    ChatTextContentPart | ChatImageContentPart,
    Field(discriminator="type"),
]


class ChatMessage(BaseModel):
    """Chat message unit used in OpenAI-compatible chat APIs."""

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
    """OpenAI-compatible chat completions request."""

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
    """Single assistant candidate returned by chat completion."""

    index: int
    message: ChatMessage
    finish_reason: str | None = None


class TokenUsage(BaseModel):
    """Token accounting object used across response types."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionsResponse(BaseModel):
    """OpenAI-compatible chat completions response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: TokenUsage


class EmbeddingInputItem(BaseModel):
    """Embedding input item wrapper for structured variants."""

    text: str


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embeddings request."""

    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] | None = "float"
    user: str | None = None


class EmbeddingData(BaseModel):
    """Single embedding vector entry in the embeddings response."""

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | str


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embeddings response."""

    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: TokenUsage


class RerankDocument(BaseModel):
    """Rerank input document wrapper."""

    text: str


class RerankRequest(BaseModel):
    """Native rerank request model."""

    model: str
    query: str
    documents: str | list[str]
    top_n: int = Field(default=0, ge=0)
    user: str | None = None


class RerankUsage(BaseModel):
    """Token accounting for rerank responses."""

    total_tokens: int


class RerankResult(BaseModel):
    """Single rerank scoring output entry."""

    index: int
    document: RerankDocument
    relevance_score: float


class RerankResponse(BaseModel):
    """Native rerank response model."""

    id: str
    model: str
    usage: RerankUsage
    results: list[RerankResult]
