from typing import Any, Literal

from pydantic import BaseModel, Field

from infer_nexus.core.enums import BackendType, ModelStatus, TaskType


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "infer-nexus"


class ModelSummary(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "infer-nexus"
    task: TaskType
    backend: BackendType
    status: ModelStatus
    alias: str | None = None


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelSummary]


class CatalogModelResponse(BaseModel):
    name: str
    alias: str | None = None
    task: TaskType
    backend: BackendType
    model_path: str
    dtype: str | None = None
    tensor_parallel_size: int = Field(ge=1)
    max_model_len: int | None = None
    cpu_per_replica: int | float = Field(gt=0)
    gpu_per_replica: int | float = Field(ge=0)
    min_replicas: int = Field(ge=0)
    max_replicas: int = Field(ge=1)
    capabilities: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    status: ModelStatus = ModelStatus.UNKNOWN


class ClusterLoadResponse(BaseModel):
    status: str
    message: str
    active_models: int


class ModelStatusResponse(BaseModel):
    name: str
    status: ModelStatus
    message: str


class OpenAIErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorDetail


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]
    name: str | None = None


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionsResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: TokenUsage


class EmbeddingInputItem(BaseModel):
    text: str


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] | None = "float"
    user: str | None = None


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | str


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: TokenUsage


class RerankDocument(BaseModel):
    text: str


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: str | list[str]
    top_n: int = Field(default=0, ge=0)
    user: str | None = None


class RerankUsage(BaseModel):
    total_tokens: int


class RerankResult(BaseModel):
    index: int
    document: RerankDocument
    relevance_score: float


class RerankResponse(BaseModel):
    id: str
    model: str
    usage: RerankUsage
    results: list[RerankResult]
