"""核心枚举定义。"""

from enum import StrEnum


class TaskType(StrEnum):
    """Supported logical inference task categories."""

    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    VLM = "vlm"


class ModelStatus(StrEnum):
    """Model/deployment lifecycle status exposed to APIs."""

    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    DRAINING = "draining"
    UNKNOWN = "unknown"


class BackendType(StrEnum):
    """Supported inference backend engines."""

    VLLM = "vllm"
    VLLM_OPENAI_PROXY = "vllm_openai_proxy"


class CompatibilityMode(StrEnum):
    """OpenAI compatibility behavior expected from a model runtime."""

    VLLM_NATIVE = "vllm_native"
    # Backward-compatible alias for older catalogs. New configs should use
    # vllm_native.
    STRICT_OPENAI = "strict_openai"
    LOCAL_BEST_EFFORT = "local_best_effort"
