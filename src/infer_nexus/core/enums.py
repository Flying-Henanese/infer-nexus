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
