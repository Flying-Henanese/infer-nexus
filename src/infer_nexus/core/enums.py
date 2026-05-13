from enum import StrEnum


class TaskType(StrEnum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    VLM = "vlm"


class ModelStatus(StrEnum):
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    DRAINING = "draining"
    UNKNOWN = "unknown"


class BackendType(StrEnum):
    VLLM = "vllm"
