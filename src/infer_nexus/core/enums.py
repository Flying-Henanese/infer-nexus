"""核心枚举定义。"""

from enum import StrEnum


class TaskType(StrEnum):
    """系统支持的逻辑推理任务类型。"""

    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    VLM = "vlm"


class ModelStatus(StrEnum):
    """对 API 暴露的模型或部署生命周期状态。"""

    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    DRAINING = "draining"
    UNKNOWN = "unknown"


class BackendType(StrEnum):
    """系统支持的推理后端类型。"""

    VLLM = "vllm"
    VLLM_OPENAI_PROXY = "vllm_openai_proxy"


class CompatibilityMode(StrEnum):
    """模型运行时预期采用的 OpenAI 兼容模式。"""

    VLLM_NATIVE = "vllm_native"
    # 兼容旧版模型目录的别名；新配置应使用 vllm_native。
    STRICT_OPENAI = "strict_openai"
    LOCAL_BEST_EFFORT = "local_best_effort"
