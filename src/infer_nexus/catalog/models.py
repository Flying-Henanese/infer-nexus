"""Typed schemas for declarative model catalog entries."""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from infer_nexus.core.enums import BackendType, ModelStatus, TaskType


class ModelConfig(BaseModel):
    """模型目录中的单个模型配置（由 ``config/models.yaml`` 加载）。"""

    # 模型在系统内的唯一标识名称，用于注册、查询和路由匹配。
    name: str
    # 模型别名（可选）：用于兼容旧名称、对外展示或简化调用。
    # 为空时通常回退使用 ``name``。
    alias: str | None = None
    # 模型支持的任务类型（如文本生成、分类等），用于请求分发与能力校验。
    task: TaskType
    # 推理后端类型，决定模型实例由哪种引擎承载（默认 VLLM）。
    backend: BackendType = BackendType.VLLM
    # 模型权重或模型资源路径（本地路径/挂载路径/远程已同步路径）。
    model_path: str
    # 推理数据类型（可选），如 fp16/bf16/fp32；为空时由后端自动推断或使用默认值。
    dtype: str | None = None
    # 张量并行度（Tensor Parallelism）大小，必须 >= 1。
    # 值越大通常可利用更多 GPU 共同承载单模型，但会增加通信开销。
    tensor_parallel_size: int = Field(ge=1)
    # 模型可处理的最大上下文长度（token 数，可选）。
    # 未设置时由后端或模型配置默认值决定。
    max_model_len: int | None = None
    # 每个副本分配的 CPU 资源，必须 > 0，可为整数或小数（表示核数）。
    cpu_per_replica: int | float = Field(gt=0)
    # 每个副本分配的 GPU 资源，必须 >= 0，可为小数（按 GPU 分片比例）。
    gpu_per_replica: int | float = Field(ge=0)
    # 单副本可使用的 GPU 显存利用率上限（0, 1]，可选）。
    # 例如 0.9 表示最多使用 90% 显存，用于降低 OOM 风险。
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    # 自动伸缩最小副本数，必须 >= 0；0 表示允许缩容至无副本（冷启动模式）。
    min_replicas: int = Field(ge=0)
    # 自动伸缩最大副本数，必须 >= 1，且需满足 >= ``min_replicas``。
    max_replicas: int = Field(ge=1)
    # 能力标签列表：用于声明模型具备的具体能力特征（如函数调用、多模态等）。
    # 常用于策略匹配、路由筛选与前端能力展示。
    capabilities: list[str] = Field(default_factory=list)
    engine_kwargs: dict[str, Any] = Field(default_factory=dict)
    # 通用标签列表：用于业务分组、环境标记、A/B 实验或运营筛选。
    labels: list[str] = Field(default_factory=list)
    # 模型状态（如可用/下线/未知），用于控制是否参与调度及展示状态。
    status: ModelStatus = ModelStatus.UNKNOWN

    @model_validator(mode="after")
    def validate_replica_bounds(self) -> "ModelConfig":
        """Ensure replica bounds are internally consistent."""
        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")
        return self


class ModelCatalogFile(BaseModel):
    """模型目录配置文件的顶层结构。"""

    # 模型配置列表：每一项对应一个可注册的模型声明。
    models: list[ModelConfig]
