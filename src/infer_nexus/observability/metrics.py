"""可观测性指标数据结构。"""

from dataclasses import dataclass


@dataclass(slots=True)
class MetricsSnapshot:
    """负载快照（阶段一占位指标）。"""

    active_models: int
    status: str = "stub"
    message: str = "metrics backend not connected"
