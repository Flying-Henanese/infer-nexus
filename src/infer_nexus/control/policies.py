"""伸缩策略数据结构。"""

from dataclasses import dataclass


@dataclass(slots=True)
class ScalingPolicy:
    """伸缩阈值策略参数。"""

    queue_length_threshold: int
    ttft_threshold_ms: int
    p95_latency_threshold_ms: int
