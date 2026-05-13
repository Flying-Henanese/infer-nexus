from dataclasses import dataclass


@dataclass(slots=True)
class ScalingPolicy:
    queue_length_threshold: int
    ttft_threshold_ms: int
    p95_latency_threshold_ms: int
