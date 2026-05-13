from dataclasses import dataclass


@dataclass(slots=True)
class MetricsSnapshot:
    active_models: int
    status: str = "stub"
    message: str = "metrics backend not connected"
