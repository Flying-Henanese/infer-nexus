from infer_nexus.catalog.models import ModelConfig


class AdmissionController:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def check_model_request(self, model: ModelConfig) -> None:
        # Phase 1 skeleton: hook for queue, TTFT, and capacity-aware decisions.
        if not self.enabled:
            return
        return
