"""请求准入控制。"""

from infer_nexus.catalog.models import ModelConfig


class AdmissionController:
    """按策略决定请求是否允许进入推理执行阶段。"""

    def __init__(self, enabled: bool = True) -> None:
        """初始化准入控制器。"""
        self.enabled = enabled

    def check_model_request(self, model: ModelConfig) -> None:
        """检查模型请求是否允许通过。"""
        # Phase 1 skeleton: hook for queue, TTFT, and capacity-aware decisions.
        if not self.enabled:
            return
        return
