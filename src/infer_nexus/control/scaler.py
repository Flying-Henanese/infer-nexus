"""伸缩控制器。"""

from infer_nexus.control.policies import ScalingPolicy


class Scaler:
    """根据伸缩策略驱动副本扩缩容。"""

    def __init__(self, policy: ScalingPolicy) -> None:
        """初始化伸缩控制器。"""
        self.policy = policy
