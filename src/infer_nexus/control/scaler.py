from infer_nexus.control.policies import ScalingPolicy


class Scaler:
    def __init__(self, policy: ScalingPolicy) -> None:
        self.policy = policy
