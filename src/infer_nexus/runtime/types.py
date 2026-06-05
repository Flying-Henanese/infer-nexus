"""定义运行时分发与执行的共享类型。"""

from dataclasses import dataclass
from typing import Any

from infer_nexus.core.enums import BackendType

@dataclass(slots=True)
class RuntimeTarget:
    """描述运行时组件的数据或行为。"""

    model_name: str
    model_alias: str | None
    backend: BackendType
    app_name: str | None
    deployment_name: str
    runtime_context: dict[str, Any]
