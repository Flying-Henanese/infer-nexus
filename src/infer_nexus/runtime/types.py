from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RuntimeTarget:
    model_name: str
    model_alias: str | None
    deployment_name: str
    runtime_context: dict[str, Any]
