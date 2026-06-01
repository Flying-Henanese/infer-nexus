"""Shared runtime transfer types used across dispatcher/executor layers."""

from dataclasses import dataclass
from typing import Any

from infer_nexus.core.enums import BackendType


@dataclass(slots=True)
class RuntimeTarget:
    """Resolved runtime dispatch target for a specific model request."""

    model_name: str
    model_alias: str | None
    backend: BackendType
    app_name: str | None
    deployment_name: str
    runtime_context: dict[str, Any]
