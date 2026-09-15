"""Private adapters for Ray and vLLM logging lifecycle differences."""

from __future__ import annotations

import logging
import os
from typing import Any

from infer_nexus.core.config import LoggingSettings


def serve_logging_config(
    settings: LoggingSettings,
    *,
    enable_access_log: bool | None = None,
) -> dict[str, Any]:
    """Build the dictionary accepted by Ray Serve 2.55's logging_config API."""
    return {
        "encoding": "JSON" if settings.format == "json" else "TEXT",
        "log_level": settings.named_levels.get("ray.serve", settings.level),
        "enable_access_log": (
            settings.access_log if enable_access_log is None else enable_access_log
        ),
    }


def configure_ray_logging_environment(settings: LoggingSettings) -> None:
    """Set Ray's pre-import format defaults without overriding operator choices."""
    encoding = "JSON" if settings.format == "json" else "TEXT"
    os.environ.setdefault(
        "RAY_LOGGING_CONFIG_ENCODING",
        encoding,
    )
    os.environ.setdefault("RAY_BACKEND_LOG_JSON", "1" if encoding == "JSON" else "0")
    os.environ.setdefault("RAY_ROTATION_MAX_BYTES", str(50 * 1024 * 1024))
    os.environ.setdefault("RAY_ROTATION_BACKUP_COUNT", "3")


def ray_core_logging_config(ray_module: Any, settings: LoggingSettings) -> Any | None:
    """Construct the pinned Ray Core logging config when the API is available."""
    config_type = getattr(ray_module, "LoggingConfig", None)
    if config_type is None:
        return None
    return config_type(
        encoding="JSON" if settings.format == "json" else "TEXT",
        log_level=settings.named_levels.get("ray", settings.level),
    )


def configure_vllm_logging(settings: LoggingSettings) -> None:
    """Let vLLM records propagate to the process-owned structured handler."""
    vllm_level = settings.named_levels.get("vllm", settings.level)
    os.environ["VLLM_LOGGING_LEVEL"] = vllm_level
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    for logger_name, candidate in logging.root.manager.loggerDict.items():
        if logger_name == "vllm" or logger_name.startswith("vllm."):
            if isinstance(candidate, logging.Logger):
                candidate.handlers.clear()
                candidate.propagate = True
                candidate.setLevel(logging.getLevelName(vllm_level))
    logging.getLogger("vllm").propagate = True
    logging.getLogger("vllm").setLevel(logging.getLevelName(vllm_level))


def serve_replica_identity(runtime_context: dict[str, Any]) -> dict[str, Any]:
    """Read Serve replica tags behind the one version-sensitive identity adapter."""
    identity = {
        "serve_app": runtime_context.get("app_name"),
        "deployment": runtime_context.get("deployment_name"),
        "replica_id": None,
    }
    try:
        from ray import serve

        replica_context = serve.get_replica_context()
    except (ImportError, RuntimeError, AttributeError):
        return identity
    identity.update(
        {
            "serve_app": getattr(replica_context, "app_name", identity["serve_app"]),
            "deployment": getattr(
                replica_context, "deployment", identity["deployment"]
            ),
            "replica_id": getattr(replica_context, "replica_tag", None),
        }
    )
    return identity
