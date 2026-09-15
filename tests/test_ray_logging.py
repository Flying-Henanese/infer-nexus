"""Compatibility checks for the Ray version pinned by the CUDA lockfile."""

import importlib
import logging
import os

import pytest

from infer_nexus.core.config import LoggingSettings
from infer_nexus.observability.ray_logging import (
    configure_ray_logging_environment,
    configure_vllm_logging,
    ray_core_logging_config,
    serve_logging_config,
)


@pytest.mark.parametrize(
    ("format_name", "expected_encoding", "expected_backend_json"),
    [("console", "TEXT", "0"), ("json", "JSON", "1")],
)
def test_ray_environment_defaults_follow_logging_profile(
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
    expected_encoding: str,
    expected_backend_json: str,
) -> None:
    monkeypatch.delenv("RAY_LOGGING_CONFIG_ENCODING", raising=False)
    monkeypatch.delenv("RAY_BACKEND_LOG_JSON", raising=False)

    configure_ray_logging_environment(LoggingSettings(format=format_name))

    assert os.environ["RAY_LOGGING_CONFIG_ENCODING"] == expected_encoding
    assert os.environ["RAY_BACKEND_LOG_JSON"] == expected_backend_json


def test_logging_adapters_match_pinned_ray_schema() -> None:
    """Core and Serve accept the JSON logging settings used by the deployer."""
    ray = pytest.importorskip("ray", reason="The pinned Ray Serve extra is required.")
    from ray.serve.schema import LoggingConfig as ServeLoggingConfig

    assert ray.__version__ == "2.55.1"
    settings = LoggingSettings(format="json", level="INFO", access_log=True)

    core_config = ray_core_logging_config(ray, settings)
    serve_config = ServeLoggingConfig(**serve_logging_config(settings))

    assert isinstance(core_config, ray.LoggingConfig)
    assert core_config.encoding == "JSON"
    assert serve_config.encoding == "JSON"
    assert serve_config.log_level == "INFO"
    assert serve_config.enable_access_log is True
    quiet_config = ServeLoggingConfig(
        **serve_logging_config(settings, enable_access_log=False)
    )
    assert quiet_config.enable_access_log is False


def test_vllm_adapter_disables_duplicate_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The installed vLLM library inherits the process-owned logging handler."""
    vllm = pytest.importorskip("vllm", reason="A CUDA or Ascend vLLM runtime is required.")
    version = getattr(vllm, "__version__", "")
    assert version.startswith("0.18.")

    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "1")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "INFO")
    logger = logging.getLogger("vllm.logging_adapter_test")
    monkeypatch.setattr(logger, "handlers", [logging.NullHandler()])
    monkeypatch.setattr(logger, "propagate", False)
    settings = LoggingSettings(named_levels={"vllm": "WARNING"})

    configure_vllm_logging(settings)
    importlib.import_module("vllm.logger")

    assert logger.handlers == []
    assert logger.propagate is True
    assert logger.level == logging.WARNING
    assert os.environ["VLLM_CONFIGURE_LOGGING"] == "0"
