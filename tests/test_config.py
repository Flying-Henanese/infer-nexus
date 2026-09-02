"""Configuration loading tests."""

from pathlib import Path

import pytest

from infer_nexus.core.config import load_settings


def test_load_settings_applies_ray_address_env_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Compose can provide the Ray address without hardcoding it in YAML."""
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "runtime:\n"
        "  execution_mode: serve\n"
        "  ray_address: yaml-ray:6379\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("INFER_NEXUS_RAY_ADDRESS", "ray-head:6379")

    settings = load_settings(settings_path)

    assert settings.runtime.execution_mode == "serve"
    assert settings.runtime.ray_address == "ray-head:6379"


def test_load_settings_empty_ray_address_env_clears_yaml_value(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """An empty deployment override keeps local non-Compose runs from inheriting YAML topology."""
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "runtime:\n"
        "  execution_mode: serve\n"
        "  ray_address: yaml-ray:6379\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("INFER_NEXUS_RAY_ADDRESS", "  ")

    settings = load_settings(settings_path)

    assert settings.runtime.ray_address is None


def test_load_settings_reads_bounded_serve_gateway_ingress_config(tmp_path: Path) -> None:
    """Gateway ingress capacity is declarative and never inherits unbounded Serve defaults."""
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "runtime:\n"
        "  gateway_ingress:\n"
        "    application_name: custom-gateway\n"
        "    route_prefix: /\n"
        "    num_replicas: 2\n"
        "    num_cpus: 0.5\n"
        "    max_ongoing_requests: 12\n"
        "    max_queued_requests: 24\n",
        encoding="utf-8",
    )

    settings = load_settings(settings_path)

    assert settings.runtime.gateway_ingress.application_name == "custom-gateway"
    assert settings.runtime.gateway_ingress.num_replicas == 2
    assert settings.runtime.gateway_ingress.num_cpus == 0.5
    assert settings.runtime.gateway_ingress.max_ongoing_requests == 12
    assert settings.runtime.gateway_ingress.max_queued_requests == 24


def test_load_settings_rejects_an_invalid_zero_serve_gateway_queue(tmp_path: Path) -> None:
    """Ray 2.55 requires a finite queue limit of at least one request."""
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "runtime:\n"
        "  gateway_ingress:\n"
        "    max_queued_requests: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_queued_requests"):
        load_settings(settings_path)
