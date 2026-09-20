"""Configuration loading tests."""

from pathlib import Path

import pytest

from infer_nexus.core.config import load_settings


@pytest.fixture(autouse=True)
def clear_logging_environment_overrides(monkeypatch) -> None:
    monkeypatch.delenv("INFER_NEXUS_LOG_FORMAT", raising=False)
    monkeypatch.delenv("INFER_NEXUS_LOG_LEVEL", raising=False)
    monkeypatch.delenv("INFER_NEXUS_EVENT_LOG_DIR", raising=False)


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


def test_load_settings_uses_observability_logging_configuration(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "service:\n  log_level: WARNING\n"
        "observability:\n  logging:\n    format: json\n    level: DEBUG\n",
        encoding="utf-8",
    )

    settings = load_settings(settings_path)

    assert settings.observability.logging.format == "json"
    assert settings.observability.logging.level == "DEBUG"


def test_load_settings_reads_legacy_log_level_when_new_level_is_absent(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("service:\n  log_level: WARNING\n", encoding="utf-8")

    settings = load_settings(settings_path)

    assert settings.observability.logging.level == "WARNING"


def test_new_logging_level_takes_precedence_over_legacy_level(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "service:\n  log_level: WARNING\n"
        "observability:\n  logging:\n    level: DEBUG\n",
        encoding="utf-8",
    )

    settings = load_settings(settings_path)

    assert settings.observability.logging.level == "DEBUG"


def test_logging_environment_overrides_level_and_format(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "observability:\n  logging:\n    format: console\n    level: WARNING\n"
        "    named_levels:\n      infer_nexus: INFO\n      ray: ERROR\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INFER_NEXUS_LOG_FORMAT", "json")
    monkeypatch.setenv("INFER_NEXUS_LOG_LEVEL", "debug")

    settings = load_settings(settings_path)

    assert settings.observability.logging.format == "json"
    assert settings.observability.logging.level == "DEBUG"
    assert settings.observability.logging.named_levels["infer_nexus"] == "DEBUG"
    assert settings.observability.logging.named_levels["ray"] == "ERROR"


def test_logging_level_environment_override_replaces_default_app_logger_level(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("observability:\n  logging:\n    level: INFO\n", encoding="utf-8")
    monkeypatch.setenv("INFER_NEXUS_LOG_LEVEL", "DEBUG")

    settings = load_settings(settings_path)

    assert settings.observability.logging.named_levels["infer_nexus"] == "DEBUG"


def test_logging_event_directory_environment_override_is_typed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "observability:\n  logging:\n    event_log_dir: /from-yaml\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INFER_NEXUS_EVENT_LOG_DIR", str(tmp_path / "events"))

    settings = load_settings(settings_path)

    assert settings.observability.logging.event_log_dir == str(tmp_path / "events")


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("format", "yaml"),
        ("level", "TRACE"),
        ("success_sample_rate", "1.1"),
        ("slow_request_ms", "-1"),
        ("event_max_bytes", "0"),
        ("event_backup_count", "0"),
        ("event_writer_error_interval_seconds", "0"),
    ],
)
def test_load_settings_rejects_invalid_logging_values(
    tmp_path: Path,
    setting: str,
    value: str,
) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        f"observability:\n  logging:\n    {setting}: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_settings(settings_path)


def test_load_settings_rejects_an_invalid_named_logger_level(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "observability:\n  logging:\n    named_levels:\n      ray.serve: TRACE\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported logging level"):
        load_settings(settings_path)


@pytest.mark.parametrize(
    ("path", "expected_format"),
    [
        ("config/settings.yaml", "console"),
        ("config/settings.compose.yaml", "json"),
        ("config/settings.ascend-compose.yaml", "json"),
    ],
)
def test_deployment_settings_select_logging_profile(path: str, expected_format: str) -> None:
    settings = load_settings(path)
    assert settings.observability.logging.format == expected_format
    if expected_format == "json":
        assert settings.observability.logging.event_log_dir == "/var/log/infer-nexus"
