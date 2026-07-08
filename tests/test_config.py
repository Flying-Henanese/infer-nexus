"""Configuration loading tests."""

from pathlib import Path

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