"""Regression coverage for host-visible Docker Compose logs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOGGED_SERVICES = ("ray-head", "ray-worker", "serve-deployer")
COMPOSE_PROFILES = (
    ("docker-compose.yml", "${LOGS_HOST_PATH:?Run python3 scripts/prepare_compose_logs.py first}", "sh"),
    ("ascend_deploy/docker-compose.yml", "${LOGS_HOST_PATH:?Run python3 scripts/prepare_compose_logs.py first}", "bash"),
)


def _volume_sources(service: dict[str, object], container_target: str) -> list[str]:
    sources: list[str] = []
    for volume in service["volumes"]:
        if isinstance(volume, dict):
            source, target = volume["source"], volume["target"]
            assert volume["bind"]["create_host_path"] is False
        else:
            source, target, *_ = volume.split(":")
        if target == container_target:
            sources.append(source)
    return sources


@pytest.mark.parametrize(("relative_path", "log_root", "nested_shell"), COMPOSE_PROFILES)
def test_compose_exports_each_service_log_tree_to_the_host(
    relative_path: str,
    log_root: str,
    nested_shell: str,
) -> None:
    """Every long-lived runtime service has an isolated, host-visible log tree."""
    with (REPOSITORY_ROOT / relative_path).open(encoding="utf-8") as compose_file:
        compose = yaml.safe_load(compose_file)

    assert "log-init" not in compose["services"]

    for service_name in LOGGED_SERVICES:
        service = compose["services"][service_name]
        command = " ".join(service["command"])
        assert "log-init" not in service.get("depends_on", {})
        assert service["user"] == "infer-nexus"
        assert _volume_sources(service, "/var/log/infer-nexus") == [
            f"{log_root}/{service_name}"
        ]
        assert _volume_sources(service, "/tmp/ray") == [
            f"{log_root}/{service_name}/ray"
        ]
        assert "run_with_log.sh" not in command
        assert "container.log" not in command
        assert "exec " in command
        assert service["environment"]["INFER_NEXUS_EVENT_LOG_DIR"] == "/var/log/infer-nexus"
        assert service["environment"]["INFER_NEXUS_PHYSICAL_SERVICE"] == service_name

        if service_name == "ray-head":
            assert "exec\n        ray start" in command or "exec ray start" in command
            continue

        nested_command = service["command"][-1]
        assert f"exec {nested_shell} -ec '" in nested_command
        nested_script = nested_command.split(f"exec {nested_shell} -ec '", 1)[1]
        nested_script = nested_script.rstrip().replace("$$", "$")
        assert nested_script.endswith("'")
        syntax_check = subprocess.run(
            [nested_shell, "-n", "-c", nested_script[:-1]],
            check=False,
            capture_output=True,
            text=True,
        )
        assert syntax_check.returncode == 0, syntax_check.stderr


def test_compose_command_keeps_stdout_and_stderr_visible_to_docker() -> None:
    """The command has no redirection, so Docker remains the combined console source."""
    result = subprocess.run(
        ["sh", "-c", "printf stdout; printf stderr >&2"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == "stdout"
    assert result.stderr == "stderr"


def test_host_log_preparation_persists_default_and_preserves_existing_files(tmp_path: Path) -> None:
    repository = tmp_path / "infer-nexus"
    repository.mkdir()
    env_file = repository / ".env"
    original = f"APP_UID={os.getuid()}\nAPP_GID={os.getgid()}\nOTHER=value\nLOGS_HOST_PATH=\n"
    env_file.write_text(original)
    command = [
        sys.executable, str(REPOSITORY_ROOT / "scripts/prepare_compose_logs.py"),
        "--env-file", str(env_file),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    log_root = tmp_path / "infer-nexus-logs"
    for service in LOGGED_SERVICES:
        assert (log_root / service / "ray").is_dir()
    assert f"LOGS_HOST_PATH='{log_root}'" in env_file.read_text()
    assert "OTHER=value" in env_file.read_text()
    sentinel = log_root / "ray-head" / "operator-sentinel.txt"
    sentinel.write_text("old sentinel\n")
    saved_env = env_file.read_text()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert env_file.read_text() == saved_env
    assert sentinel.read_text() == "old sentinel\n"


def test_host_log_preparation_rejects_relative_paths_without_rewriting_env(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    original = "LOGS_HOST_PATH=./logs\n"
    env_file.write_text(original)
    result = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "scripts/prepare_compose_logs.py"),
         "--env-file", str(env_file)], capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "absolute path" in result.stderr
    assert env_file.read_text() == original
