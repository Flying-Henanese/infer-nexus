"""Regression coverage for host-visible Docker Compose logs."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOGGED_SERVICES = ("ray-head", "ray-worker", "serve-deployer")
COMPOSE_PROFILES = (
    ("docker-compose.yml", "./logs", "sh"),
    ("ascend_deploy/docker-compose.yml", "../logs", "bash"),
)


def _volume_sources(service: dict[str, object], container_target: str) -> list[str]:
    sources: list[str] = []
    for volume in service["volumes"]:
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

    log_initializer = compose["services"]["log-init"]
    initializer_command = " ".join(log_initializer["command"])
    initializer_directory = "/logs/log-init"
    initializer_redirect = f"exec >>{initializer_directory}/container.log"
    assert initializer_directory in initializer_command
    assert initializer_command.index(initializer_directory) < initializer_command.index(
        initializer_redirect
    ) < initializer_command.index("for service in ray-head ray-worker serve-deployer")
    assert _volume_sources(log_initializer, "/logs") == [log_root]

    for service_name in LOGGED_SERVICES:
        service = compose["services"][service_name]
        command = " ".join(service["command"])
        assert service["depends_on"]["log-init"]["condition"] == "service_completed_successfully"
        assert _volume_sources(service, "/var/log/infer-nexus") == [
            f"{log_root}/{service_name}"
        ]
        assert _volume_sources(service, "/tmp/ray") == [
            f"{log_root}/{service_name}/ray"
        ]
        assert (
            "sh /app/scripts/container/run_with_log.sh "
            "/var/log/infer-nexus/container.log --"
        ) in command

        if service_name == "ray-head":
            continue

        marker = f" -- {nested_shell} -ec '"
        nested_command = service["command"][-1]
        nested_start = nested_command.index(marker) + len(marker)
        nested_script = nested_command[nested_start:].rstrip().replace("$$", "$")
        assert nested_script.endswith("'")
        syntax_check = subprocess.run(
            [nested_shell, "-n", "-c", nested_script[:-1]],
            check=False,
            capture_output=True,
            text=True,
        )
        assert syntax_check.returncode == 0, syntax_check.stderr


def test_container_log_wrapper_writes_combined_output_to_the_host_log(tmp_path: Path) -> None:
    """The wrapper preserves both streams in one host-mounted container log."""
    log_path = tmp_path / "service" / "container.log"
    result = subprocess.run(
        [
            "sh",
            str(REPOSITORY_ROOT / "scripts/container/run_with_log.sh"),
            str(log_path),
            "--",
            "sh",
            "-c",
            "printf stdout; printf stderr >&2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert log_path.read_text(encoding="utf-8") == "stdoutstderr"
