"""Deploy infer-nexus model runtime into a Ray Serve application.

This script builds runtime deployments from configured model catalog entries and
publishes them under the configured service name.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from infer_nexus.catalog.loader import load_model_catalog
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.core.config import load_settings
from infer_nexus.model_store import LocalModelStore
from infer_nexus.runtime.deployments import DeploymentFactory
from infer_nexus.runtime.serve_app import ServeApplicationBuilder


def parse_args() -> argparse.Namespace:
    """Parse Ray/Serve startup options for runtime deployment."""
    parser = argparse.ArgumentParser(description="Run infer-nexus Ray Serve model runtime.")
    parser.add_argument(
        "--settings",
        default="config/settings.yaml",
        help="Path to settings.yaml",
    )
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Ray address. Use 'auto' to connect to an existing local cluster.",
    )
    parser.add_argument(
        "--proxy-location",
        default="Disabled",
        help="Serve proxy location. Default disables Serve HTTP proxy to avoid port conflicts.",
    )
    parser.add_argument(
        "--blocking",
        action="store_true",
        help="Block and keep logging Serve application status.",
    )
    parser.add_argument(
        "--ready-timeout-seconds",
        type=float,
        default=600.0,
        help="How long to wait for the Serve application to become ready in non-blocking mode.",
    )
    parser.add_argument(
        "--ready-poll-interval-seconds",
        type=float,
        default=2.0,
        help="Polling interval while waiting for the Serve application to become ready.",
    )
    return parser.parse_args()


def _extract_status_value(status: Any) -> str | None:
    """Best-effort extraction of a Serve status enum/string value."""
    if status is None:
        return None
    value = getattr(status, "value", None)
    if isinstance(value, str):
        return value
    if isinstance(status, str):
        return status
    return str(status)


def _is_terminal_failure(status_value: str | None) -> bool:
    """Return whether a Serve status string indicates a failed terminal state."""
    if not status_value:
        return False
    normalized = status_value.upper()
    return normalized in {"DEPLOY_FAILED", "UNHEALTHY", "DELETING"}


def wait_for_serve_applications_ready(
    *,
    serve: Any,
    app_names: list[str],
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> None:
    """Wait until all named Serve applications and their deployments report healthy/running."""
    status_fn = getattr(serve, "status", None)
    if status_fn is None:
        return

    deadline = time.time() + timeout_seconds
    last_summary = "status unavailable"

    while time.time() < deadline:
        snapshot = status_fn()
        applications = getattr(snapshot, "applications", None) or {}
        app_summaries: list[str] = []
        ready_apps = 0

        for app_name in app_names:
            app_status = applications.get(app_name)
            if app_status is None:
                app_summaries.append(f"{app_name}=MISSING")
                continue

            app_status_value = _extract_status_value(getattr(app_status, "status", None))
            deployment_statuses = getattr(app_status, "deployments", None) or {}
            deployment_values = {
                name: _extract_status_value(getattr(deployment, "status", None))
                for name, deployment in deployment_statuses.items()
            }
            unhealthy = [
                f"{name}={value or 'UNKNOWN'}"
                for name, value in deployment_values.items()
                if value is None or value.upper() not in {"HEALTHY", "RUNNING"}
            ]
            app_summaries.append(
                f"{app_name}:app={app_status_value or 'UNKNOWN'},deployments={deployment_values}"
            )

            if _is_terminal_failure(app_status_value):
                raise RuntimeError(
                    f"Serve application '{app_name}' entered a failed state while deploying: "
                    f"{app_summaries[-1]}"
                )
            if any(_is_terminal_failure(value) for value in deployment_values.values()):
                raise RuntimeError(
                    f"Serve application '{app_name}' has unhealthy deployments: {app_summaries[-1]}"
                )
            if app_status_value and app_status_value.upper() in {"RUNNING", "HEALTHY"} and not unhealthy:
                ready_apps += 1

        last_summary = "; ".join(app_summaries) if app_summaries else "status unavailable"
        if ready_apps == len(app_names):
            return

        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Timed out after {timeout_seconds}s waiting for Serve applications to become ready. "
        f"Last observed state: {last_summary}"
    )


def main() -> None:
    """Initialize Ray, start Serve, and deploy the runtime application."""
    args = parse_args()

    settings = load_settings(args.settings)
    registry = ModelRegistry(load_model_catalog(settings.catalog.models_path))
    model_store = LocalModelStore.from_settings(settings.model_store)
    builder = ServeApplicationBuilder(
        model_store=model_store,
        backend_init_mode=settings.runtime.backend_init_mode,
        service_name=settings.service.name,
        deployment_factory=DeploymentFactory(
            inference_device_type=settings.cluster.inference_device_type,
        ),
    )
    builder.validate_registry_runtime_configs(registry)

    # Import lazily so config validation errors surface before Ray bootstrap.
    import ray
    from ray import serve

    runtime_env = {
        "working_dir": ".",
        # 排除这些文件，防止 Ray 自动触发环境构建逻辑
        "excludes": ["pyproject.toml", "uv.lock", ".venv", ".git"],
        "env_vars": {"RAY_RUNTIME_ENV_MODIFY_PYTHON_PATH": "0"},
    }
    ray.init(address=args.ray_address, runtime_env=runtime_env)
    serve.start(proxy_location=args.proxy_location)

    bindings = builder.build_serve_bindings(registry, serve=serve)
    app_names: list[str] = []
    for spec in builder.build_specs(registry):
        app_name = builder.build_application_name(spec.model_name)
        app_names.append(app_name)
        serve.run(
            bindings[spec.model_name],
            name=app_name,
            route_prefix=None,
            blocking=False,
        )

    wait_for_serve_applications_ready(
        serve=serve,
        app_names=app_names,
        timeout_seconds=args.ready_timeout_seconds,
        poll_interval_seconds=args.ready_poll_interval_seconds,
    )
    print(
        f"infer-nexus Serve runtime deployed {len(app_names)} application(s) "
        f"with proxy_location={args.proxy_location!r}"
    )

    if args.blocking:
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
