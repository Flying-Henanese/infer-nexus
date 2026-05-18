from __future__ import annotations

"""Deploy infer-nexus model runtime into a Ray Serve application.

This script builds runtime deployments from configured model catalog entries and
publishes them under the configured service name.
"""

import argparse

from infer_nexus.catalog.loader import load_model_catalog
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.core.config import load_settings
from infer_nexus.model_store import LocalModelStore
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
    return parser.parse_args()


def main() -> None:
    """Initialize Ray, start Serve, and deploy the runtime application."""
    args = parse_args()

    settings = load_settings(args.settings)
    registry = ModelRegistry(load_model_catalog(settings.catalog.models_path))
    model_store = LocalModelStore.from_settings(settings.model_store)
    builder = ServeApplicationBuilder(
        model_store=model_store,
        backend_init_mode=settings.runtime.backend_init_mode,
    )
    builder.validate_registry_runtime_configs(registry)

    # Import lazily so config validation errors surface before Ray bootstrap.
    import ray
    from ray import serve

    runtime_env = {
        "working_dir": ".",
        "env_vars": {"RAY_RUNTIME_ENV_MODIFY_PYTHON_PATH": "0"},
    }
    ray.init(address=args.ray_address, runtime_env=runtime_env)
    serve.start(proxy_location=args.proxy_location)

    app = builder.build_serve_application(registry, serve=serve)
    serve.run(
        app,
        name=settings.service.name,
        route_prefix=None,
        blocking=args.blocking,
    )

    if not args.blocking:
        print(
            f"infer-nexus Serve runtime deployed as app '{settings.service.name}' "
            f"with proxy_location={args.proxy_location!r}"
        )


if __name__ == "__main__":
    main()
