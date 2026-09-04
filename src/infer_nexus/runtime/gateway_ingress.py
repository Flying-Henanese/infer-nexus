"""Ray Serve-native public HTTP gateway binding.

The deployment wraps the existing FastAPI API surface. Its lifecycle builds a
``GatewayRuntime`` inside the Serve replica, so model handles remain on Ray
Serve's internal data plane and never cross a Ray Client connection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infer_nexus.core.config import Settings


@dataclass(frozen=True, slots=True)
class GatewayIngressSpec:
    """Static configuration for the one public Serve application."""

    application_name: str
    deployment_name: str
    route_prefix: str
    num_replicas: int
    num_cpus: float
    max_ongoing_requests: int
    max_queued_requests: int
    model_targets: dict[str, dict[str, str]]


class InferNexusGatewayIngress:
    """Marker class adapted into an ASGI deployment by ``serve.ingress``."""


def build_gateway_ingress_spec(
    *,
    settings: Settings,
    service_name: str,
    model_targets: dict[str, dict[str, str]],
) -> GatewayIngressSpec:
    """Validate and normalize the public ingress deployment configuration."""
    config = settings.runtime.gateway_ingress
    if not config.enabled:
        raise ValueError("The Serve-native gateway ingress is disabled by configuration.")
    application_name = config.application_name or f"{service_name}-gateway"
    if not config.route_prefix.startswith("/"):
        raise ValueError("Gateway ingress route_prefix must start with '/'.")
    return GatewayIngressSpec(
        application_name=application_name,
        deployment_name="gateway",
        route_prefix=config.route_prefix,
        num_replicas=config.num_replicas,
        num_cpus=config.num_cpus,
        max_ongoing_requests=config.max_ongoing_requests,
        max_queued_requests=config.max_queued_requests,
        model_targets=model_targets,
    )


def build_gateway_ingress_binding(*, serve: Any, settings: Settings, spec: GatewayIngressSpec) -> Any:
    """Build an ASGI Serve binding without initializing a standalone Ray client."""
    # Import lazily to keep the dependency assembly separate from Serve's
    # binding builder and avoid a module-import cycle.
    from infer_nexus.main import create_app

    app = create_app(
        settings=settings,
        model_targets=spec.model_targets,
        connect_ray=False,
    )
    # Keep an inspectable, config-derived route table alongside the runtime's
    # precomputed targets. Client requests never provide these Ray identifiers.
    app.state.gateway_model_targets = dict(spec.model_targets)
    ingress_class = serve.ingress(app)(InferNexusGatewayIngress)
    deployment = serve.deployment(
        name=spec.deployment_name,
        num_replicas=spec.num_replicas,
        max_ongoing_requests=spec.max_ongoing_requests,
        max_queued_requests=spec.max_queued_requests,
        ray_actor_options={"num_cpus": spec.num_cpus},
    )(ingress_class)
    return deployment.bind()
