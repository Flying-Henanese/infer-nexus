"""Contract tests for the KubeRay Serve-native public entrypoint."""

from pathlib import Path

import yaml

from infer_nexus.catalog.loader import load_model_catalog
from infer_nexus.core.config import load_settings


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_kubernetes_settings_explicitly_bound_the_gateway_ingress() -> None:
    """KubeRay must not inherit deployment capacity from Uvicorn defaults."""
    settings = load_settings(REPOSITORY_ROOT / "config/settings.k8s.yaml")

    assert settings.catalog.models_path == "config/models.k8s-smoke.yaml"
    assert settings.runtime.gateway_ingress.model_dump() == {
        "enabled": True,
        "application_name": "infer-nexus-gateway",
        "route_prefix": "/",
        "num_replicas": 1,
        "num_cpus": 0.5,
        "max_ongoing_requests": 16,
        "max_queued_requests": 32,
    }
    catalog = load_model_catalog(REPOSITORY_ROOT / settings.catalog.models_path)
    assert [
        (model.name, model.alias, model.min_replicas, model.max_replicas)
        for model in catalog.models
    ] == [
        ("Qwen3.5-9B", "qwen3.5-9b", 1, 1)
    ]


def test_stable_nodeport_targets_only_the_ray_head_serve_proxy() -> None:
    """The public Service must not reintroduce a standalone Uvicorn gateway."""
    manifest_path = REPOSITORY_ROOT / "deploy/kuberay/serve-gateway.service.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert manifest["kind"] == "Service"
    assert manifest["metadata"]["name"] == "infer-nexus-gateway"
    assert manifest["spec"]["type"] == "NodePort"
    assert manifest["spec"]["selector"] == {
        "ray.io/cluster": "infer-nexus-ray",
        "ray.io/node-type": "head",
    }
    assert manifest["spec"]["ports"] == [
        {"name": "http", "port": 8000, "targetPort": 8000, "nodePort": 30800}
    ]
    assert not (REPOSITORY_ROOT / "deploy/kuberay/gateway.yaml").exists()


def test_serve_deployer_declares_the_headonly_proxy_invariant() -> None:
    """A KubeRay rollout must not silently fall back to a disabled proxy."""
    manifest = (REPOSITORY_ROOT / "deploy/kuberay/serve-deployer.job.yaml").read_text(
        encoding="utf-8"
    )

    assert "--proxy-location HeadOnly" in manifest
