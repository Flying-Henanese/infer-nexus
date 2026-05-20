"""Ray Serve deployment handle resolution helpers."""

from __future__ import annotations

from typing import Any


class ServeDeploymentHandleResolver:
    """Resolve Ray Serve deployment handles within a named Serve application."""

    def __init__(self, serve: Any | None = None) -> None:
        """初始化 Serve 句柄解析器。"""
        self._serve = serve

    def require_serve(self) -> Any:
        """Import or return injected Ray Serve runtime module."""
        if self._serve is not None:
            return self._serve
        try:
            from ray import serve
        except ImportError as exc:
            raise RuntimeError(
                "Ray Serve is not installed. Install the 'serve' extra to enable handle execution."
            ) from exc
        return serve

    def get_handle(self, deployment_name: str, *, app_name: str) -> Any:
        """Get a deployment handle scoped by configured Serve application name."""
        serve = self.require_serve()
        return serve.get_deployment_handle(deployment_name, app_name=app_name)
