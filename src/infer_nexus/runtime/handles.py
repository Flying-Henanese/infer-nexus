from __future__ import annotations

from typing import Any


class ServeDeploymentHandleResolver:
    def __init__(self, app_name: str, serve: Any | None = None) -> None:
        self.app_name = app_name
        self._serve = serve

    def require_serve(self) -> Any:
        if self._serve is not None:
            return self._serve
        try:
            from ray import serve
        except ImportError as exc:
            raise RuntimeError(
                "Ray Serve is not installed. Install the 'serve' extra to enable handle execution."
            ) from exc
        return serve

    def get_handle(self, deployment_name: str) -> Any:
        serve = self.require_serve()
        return serve.get_deployment_handle(deployment_name, app_name=self.app_name)
