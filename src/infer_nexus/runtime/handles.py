"""提供 Ray Serve 部署句柄解析辅助工具。"""

from __future__ import annotations

from typing import Any


def ensure_ray_initialized(address: str | None = None) -> None:
    """Connect the current process to Ray when it is not already connected."""
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError(
            "Ray is not installed. Install the 'serve' extra to enable handle execution."
        ) from exc

    if ray.is_initialized():
        return
    ray.init(address=address)


class ServeDeploymentHandleResolver:
    """描述运行时组件的数据或行为。"""

    def __init__(self, serve: Any | None = None, ray_address: str | None = None) -> None:
        """初始化对象并保存运行时依赖。"""
        self._serve = serve
        self._ray_address = ray_address
        self._handles: dict[tuple[str, str], Any] = {}

    def require_serve(self) -> Any:
        """执行运行时相关逻辑。"""
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
        """执行运行时相关逻辑。"""
        cache_key = (app_name, deployment_name)
        if cache_key in self._handles:
            return self._handles[cache_key]
        if self._serve is None:
            ensure_ray_initialized(self._ray_address)
        serve = self.require_serve()
        handle = serve.get_deployment_handle(deployment_name, app_name=app_name)
        self._handles[cache_key] = handle
        return handle
