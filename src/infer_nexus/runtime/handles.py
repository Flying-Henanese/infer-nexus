"""提供 Ray Serve 部署句柄解析辅助工具。"""

from __future__ import annotations

from typing import Any

class ServeDeploymentHandleResolver:
    """描述运行时组件的数据或行为。"""

    def __init__(self, serve: Any | None = None) -> None:
        """初始化对象并保存运行时依赖。"""
        self._serve = serve

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
        serve = self.require_serve()
        return serve.get_deployment_handle(deployment_name, app_name=app_name)
