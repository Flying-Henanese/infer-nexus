"""API Key 认证依赖构建。"""

from collections.abc import Callable

from fastapi import Header, HTTPException, status


def build_api_key_dependency(api_key_header: str) -> Callable[..., str | None]:
    """构建读取请求头 API Key 的 FastAPI 依赖函数。"""

    async def require_api_key(x_api_key: str | None = Header(default=None, alias=api_key_header)) -> str | None:
        """读取并返回 API Key（阶段一保持可选）。"""
        # Phase 1 skeleton: auth is optional until concrete key storage is added.
        return x_api_key

    return require_api_key
