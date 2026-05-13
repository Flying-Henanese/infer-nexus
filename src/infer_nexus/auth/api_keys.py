from collections.abc import Callable

from fastapi import Header, HTTPException, status


def build_api_key_dependency(api_key_header: str) -> Callable[..., str | None]:
    async def require_api_key(x_api_key: str | None = Header(default=None, alias=api_key_header)) -> str | None:
        # Phase 1 skeleton: auth is optional until concrete key storage is added.
        return x_api_key

    return require_api_key
