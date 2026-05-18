"""服务健康检查路由。"""

from fastapi import APIRouter

from infer_nexus.core.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """存活探针：进程可响应即返回健康。"""
    return HealthResponse()


@router.get("/readyz", response_model=HealthResponse)
async def readyz() -> HealthResponse:
    """就绪探针：当前阶段与健康探针一致，后续可接入依赖检查。"""
    return HealthResponse()
