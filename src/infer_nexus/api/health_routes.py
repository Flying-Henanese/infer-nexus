"""服务健康检查路由。"""

from fastapi import APIRouter

from infer_nexus.core.schemas import HealthResponse

router = APIRouter(tags=["health"])

# response_model 指定接口返回的响应 schema
# FastAPI 会对返回值做校验、过滤并序列化后返回给客户端
@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """存活探针：进程可响应即返回健康。"""
    return HealthResponse()


@router.get("/readyz", response_model=HealthResponse)
async def readyz() -> HealthResponse:
    """就绪探针：当前阶段与健康探针一致，后续可接入依赖检查。"""
    return HealthResponse()
