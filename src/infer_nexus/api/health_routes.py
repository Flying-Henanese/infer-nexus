"""服务健康检查路由。"""

from fastapi import APIRouter, HTTPException, Request

from infer_nexus.core.schemas import HealthResponse

router = APIRouter(tags=["health"])

# response_model 指定接口返回的响应 schema
# FastAPI 会对返回值做校验、过滤并序列化后返回给客户端
@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """存活探针：进程可响应即返回健康。"""
    return HealthResponse()


@router.get("/readyz", response_model=HealthResponse)
async def readyz(request: Request) -> HealthResponse:
    """Readiness requires every configured local model Serve app to be healthy."""
    readiness_checker = getattr(request.app.state, "readiness_checker", None)
    if readiness_checker is not None and not readiness_checker():
        raise HTTPException(status_code=503, detail="one or more model Serve applications are not ready")
    return HealthResponse()
