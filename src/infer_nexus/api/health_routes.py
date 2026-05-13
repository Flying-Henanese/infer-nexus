from fastapi import APIRouter

from infer_nexus.core.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse()


@router.get("/readyz", response_model=HealthResponse)
async def readyz() -> HealthResponse:
    return HealthResponse()
