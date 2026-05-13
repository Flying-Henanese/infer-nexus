from fastapi import APIRouter, Depends

from infer_nexus.api.deps import get_load_inspector, get_registry
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.control.load_inspector import LoadInspector
from infer_nexus.core.schemas import CatalogModelResponse, ClusterLoadResponse, ModelStatusResponse

router = APIRouter(prefix="/api", tags=["platform"])


@router.get("/catalog/models", response_model=list[CatalogModelResponse])
async def list_catalog_models(registry: ModelRegistry = Depends(get_registry)) -> list[CatalogModelResponse]:
    return [CatalogModelResponse.model_validate(model.model_dump()) for model in registry.list_models()]


@router.get("/catalog/models/{model_name}", response_model=CatalogModelResponse)
async def get_catalog_model(
    model_name: str,
    registry: ModelRegistry = Depends(get_registry),
) -> CatalogModelResponse:
    model = registry.get(model_name)
    return CatalogModelResponse.model_validate(model.model_dump())


@router.get("/models/{model_name}/status", response_model=ModelStatusResponse)
async def get_model_status(
    model_name: str,
    registry: ModelRegistry = Depends(get_registry),
) -> ModelStatusResponse:
    model = registry.get(model_name)
    return ModelStatusResponse(name=model.name, status=model.status, message="runtime status not connected")


@router.get("/cluster/load", response_model=ClusterLoadResponse)
async def get_cluster_load(
    registry: ModelRegistry = Depends(get_registry),
    inspector: LoadInspector = Depends(get_load_inspector),
) -> ClusterLoadResponse:
    snapshot = inspector.snapshot(registry)
    return ClusterLoadResponse(
        status=snapshot.status,
        message=snapshot.message,
        active_models=snapshot.active_models,
    )


@router.get("/cluster/capacity", response_model=dict[str, str | int])
async def get_cluster_capacity(registry: ModelRegistry = Depends(get_registry)) -> dict[str, str | int]:
    return {
        "status": "stub",
        "message": "capacity backend not connected",
        "registered_models": len(registry.list_models()),
    }
