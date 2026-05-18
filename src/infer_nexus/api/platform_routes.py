"""平台运维接口路由（目录、状态、负载）。"""

from fastapi import APIRouter, Depends, HTTPException

from infer_nexus.api.deps import get_load_inspector, get_model_store, get_registry
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.control.load_inspector import LoadInspector
from infer_nexus.core.errors import ModelArtifactMissingError, ModelNotFoundError
from infer_nexus.core.schemas import CatalogModelResponse, ClusterLoadResponse, ModelStatusResponse
from infer_nexus.model_store import LocalModelStore

router = APIRouter(prefix="/api", tags=["platform"])


def _get_model_or_404(model_name: str, registry: ModelRegistry):
    """按模型名读取模型，未命中时转换为 HTTP 404。"""
    try:
        return registry.get(model_name)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/catalog/models", response_model=list[CatalogModelResponse])
async def list_catalog_models(registry: ModelRegistry = Depends(get_registry)) -> list[CatalogModelResponse]:
    """返回目录中的全部模型声明。"""
    return [CatalogModelResponse.model_validate(model.model_dump()) for model in registry.list_models()]


@router.get("/catalog/models/{model_name}", response_model=CatalogModelResponse)
async def get_catalog_model(
    model_name: str,
    registry: ModelRegistry = Depends(get_registry),
) -> CatalogModelResponse:
    """返回单个目录模型声明。"""
    model = _get_model_or_404(model_name, registry)
    return CatalogModelResponse.model_validate(model.model_dump())


@router.get("/models/{model_name}/status", response_model=ModelStatusResponse)
async def get_model_status(
    model_name: str,
    registry: ModelRegistry = Depends(get_registry),
    model_store: LocalModelStore = Depends(get_model_store),
) -> ModelStatusResponse:
    """返回模型状态与本地模型文件可用性。"""
    model = _get_model_or_404(model_name, registry)
    try:
        resolved_path = model_store.require_model_path(model)
    except ModelArtifactMissingError as exc:
        return ModelStatusResponse(name=model.name, status="degraded", message=str(exc))
    return ModelStatusResponse(
        name=model.name,
        status=model.status,
        message=f"runtime status not connected; local model path is present at {resolved_path}",
    )


@router.get("/cluster/load", response_model=ClusterLoadResponse)
async def get_cluster_load(
    registry: ModelRegistry = Depends(get_registry),
    inspector: LoadInspector = Depends(get_load_inspector),
) -> ClusterLoadResponse:
    """返回当前集群负载快照（阶段一为简化指标）。"""
    snapshot = inspector.snapshot(registry)
    return ClusterLoadResponse(
        status=snapshot.status,
        message=snapshot.message,
        active_models=snapshot.active_models,
    )


@router.get("/cluster/capacity", response_model=dict[str, str | int])
async def get_cluster_capacity(registry: ModelRegistry = Depends(get_registry)) -> dict[str, str | int]:
    """返回容量信息（当前为占位实现）。"""
    return {
        "status": "stub",
        "message": "capacity backend not connected",
        "registered_models": len(registry.list_models()),
    }
