"""平台运维相关的只读 API 路由。

这一组接口面向控制台、运维面板或健康检查系统，主要提供三类信息：
1. 模型目录元数据：有哪些模型被注册、它们的部署/运行配置是什么。
2. 模型状态：模型定义是否存在、本地模型产物是否可用。
3. 集群视图：当前负载快照以及容量相关的占位信息。

这些接口不负责真正的推理请求转发，推理能力由 OpenAI 兼容路由提供。
"""

from fastapi import APIRouter, Depends, HTTPException

from infer_nexus.api.deps import get_load_inspector, get_model_store, get_registry
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.control.load_inspector import LoadInspector
from infer_nexus.core.errors import ModelArtifactMissingError, ModelNotFoundError
from infer_nexus.core.schemas import CatalogModelResponse, ClusterLoadResponse, ModelStatusResponse
from infer_nexus.model_store import LocalModelStore

router = APIRouter(prefix="/api", tags=["platform"])


def _get_model_or_404(model_name: str, registry: ModelRegistry):
    """按模型名或别名读取目录项，未命中时转换为 HTTP 404。

    Args:
        model_name: 路由中传入的模型名，既可以是 canonical name，也可以是 alias。
        registry: 进程内模型注册表，用于按名称解析模型配置。

    Returns:
        对应的模型配置对象。

    Raises:
        HTTPException: 当模型不存在时抛出 404。
    """
    try:
        return registry.get(model_name)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/catalog/models", response_model=list[CatalogModelResponse])
async def list_catalog_models(registry: ModelRegistry = Depends(get_registry)) -> list[CatalogModelResponse]:
    """列出全部已注册模型的目录声明。

    这是目录级别的“全量清单”接口，适合用于：
    - 页面初始化时加载模型列表；
    - 运维侧查看当前部署了哪些模型；
    - 调试模型注册表与配置文件是否一致。

    返回值会把内部的模型配置转换成 `CatalogModelResponse`，因此会包含模型名、
    别名、任务类型、后端类型、资源配置、部署配置以及当前状态等字段。
    """
    return [CatalogModelResponse.model_validate(model.model_dump()) for model in registry.list_models()]


@router.get("/catalog/models/{model_name}", response_model=CatalogModelResponse)
async def get_catalog_model(
    model_name: str,
    registry: ModelRegistry = Depends(get_registry),
) -> CatalogModelResponse:
    """获取单个模型的目录声明。

    该接口会优先按 canonical name 查找，也支持通过 alias 或 served_model_name
    间接命中对应模型。若模型不存在，会返回 HTTP 404。

    适合用于：
    - 进入模型详情页时按名称读取配置；
    - 检查某个模型的部署参数、资源参数和标签；
    - 做目录诊断时确认某个名字是否真的已注册。
    """
    model = _get_model_or_404(model_name, registry)
    return CatalogModelResponse.model_validate(model.model_dump())


@router.get("/models/{model_name}/status", response_model=ModelStatusResponse)
async def get_model_status(
    model_name: str,
    registry: ModelRegistry = Depends(get_registry),
    model_store: LocalModelStore = Depends(get_model_store),
) -> ModelStatusResponse:
    """获取单个模型的状态摘要与本地产物可用性。

    这个接口不是运行时心跳，而是“目录状态 + 本地文件状态”的合成视图：
    - 先从注册表中找到模型定义；
    - 再检查该模型所需的本地模型产物是否存在；
    - 如果产物缺失，则返回 `degraded`，并附带缺失原因；
    - 如果产物存在，则返回目录里记录的模型状态，同时提示本地路径已可用。

    适合用于：
    - 健康面板上展示某个模型是否“可用但降级”；
    - 启动后检查模型文件是否已经成功落盘；
    - 定位“配置存在但权重文件缺失”的问题。
    """
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
    """返回当前集群负载快照。

    这是一个阶段一的简化负载接口，当前仅基于注册模型数或运行侧可见信息
    生成一个轻量快照，不代表完整的调度或资源监控系统。

    通常用于：
    - 展示当前系统是否繁忙；
    - 快速判断活跃模型数量；
    - 为后续接入更完整的监控后端预留接口形状。
    """
    snapshot = inspector.snapshot(registry)
    return ClusterLoadResponse(
        status=snapshot.status,
        message=snapshot.message,
        active_models=snapshot.active_models,
    )


@router.get("/cluster/capacity", response_model=dict[str, str | int])
async def get_cluster_capacity(registry: ModelRegistry = Depends(get_registry)) -> dict[str, str | int]:
    """返回集群容量信息。

    当前这是一个占位接口，用于预留“容量后端”接入点。现在返回的内容只包含：
    - `status`: 固定为 `stub`，表示尚未接入真实容量系统；
    - `message`: 说明当前没有容量后端；
    - `registered_models`: 目录中已注册模型的数量。

    后续如果接入真实的容量、配额或资源池系统，这里可以平滑升级为真实指标。
    """
    return {
        "status": "stub",
        "message": "capacity backend not connected",
        "registered_models": len(registry.list_models()),
    }
