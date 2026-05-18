"""OpenAI 兼容接口路由。"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from infer_nexus.api.deps import (
    get_admission_controller,
    get_model_store,
    get_registry,
    get_runtime_dispatcher,
)
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.control.admission import AdmissionController
from infer_nexus.core.enums import TaskType
from infer_nexus.core.errors import (
    AdmissionRejectedError,
    BackendRequestValidationError,
    ModelArtifactMissingError,
    ModelNotFoundError,
    RuntimeNotConnectedError,
)
from infer_nexus.core.schemas import (
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelListResponse,
    ModelSummary,
    OpenAIErrorDetail,
    OpenAIErrorResponse,
    RerankRequest,
    RerankResponse,
)
from infer_nexus.model_store import LocalModelStore
from infer_nexus.runtime.dispatcher import RuntimeDispatcher

router = APIRouter(prefix="/v1", tags=["openai"])
compat_router = APIRouter(tags=["openai"])


def openai_error_response(
    status_code: int,
    message: str,
    *,
    error_type: str,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    """构造 OpenAI 风格错误响应，统一网关错误返回格式。"""
    payload = OpenAIErrorResponse(
        error=OpenAIErrorDetail(message=message, type=error_type, param=param, code=code)
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


@router.get("/models", response_model=ModelListResponse)
async def list_models(registry: ModelRegistry = Depends(get_registry)) -> ModelListResponse:
    """列出对外可见模型清单（以 alias 优先作为展示 ID）。"""
    return ModelListResponse(
        data=[
            ModelSummary(
                id=model.alias or model.name,
                task=model.task,
                backend=model.backend,
                status=model.status,
                alias=model.alias,
            )
            for model in registry.list_models()
        ]
    )


@router.post("/chat/completions", response_model=ChatCompletionsResponse)
async def create_chat_completion(
    request: ChatCompletionsRequest,
    registry: ModelRegistry = Depends(get_registry),
    admission: AdmissionController = Depends(get_admission_controller),
    model_store: LocalModelStore = Depends(get_model_store),
    dispatcher: RuntimeDispatcher = Depends(get_runtime_dispatcher),
) -> ChatCompletionsResponse | JSONResponse:
    """处理聊天补全请求，执行模型校验、准入校验和运行时分发。"""
    # 第一步：检查请求模型是否已注册。
    try:
        model = registry.get(request.model)
    except ModelNotFoundError:
        return openai_error_response(
            404,
            f"The model '{request.model}' does not exist.",
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
        )

    # 第二步：检查模型任务类型，防止把 embedding/rerank 模型误用于 chat 接口。
    if model.task is not TaskType.CHAT:
        return openai_error_response(
            400,
            f"Model '{request.model}' does not support chat completions.",
            error_type="invalid_request_error",
            param="model",
            code="unsupported_task_type",
        )

    # 第三步：检查模型文件是否存在，并通过准入控制后进入运行时执行。
    try:
        model_store.require_model_path(model)
        admission.check_model_request(model)
        return await dispatcher.dispatch_chat(model, request)
    except ModelArtifactMissingError as exc:
        return openai_error_response(
            503,
            str(exc),
            error_type="service_unavailable_error",
            code="model_artifact_missing",
        )
    except AdmissionRejectedError as exc:
        return openai_error_response(
            429,
            str(exc),
            error_type="rate_limit_error",
            code=exc.code,
        )
    except RuntimeNotConnectedError as exc:
        return openai_error_response(
            501,
            str(exc),
            error_type="not_implemented_error",
            code=exc.code,
        )
    except BackendRequestValidationError as exc:
        return openai_error_response(
            400,
            str(exc),
            error_type="invalid_request_error",
            code=exc.code,
        )


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    registry: ModelRegistry = Depends(get_registry),
    admission: AdmissionController = Depends(get_admission_controller),
    model_store: LocalModelStore = Depends(get_model_store),
    dispatcher: RuntimeDispatcher = Depends(get_runtime_dispatcher),
) -> EmbeddingResponse | JSONResponse:
    """处理向量化请求，执行模型校验、准入校验和运行时分发。"""
    try:
        model = registry.get(request.model)
    except ModelNotFoundError:
        return openai_error_response(
            404,
            f"The model '{request.model}' does not exist.",
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
        )

    if model.task is not TaskType.EMBEDDING:
        return openai_error_response(
            400,
            f"Model '{request.model}' does not support embeddings.",
            error_type="invalid_request_error",
            param="model",
            code="unsupported_task_type",
        )

    try:
        model_store.require_model_path(model)
        admission.check_model_request(model)
        return await dispatcher.dispatch_embedding(model, request)
    except ModelArtifactMissingError as exc:
        return openai_error_response(
            503,
            str(exc),
            error_type="service_unavailable_error",
            code="model_artifact_missing",
        )
    except AdmissionRejectedError as exc:
        return openai_error_response(
            429,
            str(exc),
            error_type="rate_limit_error",
            code=exc.code,
        )
    except RuntimeNotConnectedError as exc:
        return openai_error_response(
            501,
            str(exc),
            error_type="not_implemented_error",
            code=exc.code,
        )
    except BackendRequestValidationError as exc:
        return openai_error_response(
            400,
            str(exc),
            error_type="invalid_request_error",
            code=exc.code,
        )


async def _create_rerank_impl(
    request: RerankRequest,
    registry: ModelRegistry,
    admission: AdmissionController,
    model_store: LocalModelStore,
    dispatcher: RuntimeDispatcher,
) -> RerankResponse | JSONResponse:
    """Rerank 共享实现，供 `/v1/rerank` 与兼容路径复用。"""
    try:
        model = registry.get(request.model)
    except ModelNotFoundError:
        return openai_error_response(
            404,
            f"The model '{request.model}' does not exist.",
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
        )

    if model.task is not TaskType.RERANK:
        return openai_error_response(
            400,
            f"Model '{request.model}' does not support rerank.",
            error_type="invalid_request_error",
            param="model",
            code="unsupported_task_type",
        )

    try:
        model_store.require_model_path(model)
        admission.check_model_request(model)
        return await dispatcher.dispatch_rerank(model, request)
    except ModelArtifactMissingError as exc:
        return openai_error_response(
            503,
            str(exc),
            error_type="service_unavailable_error",
            code="model_artifact_missing",
        )
    except AdmissionRejectedError as exc:
        return openai_error_response(
            429,
            str(exc),
            error_type="rate_limit_error",
            code=exc.code,
        )
    except RuntimeNotConnectedError as exc:
        return openai_error_response(
            501,
            str(exc),
            error_type="not_implemented_error",
            code=exc.code,
        )
    except BackendRequestValidationError as exc:
        return openai_error_response(
            400,
            str(exc),
            error_type="invalid_request_error",
            code=exc.code,
        )


@router.post("/rerank", response_model=RerankResponse)
@compat_router.post("/rerank", response_model=RerankResponse)
async def create_rerank(
    request: RerankRequest,
    registry: ModelRegistry = Depends(get_registry),
    admission: AdmissionController = Depends(get_admission_controller),
    model_store: LocalModelStore = Depends(get_model_store),
    dispatcher: RuntimeDispatcher = Depends(get_runtime_dispatcher),
) -> RerankResponse | JSONResponse:
    """处理 rerank 请求。"""
    return await _create_rerank_impl(request, registry, admission, model_store, dispatcher)
