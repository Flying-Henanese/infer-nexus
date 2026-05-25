"""OpenAI 兼容接口路由。"""

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from starlette.responses import Response

from infer_nexus.api.deps import (
    get_admission_controller,
    get_model_store,
    get_registry,
    get_runtime_dispatcher,
)
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.control.admission import AdmissionController
from infer_nexus.core.enums import BackendType, TaskType
from infer_nexus.core.errors import (
    AdmissionRejectedError,
    BackendRequestValidationError,
    ModelArtifactMissingError,
    ModelNotFoundError,
    RuntimeExecutionError,
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
logger = logging.getLogger(__name__)


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


def runtime_not_connected_status(code: str) -> tuple[int, str]:
    """Map gateway-stage runtime errors to stable OpenAI-style response classes."""
    if code == "upstream_timeout":
        return 504, "service_unavailable_error"
    if code == "backend_misconfigured":
        return 500, "internal_server_error"
    if code == "unsupported_parameter":
        return 400, "invalid_request_error"
    return 501, "not_implemented_error"


def runtime_execution_status(code: str) -> tuple[int, str]:
    """Map backend execution failures to API-facing status classes when they reflect request shape."""
    if code in {"unsupported_parameter", "unsupported_message_content", "invalid_input"}:
        return 400, "invalid_request_error"
    return 500, "internal_server_error"


@router.get("/models", response_model=ModelListResponse)
async def list_models(registry: ModelRegistry = Depends(get_registry)) -> ModelListResponse:
    """列出对外可见模型清单（以 alias 优先作为展示 ID）。"""
    return ModelListResponse(
        data=[
            ModelSummary(
                id=model.served_model_name or model.alias or model.name,
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
) -> ChatCompletionsResponse | JSONResponse | Response:
    """处理聊天补全请求，执行模型校验、准入校验和运行时分发。"""
    # 第一步：检查请求模型是否已注册。
    try:
        model = registry.get(request.model)
    # 如果找不到模型，则向接口返回 OpenAI 兼容的 404 错误响应
    except ModelNotFoundError:
        return openai_error_response(
            404,
            f"The model '{request.model}' does not exist.",
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
        )

    # 第二步：检查模型任务类型
    # 防止把 embedding/rerank 模型误用于当前链路中的chat 接口。
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
        # 对于非 VLLM_OPENAI_PROXY 后端的模型，才检查模型文件路径是否存在。
        # 因为 VLLM_OPENAI_PROXY 后端的模型是通过代理转发到外部 OpenAI API 的，
        # 不涉及本地模型文件，所以不需要检查模型路径。
        if model.backend != BackendType.VLLM_OPENAI_PROXY:
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
        status_code, error_type = runtime_not_connected_status(exc.code)
        return openai_error_response(
            status_code,
            str(exc),
            error_type=error_type,
            code=exc.code,
        )
    except RuntimeExecutionError as exc:
        logger.exception("Chat completion runtime execution failed for model '%s'.", request.model)
        status_code, error_type = runtime_execution_status(exc.code)
        return openai_error_response(
            status_code,
            str(exc),
            error_type=error_type,
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
) -> EmbeddingResponse | JSONResponse | Response:
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
        if model.backend != BackendType.VLLM_OPENAI_PROXY:
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
        status_code, error_type = runtime_not_connected_status(exc.code)
        return openai_error_response(
            status_code,
            str(exc),
            error_type=error_type,
            code=exc.code,
        )
    except RuntimeExecutionError as exc:
        logger.exception("Embedding runtime execution failed for model '%s'.", request.model)
        status_code, error_type = runtime_execution_status(exc.code)
        return openai_error_response(
            status_code,
            str(exc),
            error_type=error_type,
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
) -> RerankResponse | JSONResponse | Response:
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
        if model.backend != BackendType.VLLM_OPENAI_PROXY:
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
        status_code, error_type = runtime_not_connected_status(exc.code)
        return openai_error_response(
            status_code,
            str(exc),
            error_type=error_type,
            code=exc.code,
        )
    except RuntimeExecutionError as exc:
        logger.exception("Rerank runtime execution failed for model '%s'.", request.model)
        status_code, error_type = runtime_execution_status(exc.code)
        return openai_error_response(
            status_code,
            str(exc),
            error_type=error_type,
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
) -> RerankResponse | JSONResponse | Response:
    """处理 rerank 请求。"""
    return await _create_rerank_impl(request, registry, admission, model_store, dispatcher)
