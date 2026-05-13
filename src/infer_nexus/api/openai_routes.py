from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from infer_nexus.api.deps import get_admission_controller, get_registry
from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.control.admission import AdmissionController
from infer_nexus.core.enums import TaskType
from infer_nexus.core.errors import AdmissionRejectedError, ModelNotFoundError
from infer_nexus.core.schemas import (
    ChatCompletionsRequest,
    EmbeddingRequest,
    ModelListResponse,
    ModelSummary,
    OpenAIErrorDetail,
    OpenAIErrorResponse,
)

router = APIRouter(prefix="/v1", tags=["openai"])


def openai_error_response(
    status_code: int,
    message: str,
    *,
    error_type: str,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    payload = OpenAIErrorResponse(
        error=OpenAIErrorDetail(message=message, type=error_type, param=param, code=code)
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


@router.get("/models", response_model=ModelListResponse)
async def list_models(registry: ModelRegistry = Depends(get_registry)) -> ModelListResponse:
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


@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionsRequest,
    registry: ModelRegistry = Depends(get_registry),
    admission: AdmissionController = Depends(get_admission_controller),
) -> JSONResponse:
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

    if model.task is not TaskType.CHAT:
        return openai_error_response(
            400,
            f"Model '{request.model}' does not support chat completions.",
            error_type="invalid_request_error",
            param="model",
            code="unsupported_task_type",
        )

    try:
        admission.check_model_request(model)
    except AdmissionRejectedError as exc:
        return openai_error_response(
            429,
            str(exc),
            error_type="rate_limit_error",
            code=exc.code,
        )

    return openai_error_response(
        501,
        "Chat completions runtime is not connected yet.",
        error_type="not_implemented_error",
        code="runtime_not_connected",
    )


@router.post("/embeddings")
async def create_embedding(
    request: EmbeddingRequest,
    registry: ModelRegistry = Depends(get_registry),
    admission: AdmissionController = Depends(get_admission_controller),
) -> JSONResponse:
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
        admission.check_model_request(model)
    except AdmissionRejectedError as exc:
        return openai_error_response(
            429,
            str(exc),
            error_type="rate_limit_error",
            code=exc.code,
        )

    return openai_error_response(
        501,
        "Embeddings runtime is not connected yet.",
        error_type="not_implemented_error",
        code="runtime_not_connected",
    )
