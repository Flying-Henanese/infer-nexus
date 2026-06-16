"""OpenAI 兼容接口路由。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import json
import logging
from time import perf_counter
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from infer_nexus.api.deps import (
    get_admission_controller,
    get_model_store,
    get_registry,
    get_runtime_dispatcher,
)
from infer_nexus.catalog.models import ModelConfig
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
from infer_nexus.observability.metrics import GATEWAY_METRICS
from infer_nexus.runtime.dispatcher import RuntimeDispatcher

router = APIRouter(prefix="/v1", tags=["openai"])
compat_router = APIRouter(tags=["openai"])
logger = logging.getLogger(__name__)

UNKNOWN_MODEL_LABEL = "unknown"
RouteResponse = ChatCompletionsResponse | EmbeddingResponse | RerankResponse | JSONResponse | Response
RouteCallable = TypeVar("RouteCallable", bound=Callable[..., Awaitable[RouteResponse]])


@dataclass(slots=True)
class _RequestObservation:
    """保存单次 OpenAI 兼容请求的指标上下文。"""

    start_time: float
    task: str
    endpoint: str
    model: str = UNKNOWN_MODEL_LABEL
    inflight_started: bool = False


_current_observation: ContextVar[_RequestObservation | None] = ContextVar(
    "openai_request_observation",
    default=None,
)


def _metric_model_label(model: ModelConfig) -> str:
    """把模型配置归一化为有界 Prometheus 标签。"""
    return str(model.alias or model.served_model_name or model.name or UNKNOWN_MODEL_LABEL)


def _metric_task_label(task: TaskType | str) -> str:
    """返回稳定任务标签。"""
    return task.value if isinstance(task, TaskType) else str(task)


def _observe_resolved_model(model: ModelConfig) -> None:
    """在路由解析到有效模型后启动 inflight 指标。"""
    observation = _current_observation.get()
    if observation is None:
        return
    observation.model = _metric_model_label(model)
    if not observation.inflight_started:
        GATEWAY_METRICS.inc_inflight(
            model=observation.model,
            task=observation.task,
            endpoint=observation.endpoint,
        )
        observation.inflight_started = True


def _error_code_from_response(response: RouteResponse) -> str | None:
    """从 OpenAI 风格错误响应中读取稳定错误码。"""
    if not isinstance(response, JSONResponse):
        return None
    try:
        payload = json.loads(response.body.decode("utf-8"))
    except Exception:
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) and code else None


def _observe_request_result(observation: _RequestObservation, response: RouteResponse) -> None:
    """根据响应统一记录请求完成指标。"""
    status_code = getattr(response, "status_code", 200)
    error_code = _error_code_from_response(response)
    status = "error" if status_code >= 400 or error_code else "success"
    GATEWAY_METRICS.observe_request(
        model=observation.model,
        task=observation.task,
        endpoint=observation.endpoint,
        status=status,
        latency_seconds=perf_counter() - observation.start_time,
    )
    if error_code:
        GATEWAY_METRICS.observe_error(
            model=observation.model,
            task=observation.task,
            code=error_code,
        )
    if status_code == 429 and error_code:
        GATEWAY_METRICS.observe_admission_rejection(
            model=observation.model,
            reason=error_code,
        )
    GATEWAY_METRICS.observe_token_usage(
        model=observation.model,
        task=observation.task,
        usage=getattr(response, "usage", None),
    )


def observe_openai_request(task: TaskType, *, endpoint: str | None = None) -> Callable[[RouteCallable], RouteCallable]:
    """装饰 OpenAI 兼容路由，统一记录请求级 Prometheus 指标。"""

    def decorator(func: RouteCallable) -> RouteCallable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> RouteResponse:
            http_request = kwargs.get("http_request")
            path = endpoint
            if path is None and isinstance(http_request, Request):
                path = http_request.url.path
            observation = _RequestObservation(
                start_time=perf_counter(),
                task=_metric_task_label(task),
                endpoint=path or "unknown",
            )
            token = _current_observation.set(observation)
            try:
                response = await func(*args, **kwargs)
                _observe_request_result(observation, response)
                return response
            except Exception:
                GATEWAY_METRICS.observe_request(
                    model=observation.model,
                    task=observation.task,
                    endpoint=observation.endpoint,
                    status="error",
                    latency_seconds=perf_counter() - observation.start_time,
                )
                GATEWAY_METRICS.observe_error(
                    model=observation.model,
                    task=observation.task,
                    code="unhandled_exception",
                )
                raise
            finally:
                if observation.inflight_started:
                    GATEWAY_METRICS.dec_inflight(
                        model=observation.model,
                        task=observation.task,
                        endpoint=observation.endpoint,
                    )
                _current_observation.reset(token)

        return wrapper  # type: ignore[return-value]

    return decorator


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
    if code == "runtime_circuit_open":
        return 503, "service_unavailable_error"
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


def _resolve_model_for_task(
    request_model: str,
    registry: ModelRegistry,
    expected_task: TaskType,
    unsupported_message: str,
) -> ModelConfig | JSONResponse:
    """解析模型并校验任务类型，失败时返回 OpenAI 风格错误响应。"""
    try:
        model = registry.get(request_model)
    except ModelNotFoundError:
        return openai_error_response(
            404,
            f"The model '{request_model}' does not exist.",
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
        )

    _observe_resolved_model(model)
    if model.task is not expected_task:
        return openai_error_response(
            400,
            unsupported_message,
            error_type="invalid_request_error",
            param="model",
            code="unsupported_task_type",
        )
    return model


def _check_model_ready(
    model: ModelConfig,
    *,
    model_store: LocalModelStore,
    admission: AdmissionController,
) -> None:
    """检查本地模型产物与准入控制。"""
    if model.backend != BackendType.VLLM_OPENAI_PROXY:
        model_store.require_model_path(model)
    admission.check_model_request(model)


def _runtime_error_response(exc: Exception, *, operation: str, request_model: str) -> JSONResponse:
    """把运行前/运行时异常转换为 OpenAI 风格错误响应。"""
    if isinstance(exc, ModelArtifactMissingError):
        return openai_error_response(
            503,
            str(exc),
            error_type="service_unavailable_error",
            code="model_artifact_missing",
        )
    if isinstance(exc, AdmissionRejectedError):
        if exc.code == "gateway_overloaded":
            return openai_error_response(
                503,
                str(exc),
                error_type="service_unavailable_error",
                code=exc.code,
            )
        return openai_error_response(
            429,
            str(exc),
            error_type="rate_limit_error",
            code=exc.code,
        )
    if isinstance(exc, RuntimeNotConnectedError):
        status_code, error_type = runtime_not_connected_status(exc.code)
        return openai_error_response(
            status_code,
            str(exc),
            error_type=error_type,
            code=exc.code,
        )
    if isinstance(exc, RuntimeExecutionError):
        logger.exception("%s runtime execution failed for model '%s'.", operation, request_model)
        status_code, error_type = runtime_execution_status(exc.code)
        return openai_error_response(
            status_code,
            str(exc),
            error_type=error_type,
            code=exc.code,
        )
    if isinstance(exc, BackendRequestValidationError):
        return openai_error_response(
            400,
            str(exc),
            error_type="invalid_request_error",
            code=exc.code,
        )
    raise exc


@router.get("/models", response_model=ModelListResponse)
async def list_models(registry: ModelRegistry = Depends(get_registry)) -> ModelListResponse:
    """列出对外可见模型清单（以 alias 优先作为展示 ID，因为在请求时要使用alias指定模型）。"""
    return ModelListResponse(
        data=[
            ModelSummary(
                id=model.served_model_name or model.alias or model.name,
                task=model.task,
                backend=model.backend,
                compat_mode=model.compat_mode,
                status=model.status,
                alias=model.alias,
            )
            for model in registry.list_models()
        ]
    )


@router.post("/chat/completions", response_model=ChatCompletionsResponse)
@observe_openai_request(TaskType.CHAT, endpoint="/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionsRequest,
    registry: ModelRegistry = Depends(get_registry),
    admission: AdmissionController = Depends(get_admission_controller),
    model_store: LocalModelStore = Depends(get_model_store),
    dispatcher: RuntimeDispatcher = Depends(get_runtime_dispatcher),
) -> ChatCompletionsResponse | JSONResponse | Response:
    """处理聊天补全请求，执行模型校验、准入校验和运行时分发。"""
    model = _resolve_model_for_task(
        request.model,
        registry,
        TaskType.CHAT,
        f"Model '{request.model}' does not support chat completions.",
    )
    if isinstance(model, JSONResponse):
        return model

    try:
        _check_model_ready(model, model_store=model_store, admission=admission)
        return await dispatcher.dispatch_chat(model, request)
    except (
        ModelArtifactMissingError,
        AdmissionRejectedError,
        RuntimeNotConnectedError,
        RuntimeExecutionError,
        BackendRequestValidationError,
    ) as exc:
        return _runtime_error_response(exc, operation="Chat completion", request_model=request.model)


@router.post("/embeddings", response_model=EmbeddingResponse)
@observe_openai_request(TaskType.EMBEDDING, endpoint="/v1/embeddings")
async def create_embedding(
    request: EmbeddingRequest,
    registry: ModelRegistry = Depends(get_registry),
    admission: AdmissionController = Depends(get_admission_controller),
    model_store: LocalModelStore = Depends(get_model_store),
    dispatcher: RuntimeDispatcher = Depends(get_runtime_dispatcher),
) -> EmbeddingResponse | JSONResponse | Response:
    """处理向量化请求，执行模型校验、准入校验和运行时分发。"""
    model = _resolve_model_for_task(
        request.model,
        registry,
        TaskType.EMBEDDING,
        f"Model '{request.model}' does not support embeddings.",
    )
    if isinstance(model, JSONResponse):
        return model

    try:
        _check_model_ready(model, model_store=model_store, admission=admission)
        return await dispatcher.dispatch_embedding(model, request)
    except (
        ModelArtifactMissingError,
        AdmissionRejectedError,
        RuntimeNotConnectedError,
        RuntimeExecutionError,
        BackendRequestValidationError,
    ) as exc:
        return _runtime_error_response(exc, operation="Embedding", request_model=request.model)


async def _create_rerank_impl(
    request: RerankRequest,
    registry: ModelRegistry,
    admission: AdmissionController,
    model_store: LocalModelStore,
    dispatcher: RuntimeDispatcher,
) -> RerankResponse | JSONResponse | Response:
    """Rerank 共享实现，供 `/v1/rerank` 与兼容路径复用。"""
    model = _resolve_model_for_task(
        request.model,
        registry,
        TaskType.RERANK,
        f"Model '{request.model}' does not support rerank.",
    )
    if isinstance(model, JSONResponse):
        return model

    try:
        _check_model_ready(model, model_store=model_store, admission=admission)
        return await dispatcher.dispatch_rerank(model, request)
    except (
        ModelArtifactMissingError,
        AdmissionRejectedError,
        RuntimeNotConnectedError,
        RuntimeExecutionError,
        BackendRequestValidationError,
    ) as exc:
        return _runtime_error_response(exc, operation="Rerank", request_model=request.model)


@router.post("/rerank", response_model=RerankResponse)
@compat_router.post("/rerank", response_model=RerankResponse)
@observe_openai_request(TaskType.RERANK)
async def create_rerank(
    request: RerankRequest,
    http_request: Request,
    registry: ModelRegistry = Depends(get_registry),
    admission: AdmissionController = Depends(get_admission_controller),
    model_store: LocalModelStore = Depends(get_model_store),
    dispatcher: RuntimeDispatcher = Depends(get_runtime_dispatcher),
) -> RerankResponse | JSONResponse | Response:
    """处理 rerank 请求。"""
    return await _create_rerank_impl(request, registry, admission, model_store, dispatcher)
