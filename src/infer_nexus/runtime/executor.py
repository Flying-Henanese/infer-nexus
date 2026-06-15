"""连接 API 请求、模型部署目标和实际推理执行路径。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
import inspect
import json
import logging
import os
from time import perf_counter, time
from typing import Any
from uuid import uuid4

import httpx
from starlette.responses import JSONResponse, Response, StreamingResponse

from infer_nexus.catalog.models import ProxyConfig
from infer_nexus.core.enums import BackendType, CompatibilityMode
from infer_nexus.core.errors import (
    AdmissionRejectedError,
    BackendConfigurationError,
    RuntimeExecutionError,
    RuntimeNotConnectedError,
)
from infer_nexus.core.schemas import (
    ChatCompletionChoice,
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    ChatMessage,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
    TokenUsage,
)
from infer_nexus.observability.metrics import GATEWAY_METRICS
from infer_nexus.runtime.deployments import ModelRuntimeReplica
from infer_nexus.runtime.handles import ServeDeploymentHandleResolver
from infer_nexus.runtime.types import RuntimeTarget

logger = logging.getLogger(__name__)


class _ServeDeploymentGuard:
    """Gateway-side fail-fast guard for one Serve deployment."""

    def __init__(
        self,
        *,
        key: str,
        max_inflight: int,
        circuit_breaker_enabled: bool,
        failure_threshold: int,
        cooldown_seconds: int | float,
    ) -> None:
        self.key = key
        self.max_inflight = max_inflight
        self.circuit_breaker_enabled = circuit_breaker_enabled
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._inflight = 0
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def check_circuit(self) -> None:
        """Reject requests while the deployment circuit is open."""
        if not self.circuit_breaker_enabled or self._opened_at is None:
            return
        elapsed = time() - self._opened_at
        if elapsed < self.cooldown_seconds:
            remaining = self.cooldown_seconds - elapsed
            raise RuntimeNotConnectedError(
                f"Serve deployment '{self.key}' circuit breaker is open; "
                f"retry after {remaining:.1f}s.",
                code="runtime_circuit_open",
            )
        self._opened_at = None

    async def acquire(self) -> None:
        """Acquire one gateway-side inflight slot or fail fast."""
        self.check_circuit()
        if self.max_inflight > 0 and self._inflight >= self.max_inflight:
            raise AdmissionRejectedError(
                f"Serve deployment '{self.key}' has reached the gateway inflight limit "
                f"({self.max_inflight}).",
                code="model_overloaded",
            )
        self._inflight += 1

    def release(self) -> None:
        """Release one gateway-side inflight slot."""
        if self._inflight > 0:
            self._inflight -= 1

    def record_success(self) -> None:
        """Close the breaker after a successful request."""
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Open the breaker after enough consecutive runtime failures."""
        if not self.circuit_breaker_enabled:
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._opened_at = time()


@dataclass(slots=True)
class RuntimeExecutor:
    """统一执行本地副本、Ray Serve 句柄和 OpenAI 代理模型请求。"""

    mode: str = "stub"
    handle_resolver: ServeDeploymentHandleResolver | None = None
    serve_request_timeout_seconds: int | float = 120
    max_inflight_per_model: int = 0
    circuit_breaker_enabled: bool = False
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_cooldown_seconds: int | float = 60
    _guards: dict[str, _ServeDeploymentGuard] = field(default_factory=dict, init=False)

    def _guard_key(self, target: RuntimeTarget) -> str:
        """Return the stable key used for per-deployment gateway safeguards."""
        return f"{target.app_name or 'unknown'}:{target.deployment_name}"

    def _get_guard(self, target: RuntimeTarget) -> "_ServeDeploymentGuard":
        """Return the per-deployment fail-fast guard."""
        key = self._guard_key(target)
        guard = self._guards.get(key)
        if guard is None:
            guard = _ServeDeploymentGuard(
                key=key,
                max_inflight=self.max_inflight_per_model,
                circuit_breaker_enabled=self.circuit_breaker_enabled,
                failure_threshold=self.circuit_breaker_failure_threshold,
                cooldown_seconds=self.circuit_breaker_cooldown_seconds,
            )
            self._guards[key] = guard
        return guard

    async def execute_chat(
        self,
        *,
        target: RuntimeTarget,
        request: ChatCompletionsRequest,
    ) -> ChatCompletionsResponse | Response:
        """执行 chat completion，并按模型后端选择代理、Serve 或本地路径。"""
        if target.backend == BackendType.VLLM_OPENAI_PROXY:
            return await self._execute_proxy_chat(target=target, request=request)
        if request.stream:
            return await self._execute_chat_stream(target=target, request=request)

        # serve 模式走远程句柄；stub 模式本地实例化副本，便于本地开发与测试。
        if self.mode == "serve":
            payload = await self._invoke_handle(
                target=target,
                method_name="chat_completion",
                payload=request.model_dump(mode="json"),
            )
        else:
            payload = await self._invoke_local_replica(
                target=target,
                method_name="chat_completion",
                payload=request.model_dump(mode="json"),
            )
        if self._is_openai_chat_payload(payload):
            return JSONResponse(content=self._strip_internal_status(payload))
        return self._build_chat_response_from_payload(request, target, payload)

    async def _execute_chat_stream(
        self,
        *,
        target: RuntimeTarget,
        request: ChatCompletionsRequest,
    ) -> StreamingResponse:
        """执行流式 chat completion，并在返回响应前预取首块暴露早期错误。"""
        stream_start_time = perf_counter()
        if self.mode == "serve":
            chunks = await self._invoke_handle_stream(
                target=target,
                method_name="chat_completion_stream",
                payload=request.model_dump(mode="json"),
            )
        else:
            chunks = await self._invoke_local_replica_stream(
                target=target,
                method_name="chat_completion_stream",
                payload=request.model_dump(mode="json"),
            )
        # Prime one chunk before sending response headers so unsupported streaming requests
        # fail as JSON errors instead of returning a broken 200 SSE connection.
        chunks = await self._prime_stream_chunks(chunks)
        return self._build_chat_stream_response(
            request,
            target,
            chunks,
            stream_start_time=stream_start_time,
        )

    async def execute_embedding(
        self,
        *,
        target: RuntimeTarget,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse | Response:
        """执行 embedding 请求，并把后端载荷整理为 OpenAI 兼容响应。"""
        if target.backend == BackendType.VLLM_OPENAI_PROXY:
            return await self._execute_proxy_embedding(target=target, request=request)
        if self.mode == "serve":
            payload = await self._invoke_handle(
                target=target,
                method_name="embedding",
                payload=request.model_dump(mode="json"),
            )
        else:
            payload = await self._invoke_local_replica(
                target=target,
                method_name="embedding",
                payload=request.model_dump(mode="json"),
            )
        return self._build_embedding_response_from_payload(request, target, payload)

    async def execute_rerank(
        self,
        *,
        target: RuntimeTarget,
        request: RerankRequest,
    ) -> RerankResponse | Response:
        """执行 rerank 请求，并把后端载荷整理为统一 rerank 响应。"""
        if target.backend == BackendType.VLLM_OPENAI_PROXY:
            return await self._execute_proxy_rerank(target=target, request=request)
        if self.mode == "serve":
            payload = await self._invoke_handle(
                target=target,
                method_name="rerank",
                payload=request.model_dump(mode="json"),
            )
        else:
            payload = await self._invoke_local_replica(
                target=target,
                method_name="rerank",
                payload=request.model_dump(mode="json"),
            )
        return self._build_rerank_response_from_payload(request, target, payload)

    async def _invoke_local_replica(
        self,
        *,
        target: RuntimeTarget,
        method_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """在 stub/dev 模式下直接实例化副本并调用指定方法。"""
        # Local path is used for stub/dev mode without requiring Ray Serve connectivity.
        try:
            replica = ModelRuntimeReplica(target.runtime_context)
            method = getattr(replica, method_name)
            return await method(request_payload=payload)
        except BackendConfigurationError as exc:
            raise RuntimeExecutionError(str(exc), code="backend_misconfigured") from exc

    async def _invoke_local_replica_stream(
        self,
        *,
        target: RuntimeTarget,
        method_name: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
        """在 stub/dev 模式下调用本地副本的流式方法并归一化结果。"""
        try:
            replica = ModelRuntimeReplica(target.runtime_context)
            method = getattr(replica, method_name)
            return self._normalize_stream_result(method(request_payload=payload))
        except BackendConfigurationError as exc:
            raise RuntimeExecutionError(str(exc), code="backend_misconfigured") from exc

    async def _invoke_handle(
        self,
        *,
        target: RuntimeTarget,
        method_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """在 serve 模式下解析部署句柄并远程调用副本方法。"""
        # Serve mode relies on model-specific deployment handles resolved by app name + deployment name.
        if self.handle_resolver is None:
            raise RuntimeNotConnectedError(
                f"Runtime executor is configured for serve mode but no handle resolver is available "
                f"for deployment '{target.deployment_name}'."
            )

        try:
            if not target.app_name:
                raise RuntimeNotConnectedError(
                    f"Runtime target for deployment '{target.deployment_name}' does not define a Serve app name.",
                    code="backend_misconfigured",
                )
            handle = self.handle_resolver.get_handle(
                target.deployment_name,
                app_name=target.app_name,
            )
        except RuntimeNotConnectedError:
            raise
        except Exception as exc:
            raise RuntimeNotConnectedError(
                f"Failed to resolve Serve handle for deployment '{target.deployment_name}': {exc}"
            ) from exc

        remote_method = getattr(handle, method_name, None)
        if remote_method is None or not hasattr(remote_method, "remote"):
            raise RuntimeNotConnectedError(
                f"Serve handle for deployment '{target.deployment_name}' does not expose "
                f"'{method_name}.remote(...)'."
            )

        try:
            guard = self._get_guard(target)
            await guard.acquire()
            response = remote_method.remote(request_payload=payload)
            try:
                result = await asyncio.wait_for(
                    self._await_handle_response(response),
                    timeout=self.serve_request_timeout_seconds,
                )
            except TimeoutError as exc:
                guard.record_failure()
                raise self._serve_timeout_error(target, method_name) from exc
            except Exception:
                guard.record_failure()
                raise
            else:
                guard.record_success()
                return result
            finally:
                guard.release()
        except AdmissionRejectedError:
            raise
        except RuntimeNotConnectedError:
            raise
        except Exception as exc:
            code = self._extract_execution_error_code(exc)
            message = self._extract_execution_error_message(exc)
            logger.exception(
                "Serve execution failed for deployment '%s' in app '%s' method '%s'.",
                target.deployment_name,
                target.app_name,
                method_name,
            )
            if code in {"unsupported_parameter", "unsupported_message_content", "invalid_input"}:
                raise RuntimeExecutionError(message, code=code) from exc
            raise RuntimeExecutionError(
                f"Serve execution failed for deployment '{target.deployment_name}' "
                f"in app '{target.app_name}' method '{method_name}': {message}",
                code=code,
            ) from exc

    async def _invoke_handle_stream(
        self,
        *,
        target: RuntimeTarget,
        method_name: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
        """在 serve 模式下通过部署句柄调用远程流式副本方法。"""
        if self.handle_resolver is None:
            raise RuntimeNotConnectedError(
                f"Runtime executor is configured for serve mode but no handle resolver is available "
                f"for deployment '{target.deployment_name}'."
            )

        try:
            if not target.app_name:
                raise RuntimeNotConnectedError(
                    f"Runtime target for deployment '{target.deployment_name}' does not define a Serve app name.",
                    code="backend_misconfigured",
                )
            handle = self.handle_resolver.get_handle(
                target.deployment_name,
                app_name=target.app_name,
            )
        except RuntimeNotConnectedError:
            raise
        except Exception as exc:
            raise RuntimeNotConnectedError(
                f"Failed to resolve Serve handle for deployment '{target.deployment_name}': {exc}"
            ) from exc

        remote_method = getattr(handle, method_name, None)
        if remote_method is None or not hasattr(remote_method, "remote"):
            raise RuntimeNotConnectedError(
                f"Serve handle for deployment '{target.deployment_name}' does not expose "
                f"'{method_name}.remote(...)'."
            )

        try:
            guard = self._get_guard(target)
            await guard.acquire()
            stream_handle = handle
            options_method = getattr(handle, "options", None)
            if callable(options_method):
                try:
                    stream_handle = options_method(stream=True)
                except Exception:
                    # Fallback to legacy handle invocation for environments lacking stream options support.
                    stream_handle = handle
            stream_remote_method = getattr(stream_handle, method_name, None)
            if stream_remote_method is None or not hasattr(stream_remote_method, "remote"):
                stream_remote_method = remote_method

            response = stream_remote_method.remote(request_payload=payload)
            return self._guard_stream_chunks(
                self._normalize_stream_result(response),
                guard=guard,
                target=target,
                method_name=method_name,
            )
        except AdmissionRejectedError:
            raise
        except RuntimeNotConnectedError:
            raise
        except Exception as exc:
            if "guard" in locals():
                guard.record_failure()
                guard.release()
            code = self._extract_execution_error_code(exc)
            message = self._extract_execution_error_message(exc)
            logger.exception(
                "Serve streaming execution failed for deployment '%s' in app '%s' method '%s'.",
                target.deployment_name,
                target.app_name,
                method_name,
            )
            if code in {"unsupported_parameter", "unsupported_message_content", "invalid_input"}:
                raise RuntimeExecutionError(message, code=code) from exc
            raise RuntimeExecutionError(
                f"Serve streaming execution failed for deployment '{target.deployment_name}' "
                f"in app '{target.app_name}' method '{method_name}': {message}",
                code=code,
            ) from exc

    def _serve_timeout_error(self, target: RuntimeTarget, method_name: str) -> RuntimeNotConnectedError:
        """Build a stable upstream timeout error for a Serve handle call."""
        return RuntimeNotConnectedError(
            f"Serve request timed out after {self.serve_request_timeout_seconds}s for "
            f"deployment '{target.deployment_name}' in app '{target.app_name}' "
            f"method '{method_name}'.",
            code="upstream_timeout",
        )

    async def _guard_stream_chunks(
        self,
        chunks: AsyncIterator[dict[str, Any] | bytes | str],
        *,
        guard: _ServeDeploymentGuard,
        target: RuntimeTarget,
        method_name: str,
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
        """Apply timeout, circuit accounting, and inflight release to a Serve stream."""
        iterator = chunks.__aiter__()
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        iterator.__anext__(),
                        timeout=self.serve_request_timeout_seconds,
                    )
                except StopAsyncIteration:
                    guard.record_success()
                    return
                except TimeoutError as exc:
                    guard.record_failure()
                    raise self._serve_timeout_error(target, method_name) from exc
                yield chunk
        except asyncio.CancelledError:
            raise
        except RuntimeNotConnectedError:
            raise
        except Exception:
            guard.record_failure()
            raise
        finally:
            guard.release()

    async def _await_handle_response(self, response: Any) -> Any:
        """兼容 awaitable、ObjectRef 风格和普通返回值。"""
        if inspect.isawaitable(response):
            return await response
        if hasattr(response, "result"):
            return response.result()
        return response

    async def _normalize_stream_result(self, response: Any) -> AsyncIterator[dict[str, Any] | bytes | str]:
        """把同步、异步和 Ray 风格的流式返回统一成异步迭代器。"""
        if hasattr(response, "__aiter__"):
            async for chunk in response:
                yield chunk
            return

        if inspect.isawaitable(response):
            response = await response
        if hasattr(response, "result") and not hasattr(response, "__aiter__"):
            response = response.result()

        if hasattr(response, "__aiter__"):
            async for chunk in response:
                yield chunk
            return

        if isinstance(response, (bytes, str, dict)):
            yield response
            return

        if isinstance(response, Iterable):
            for chunk in response:
                yield chunk
            return

        raise RuntimeExecutionError(
            f"Streaming method returned unsupported payload type '{type(response).__name__}'.",
            code="runtime_execution_failed",
        )

    async def _prime_stream_chunks(
        self,
        chunks: AsyncIterator[dict[str, Any] | bytes | str],
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
        """先消费首个流式片段，让参数错误能以 JSON 错误返回。"""
        iterator = chunks.__aiter__()
        try:
            first = await iterator.__anext__()
        except StopAsyncIteration:
            async def empty() -> AsyncIterator[dict[str, Any] | bytes | str]:
                """返回一个空的异步流。"""
                if False:
                    yield {}
            return empty()

        async def replay() -> AsyncIterator[dict[str, Any] | bytes | str]:
            """重放已预取的首块并继续转发剩余片段。"""
            yield first
            async for chunk in iterator:
                yield chunk

        return replay()

    def _parse_proxy_config(self, target: RuntimeTarget) -> ProxyConfig:
        """从运行时目标中读取并校验 OpenAI 兼容代理配置。"""
        raw = target.runtime_context.get("proxy_config") or {}
        try:
            return ProxyConfig.model_validate(raw)
        except Exception as exc:
            raise RuntimeNotConnectedError(
                f"Proxy config for model '{target.model_name}' is invalid: {exc}",
                code="backend_misconfigured",
            ) from exc

    def _build_proxy_headers(self, proxy_config: ProxyConfig, *, request_id: str) -> dict[str, str]:
        """根据代理鉴权策略和请求追踪设置构造上游请求头。"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if proxy_config.headers_policy.pass_request_id:
            headers["X-Request-ID"] = request_id
        if proxy_config.auth.mode == "static_bearer" and proxy_config.auth.token:
            headers["Authorization"] = f"Bearer {proxy_config.auth.token}"
        if proxy_config.auth.mode == "bearer_env":
            if not proxy_config.auth.env_var:
                raise RuntimeNotConnectedError(
                    "proxy auth.mode=bearer_env requires auth.env_var",
                    code="backend_misconfigured",
                )
            token = os.getenv(proxy_config.auth.env_var)
            if not token:
                raise RuntimeNotConnectedError(
                    f"proxy auth env var '{proxy_config.auth.env_var}' is not set",
                    code="backend_misconfigured",
                )
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _request_proxy(
        self,
        *,
        proxy_config: ProxyConfig,
        path: str,
        payload: dict[str, Any],
        request_id: str,
    ) -> httpx.Response:
        """向上游代理发起 JSON 请求，并按配置处理超时和重试。"""
        upstream = proxy_config.upstream_base_url.rstrip("/")
        timeout = httpx.Timeout(
            connect=proxy_config.timeout.connect_seconds,
            read=proxy_config.timeout.read_seconds,
            write=proxy_config.timeout.write_seconds,
            pool=proxy_config.timeout.pool_seconds,
        )
        url = f"{upstream}/{path.lstrip('/')}"
        headers = self._build_proxy_headers(proxy_config, request_id=request_id)
        attempts = max(1, proxy_config.retry.max_attempts)
        backoff = proxy_config.retry.backoff_ms / 1000.0
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                if attempt < attempts and response.status_code in proxy_config.retry.retry_on_status:
                    if backoff > 0:
                        await asyncio.sleep(backoff * attempt)
                    continue
                return response
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                last_exc = exc
                if attempt >= attempts:
                    break
                if backoff > 0:
                    await asyncio.sleep(backoff * attempt)

        raise RuntimeNotConnectedError(
            f"proxy upstream request failed for '{url}': {last_exc}",
            code="upstream_timeout",
        )

    def _proxy_payload(self, *, model_name: str, payload: dict[str, Any], proxy_config: ProxyConfig) -> dict[str, Any]:
        """替换或保留请求中的模型名，生成发送给上游的载荷。"""
        request_payload = dict(payload)
        request_payload["model"] = proxy_config.upstream_model_name or model_name
        return request_payload

    def _to_proxy_response(self, response: httpx.Response, *, request_id: str) -> Response:
        """将上游 HTTP 响应原样包装为网关响应并附加请求 ID。"""
        headers = {"X-Infer-Nexus-Request-ID": request_id}
        content_type = response.headers.get("content-type")
        if content_type:
            headers["content-type"] = content_type
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=headers,
        )

    async def _execute_proxy_chat(
        self,
        *,
        target: RuntimeTarget,
        request: ChatCompletionsRequest,
    ) -> ChatCompletionsResponse | Response:
        """把 chat 请求转发到 OpenAI 兼容上游，必要时走 SSE 透传。"""
        proxy_config = self._parse_proxy_config(target)
        request_id = uuid4().hex
        payload = self._proxy_payload(
            model_name=request.model,
            payload=request.model_dump(mode="json", exclude_none=True),
            proxy_config=proxy_config,
        )
        if request.stream:
            if not proxy_config.streaming.enabled or not proxy_config.streaming.passthrough_sse:
                raise RuntimeNotConnectedError(
                    "proxy model does not allow stream passthrough",
                    code="unsupported_parameter",
                )
            return await self._execute_proxy_stream(
                proxy_config=proxy_config,
                path="/chat/completions",
                payload=payload,
                request_id=request_id,
                model_label=self._stream_metric_model(target, request),
                stream_start_time=perf_counter(),
            )
        response = await self._request_proxy(
            proxy_config=proxy_config,
            path="/chat/completions",
            payload=payload,
            request_id=request_id,
        )
        return self._to_proxy_response(response, request_id=request_id)

    async def _execute_proxy_stream(
        self,
        *,
        proxy_config: ProxyConfig,
        path: str,
        payload: dict[str, Any],
        request_id: str,
        model_label: str,
        stream_start_time: float,
    ) -> Response:
        """建立上游流式 HTTP 连接并把 SSE 字节流透传给客户端。"""
        upstream = proxy_config.upstream_base_url.rstrip("/")
        timeout = httpx.Timeout(
            connect=proxy_config.timeout.connect_seconds,
            read=proxy_config.timeout.read_seconds,
            write=proxy_config.timeout.write_seconds,
            pool=proxy_config.timeout.pool_seconds,
        )
        url = f"{upstream}/{path.lstrip('/')}"
        headers = self._build_proxy_headers(proxy_config, request_id=request_id)
        client = httpx.AsyncClient(timeout=timeout)
        request = client.build_request("POST", url, json=payload, headers=headers)
        response = await client.send(request, stream=True)
        if response.status_code >= 400:
            await response.aread()
            await response.aclose()
            await client.aclose()
            return self._to_proxy_response(response, request_id=request_id)

        async def iterator() -> Any:
            """转发上游字节流并确保响应和客户端连接最终关闭。"""
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return StreamingResponse(
            self._observe_stream_chunks(
                iterator(),
                model_label=model_label,
                stream_start_time=stream_start_time,
            ),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "text/event-stream"),
            headers={"X-Infer-Nexus-Request-ID": request_id},
        )

    def _extract_execution_error_code(self, exc: Exception) -> str:
        """从 Serve 包装异常和嵌套异常中提取稳定错误码。"""
        candidates = [
            exc,
            getattr(exc, "cause", None),
            getattr(exc, "__cause__", None),
        ]
        for candidate in candidates:
            code = getattr(candidate, "code", None)
            if isinstance(code, str) and code:
                return code

        message = str(exc)
        validation_signatures = (
            "BackendConfigurationError",
            "BackendRequestValidationError",
            "RuntimeExecutionError: This model does not allow",
            "RuntimeExecutionError: Streaming chat completions are not supported",
            "RuntimeExecutionError: This model does not support multimodal chat content",
            "RuntimeExecutionError: Multimodal chat content must include",
        )
        if any(signature in message for signature in validation_signatures):
            if "BackendConfigurationError" in message:
                return "backend_misconfigured"
            if "multimodal chat content" in message:
                return "unsupported_message_content"
            if "must include at least one content block" in message:
                return "invalid_input"
            return "unsupported_parameter"

        return "runtime_execution_failed"

    def _extract_execution_error_message(self, exc: Exception) -> str:
        """剥离框架异常前缀，提取适合返回给客户端的错误消息。"""
        candidates = [str(exc)]
        for attr in ("cause", "__cause__"):
            nested = getattr(exc, attr, None)
            if nested is not None:
                candidates.append(str(nested))

        prefixes = (
            "RuntimeExecutionError: ",
            "BackendRequestValidationError: ",
            "BackendConfigurationError: ",
            "RuntimeNotConnectedError: ",
        )
        for text in candidates:
            for prefix in prefixes:
                if prefix in text:
                    return text.split(prefix, 1)[1].strip()

        return str(exc)

    async def _execute_proxy_embedding(
        self,
        *,
        target: RuntimeTarget,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse | Response:
        """把 embedding 请求转发到 OpenAI 兼容上游并透传响应。"""
        proxy_config = self._parse_proxy_config(target)
        request_id = uuid4().hex
        response = await self._request_proxy(
            proxy_config=proxy_config,
            path="/embeddings",
            payload=self._proxy_payload(
                model_name=request.model,
                payload=request.model_dump(mode="json", exclude_none=True),
                proxy_config=proxy_config,
            ),
            request_id=request_id,
        )
        return self._to_proxy_response(response, request_id=request_id)

    async def _execute_proxy_rerank(
        self,
        *,
        target: RuntimeTarget,
        request: RerankRequest,
    ) -> RerankResponse | Response:
        """把 rerank 请求转发到上游服务并透传响应。"""
        proxy_config = self._parse_proxy_config(target)
        request_id = uuid4().hex
        response = await self._request_proxy(
            proxy_config=proxy_config,
            path="/rerank",
            payload=self._proxy_payload(
                model_name=request.model,
                payload=request.model_dump(mode="json", exclude_none=True),
                proxy_config=proxy_config,
            ),
            request_id=request_id,
        )
        return self._to_proxy_response(response, request_id=request_id)

    def _build_chat_response_from_payload(
        self,
        request: ChatCompletionsRequest,
        target: RuntimeTarget,
        payload: dict[str, Any],
    ) -> ChatCompletionsResponse:
        """把最小后端 chat 载荷补齐为 OpenAI chat completion 响应。"""
        # payload 允许后端按最小约定返回字段；此处补齐默认值并强制映射到外部协议。
        return ChatCompletionsResponse(
            id=payload.get("id", f"chatcmpl-{uuid4().hex}"),
            created=payload.get("created", int(time())),
            model=payload.get(
                "model",
                target.runtime_context.get("served_model_name", request.model),
            ),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role='assistant',
                        content=payload.get(
                            "content",
                            f"serve chat response from deployment '{target.deployment_name}' "
                            f"for model '{target.model_name}'",
                        ),
                        reasoning_content=payload.get("reasoning_content"),
                        reasoning=payload.get("reasoning"),
                    ),
                    finish_reason=payload.get("finish_reason", 'stop'),
                )
            ],
            usage=TokenUsage.model_validate(
                payload.get(
                    "usage",
                    {
                        "prompt_tokens": len(request.messages),
                        "completion_tokens": 8,
                        "total_tokens": len(request.messages) + 8,
                    },
                )
            ),
        )

    def _is_openai_chat_payload(self, payload: dict[str, Any]) -> bool:
        """判断后端载荷是否已经是完整 OpenAI chat completion。"""
        return payload.get("object") == "chat.completion" and isinstance(payload.get("choices"), list)

    def _strip_internal_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        """移除内部 `status=ok` 字段，避免污染透传 OpenAI 响应。"""
        if payload.get("status") != "ok":
            return dict(payload)
        return {key: value for key, value in payload.items() if key != "status"}

    def _encode_sse_chunk(self, chunk_payload: dict[str, Any]) -> bytes:
        """把字典事件编码成 OpenAI SSE 的 `data:` 数据块。"""
        return f"data: {json.dumps(chunk_payload, ensure_ascii=False)}\n\n".encode("utf-8")

    def _is_chat_delta_event(self, chunk: dict[str, Any]) -> bool:
        """识别内部约定的聊天增量事件。"""
        return chunk.get("type") == "chat_delta"

    def _is_done_sse_chunk(self, chunk: bytes | str) -> bool:
        """检测原始 SSE 片段里是否已经包含 `[DONE]` 结束标记。"""
        if isinstance(chunk, bytes):
            text = chunk.decode("utf-8", errors="ignore")
        else:
            text = chunk
        return "data: [DONE]" in text

    def _stream_metric_model(self, target: RuntimeTarget, request: ChatCompletionsRequest) -> str:
        """返回流式指标使用的稳定模型标签。"""
        return str(
            target.model_alias
            or target.runtime_context.get("served_model_name")
            or target.model_name
            or request.model
        )

    def _observe_stream_emit(
        self,
        *,
        model_label: str,
        stream_start_time: float,
        previous_emit_time: float | None,
    ) -> float:
        """记录流式首块延迟或相邻块间隔，并返回当前输出时间。"""
        now = perf_counter()
        if previous_emit_time is None:
            GATEWAY_METRICS.observe_stream_ttft(
                model=model_label,
                seconds=now - stream_start_time,
            )
        else:
            GATEWAY_METRICS.observe_stream_chunk_interval(
                model=model_label,
                seconds=now - previous_emit_time,
            )
        return now

    async def _observe_stream_chunks(
        self,
        chunks: AsyncIterator[bytes],
        *,
        model_label: str,
        stream_start_time: float,
    ) -> AsyncIterator[bytes]:
        """包装 SSE 字节流，统一记录 TTFT 和相邻输出块间隔。"""
        previous_emit_time: float | None = None
        try:
            async for chunk in chunks:
                previous_emit_time = self._observe_stream_emit(
                    model_label=model_label,
                    stream_start_time=stream_start_time,
                    previous_emit_time=previous_emit_time,
                )
                yield chunk
        except asyncio.CancelledError:
            GATEWAY_METRICS.observe_stream_completion(model=model_label, status="cancelled")
            raise
        except Exception:
            GATEWAY_METRICS.observe_stream_completion(model=model_label, status="error")
            raise
        else:
            GATEWAY_METRICS.observe_stream_completion(model=model_label, status="success")

    def _build_chat_stream_response(
        self,
        request: ChatCompletionsRequest,
        target: RuntimeTarget,
        chunks: dict[str, Any] | AsyncIterator[dict[str, Any] | bytes | str],
        *,
        stream_start_time: float,
    ) -> StreamingResponse:
        """根据兼容模式选择 SSE 透传或内部增量事件映射。"""
        model_label = self._stream_metric_model(target, request)
        if not isinstance(chunks, dict):
            runtime_spec = target.runtime_context.get("runtime_spec") or {}
            compat_mode = target.runtime_context.get("compat_mode") or runtime_spec.get("compat_mode")
            if str(compat_mode) in {
                CompatibilityMode.VLLM_NATIVE.value,
                CompatibilityMode.STRICT_OPENAI.value,
            }:
                return self._build_passthrough_stream_response(
                    chunks,
                    model_label=model_label,
                    stream_start_time=stream_start_time,
                )
            return self._build_mapped_chat_stream_response(
                request,
                target,
                chunks,
                model_label=model_label,
                stream_start_time=stream_start_time,
            )

        payload = chunks
        created = payload.get("created", int(time()))
        response_id = payload.get("id", f"chatcmpl-{uuid4().hex}")
        model = payload.get(
            "model",
            target.runtime_context.get("served_model_name", request.model),
        )
        content = payload.get(
            "content",
            f"serve chat response from deployment '{target.deployment_name}' "
            f"for model '{target.model_name}'",
        )
        finish_reason = payload.get("finish_reason", "stop")
        request_id = uuid4().hex

        async def iterator() -> Any:
            """把单次载荷或预置片段输出为 OpenAI chat SSE 流。"""
            stream_chunks = payload.get("stream_chunks")
            if isinstance(stream_chunks, list):
                saw_done = False
                for stream_chunk in stream_chunks:
                    if isinstance(stream_chunk, bytes):
                        saw_done = saw_done or self._is_done_sse_chunk(stream_chunk)
                        yield stream_chunk
                    elif isinstance(stream_chunk, str):
                        saw_done = saw_done or self._is_done_sse_chunk(stream_chunk)
                        yield stream_chunk.encode("utf-8")
                    elif isinstance(stream_chunk, dict):
                        yield self._encode_sse_chunk(stream_chunk)
                if not saw_done:
                    yield b"data: [DONE]\n\n"
                return

            first_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield self._encode_sse_chunk(first_chunk)

            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            yield self._encode_sse_chunk(final_chunk)
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            self._observe_stream_chunks(
                iterator(),
                model_label=model_label,
                stream_start_time=stream_start_time,
            ),
            media_type="text/event-stream",
            headers={"X-Infer-Nexus-Request-ID": request_id},
        )

    def _build_passthrough_stream_response(
        self,
        chunks: AsyncIterator[dict[str, Any] | bytes | str],
        *,
        model_label: str,
        stream_start_time: float,
    ) -> StreamingResponse:
        """直接透传后端 SSE 片段，并在缺失时补充 `[DONE]`。"""
        request_id = uuid4().hex

        async def iterator() -> Any:
            """逐块转发后端流式片段，兼容 bytes、str 和 dict 事件。"""
            saw_done = False
            try:
                async for chunk in chunks:
                    if isinstance(chunk, bytes):
                        saw_done = saw_done or self._is_done_sse_chunk(chunk)
                        yield chunk
                    elif isinstance(chunk, str):
                        saw_done = saw_done or self._is_done_sse_chunk(chunk)
                        yield chunk.encode("utf-8")
                    elif isinstance(chunk, dict):
                        yield self._encode_sse_chunk(chunk)
            except Exception as exc:
                logger.exception(
                    "SSE stream iteration failed for request_id '%s': %s",
                    request_id,
                    exc,
                )
                return
            if not saw_done:
                yield b"data: [DONE]\n\n"

        return StreamingResponse(
            self._observe_stream_chunks(
                iterator(),
                model_label=model_label,
                stream_start_time=stream_start_time,
            ),
            media_type="text/event-stream",
            headers={"X-Infer-Nexus-Request-ID": request_id},
        )

    def _build_mapped_chat_stream_response(
        self,
        request: ChatCompletionsRequest,
        target: RuntimeTarget,
        chunks: AsyncIterator[dict[str, Any] | bytes | str],
        *,
        model_label: str,
        stream_start_time: float,
    ) -> StreamingResponse:
        """把内部 chat_delta 事件映射为 OpenAI chat completion chunk。"""
        request_id = uuid4().hex

        async def iterator() -> Any:
            """维护 role、delta 和 finish 事件顺序并生成 SSE 输出。"""
            saw_done = False
            sent_role = False
            sent_finish = False
            response_id = f"chatcmpl-{uuid4().hex}"
            created = int(time())
            model = target.runtime_context.get("served_model_name", request.model)
            try:
                async for chunk in chunks:
                    if isinstance(chunk, bytes):
                        saw_done = saw_done or self._is_done_sse_chunk(chunk)
                        yield chunk
                        continue
                    if isinstance(chunk, str):
                        saw_done = saw_done or self._is_done_sse_chunk(chunk)
                        yield chunk.encode("utf-8")
                        continue
                    if not self._is_chat_delta_event(chunk):
                        yield self._encode_sse_chunk(chunk)
                        continue

                    response_id = str(chunk.get("id") or response_id)
                    created = int(chunk.get("created") or created)
                    model = str(chunk.get("model") or model)
                    if not sent_role:
                        yield self._encode_sse_chunk(
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"role": "assistant"},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                        sent_role = True

                    delta_text = chunk.get("delta_text")
                    if isinstance(delta_text, str) and delta_text:
                        yield self._encode_sse_chunk(
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": delta_text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )

                    finish_reason = chunk.get("finish_reason")
                    if finish_reason:
                        yield self._encode_sse_chunk(
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": finish_reason,
                                    }
                                ],
                            }
                        )
                        sent_finish = True
                        yield b"data: [DONE]\n\n"
                        saw_done = True
            except Exception as exc:
                logger.exception(
                    "SSE stream iteration failed for request_id '%s': %s",
                    request_id,
                    exc,
                )
                return
            if not saw_done:
                if sent_role and not sent_finish:
                    yield self._encode_sse_chunk(
                        {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    )
                yield b"data: [DONE]\n\n"

        return StreamingResponse(
            self._observe_stream_chunks(
                iterator(),
                model_label=model_label,
                stream_start_time=stream_start_time,
            ),
            media_type="text/event-stream",
            headers={"X-Infer-Nexus-Request-ID": request_id},
        )

    def _build_embedding_response_from_payload(
        self,
        request: EmbeddingRequest,
        target: RuntimeTarget,
        payload: dict[str, Any],
    ) -> EmbeddingResponse:
        """把后端 embedding 载荷校验并补齐为 OpenAI embedding 响应。"""
        # embedding data 逐项做结构校验，确保输出稳定且可被 OpenAI SDK 消费。
        return EmbeddingResponse(
            data=[
                EmbeddingData.model_validate(item)
                for item in payload.get("data", [])
            ],
            model=payload.get(
                "model",
                target.runtime_context.get("served_model_name", request.model),
            ),
            usage=TokenUsage.model_validate(
                payload.get(
                    "usage",
                    {
                        "prompt_tokens": (
                            len(request.input) if isinstance(request.input, list) else 1
                        ),
                        "completion_tokens": 0,
                        "total_tokens": (
                            len(request.input) if isinstance(request.input, list) else 1
                        ),
                    },
                )
            ),
        )

    def _build_rerank_response_from_payload(
        self,
        request: RerankRequest,
        target: RuntimeTarget,
        payload: dict[str, Any],
    ) -> RerankResponse:
        """把后端 rerank 载荷校验并补齐兜底结果。"""
        # 当后端无返回时提供可解释的降级结果，避免接口层直接失败。
        document_list = request.documents if isinstance(request.documents, list) else [request.documents]
        fallback_results = [
            {
                "index": index,
                "document": {"text": document},
                "relevance_score": 0.0,
            }
            for index, document in enumerate(
                document_list[: request.top_n] if request.top_n > 0 else document_list
            )
        ]
        return RerankResponse(
            id=payload.get("id", f"rerank-{uuid4().hex}"),
            model=payload.get(
                "model",
                target.runtime_context.get("served_model_name", request.model),
            ),
            usage=RerankUsage.model_validate(
                payload.get("usage", {"total_tokens": 1 + len(document_list)})
            ),
            results=[
                RerankResult.model_validate(item)
                for item in payload.get("results", fallback_results)
            ],
        )
