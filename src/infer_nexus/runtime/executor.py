"""Execution adapter that bridges request schemas with runtime invocation paths."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
import inspect
import json
import logging
import os
from time import time
from typing import Any
from uuid import uuid4

import httpx
from starlette.responses import JSONResponse, Response, StreamingResponse

from infer_nexus.catalog.models import ProxyConfig
from infer_nexus.core.enums import BackendType
from infer_nexus.core.errors import RuntimeExecutionError, RuntimeNotConnectedError
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
from infer_nexus.runtime.deployments import ModelRuntimeReplica
from infer_nexus.runtime.handles import ServeDeploymentHandleResolver
from infer_nexus.runtime.types import RuntimeTarget

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeExecutor:
    """Execute inference requests via Serve handles or local stub replicas."""

    mode: str = "stub"
    handle_resolver: ServeDeploymentHandleResolver | None = None

    async def execute_chat(
        self,
        *,
        target: RuntimeTarget,
        request: ChatCompletionsRequest,
    ) -> ChatCompletionsResponse | Response:
        """执行聊天请求并转换为统一响应结构。"""
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
        return self._build_chat_stream_response(request, target, chunks)

    async def execute_embedding(
        self,
        *,
        target: RuntimeTarget,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse | Response:
        """执行向量化请求并转换为统一响应结构。"""
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
        """执行 rerank 请求并转换为统一响应结构。"""
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
        """在本地进程内调用副本方法（stub/dev 路径）。"""
        # Local path is used for stub/dev mode without requiring Ray Serve connectivity.
        replica = ModelRuntimeReplica(target.runtime_context)
        method = getattr(replica, method_name)
        return await method(payload)

    async def _invoke_local_replica_stream(
        self,
        *,
        target: RuntimeTarget,
        method_name: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
        """在本地进程内调用副本 streaming 方法（stub/dev 路径）。"""
        replica = ModelRuntimeReplica(target.runtime_context)
        method = getattr(replica, method_name)
        return self._normalize_stream_result(method(payload))

    async def _invoke_handle(
        self,
        *,
        target: RuntimeTarget,
        method_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """通过 Ray Serve deployment handle 调用远程副本方法。"""
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
            response = remote_method.remote(payload)
            return await self._await_handle_response(response)
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
        """通过 Ray Serve deployment handle 调用远程 streaming 副本方法。"""
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

            response = stream_remote_method.remote(payload)
            return self._normalize_stream_result(response)
        except RuntimeNotConnectedError:
            raise
        except Exception as exc:
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

    async def _await_handle_response(self, response: Any) -> Any:
        """Normalize different Ray/Serve return shapes into awaited payload."""
        if inspect.isawaitable(response):
            return await response
        if hasattr(response, "result"):
            return response.result()
        return response

    async def _normalize_stream_result(self, response: Any) -> AsyncIterator[dict[str, Any] | bytes | str]:
        """Normalize local, fake, and Serve streaming return shapes into an async iterator."""
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
        iterator = chunks.__aiter__()
        try:
            first = await iterator.__anext__()
        except StopAsyncIteration:
            async def empty() -> AsyncIterator[dict[str, Any] | bytes | str]:
                if False:
                    yield {}
            return empty()

        async def replay() -> AsyncIterator[dict[str, Any] | bytes | str]:
            yield first
            async for chunk in iterator:
                yield chunk

        return replay()

    def _parse_proxy_config(self, target: RuntimeTarget) -> ProxyConfig:
        raw = target.runtime_context.get("proxy_config") or {}
        try:
            return ProxyConfig.model_validate(raw)
        except Exception as exc:
            raise RuntimeNotConnectedError(
                f"Proxy config for model '{target.model_name}' is invalid: {exc}",
                code="backend_misconfigured",
            ) from exc

    def _build_proxy_headers(self, proxy_config: ProxyConfig, *, request_id: str) -> dict[str, str]:
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
        request_payload = dict(payload)
        request_payload["model"] = proxy_config.upstream_model_name or model_name
        return request_payload

    def _to_proxy_response(self, response: httpx.Response, *, request_id: str) -> Response:
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
    ) -> Response:
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
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return StreamingResponse(
            iterator(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "text/event-stream"),
            headers={"X-Infer-Nexus-Request-ID": request_id},
        )

    def _extract_execution_error_code(self, exc: Exception) -> str:
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
            "BackendRequestValidationError",
            "RuntimeExecutionError: This model does not allow",
            "RuntimeExecutionError: Streaming chat completions are not supported",
            "RuntimeExecutionError: This model does not support multimodal chat content",
            "RuntimeExecutionError: Multimodal chat content must include",
        )
        if any(signature in message for signature in validation_signatures):
            if "multimodal chat content" in message:
                return "unsupported_message_content"
            if "must include at least one content block" in message:
                return "invalid_input"
            return "unsupported_parameter"

        return "runtime_execution_failed"

    def _extract_execution_error_message(self, exc: Exception) -> str:
        candidates = [str(exc)]
        for attr in ("cause", "__cause__"):
            nested = getattr(exc, attr, None)
            if nested is not None:
                candidates.append(str(nested))

        prefixes = (
            "RuntimeExecutionError: ",
            "BackendRequestValidationError: ",
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
        """Adapt backend payload to OpenAI-compatible chat response schema."""
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
        return payload.get("object") == "chat.completion" and isinstance(payload.get("choices"), list)

    def _strip_internal_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("status") != "ok":
            return dict(payload)
        return {key: value for key, value in payload.items() if key != "status"}

    def _encode_sse_chunk(self, chunk_payload: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(chunk_payload, ensure_ascii=False)}\n\n".encode("utf-8")

    def _is_chat_delta_event(self, chunk: dict[str, Any]) -> bool:
        return chunk.get("type") == "chat_delta"

    def _is_done_sse_chunk(self, chunk: bytes | str) -> bool:
        if isinstance(chunk, bytes):
            text = chunk.decode("utf-8", errors="ignore")
        else:
            text = chunk
        return "data: [DONE]" in text

    def _build_chat_stream_response(
        self,
        request: ChatCompletionsRequest,
        target: RuntimeTarget,
        chunks: dict[str, Any] | AsyncIterator[dict[str, Any] | bytes | str],
    ) -> StreamingResponse:
        if not isinstance(chunks, dict):
            return self._build_mapped_chat_stream_response(request, target, chunks)

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
            iterator(),
            media_type="text/event-stream",
            headers={"X-Infer-Nexus-Request-ID": request_id},
        )

    def _build_passthrough_stream_response(
        self,
        chunks: AsyncIterator[dict[str, Any] | bytes | str],
    ) -> StreamingResponse:
        request_id = uuid4().hex

        async def iterator() -> Any:
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
            iterator(),
            media_type="text/event-stream",
            headers={"X-Infer-Nexus-Request-ID": request_id},
        )

    def _build_mapped_chat_stream_response(
        self,
        request: ChatCompletionsRequest,
        target: RuntimeTarget,
        chunks: AsyncIterator[dict[str, Any] | bytes | str],
    ) -> StreamingResponse:
        request_id = uuid4().hex

        async def iterator() -> Any:
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
            iterator(),
            media_type="text/event-stream",
            headers={"X-Infer-Nexus-Request-ID": request_id},
        )

    def _build_embedding_response_from_payload(
        self,
        request: EmbeddingRequest,
        target: RuntimeTarget,
        payload: dict[str, Any],
    ) -> EmbeddingResponse:
        """Adapt backend payload to OpenAI-compatible embeddings response schema."""
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
        """Adapt backend payload to native rerank response schema with safe fallback results."""
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
