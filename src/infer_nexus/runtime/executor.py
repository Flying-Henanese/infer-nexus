"""Execution adapter that bridges request schemas with runtime invocation paths."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import os
from time import time
from typing import Any
from uuid import uuid4

import httpx
from starlette.responses import JSONResponse, Response, StreamingResponse

from infer_nexus.catalog.models import ProxyConfig
from infer_nexus.core.enums import BackendType
from infer_nexus.core.errors import RuntimeNotConnectedError
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
        return self._build_chat_response_from_payload(request, target, payload)

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
            raise RuntimeNotConnectedError(
                f"Serve execution failed for deployment '{target.deployment_name}' method '{method_name}': "
                f"{exc}"
            ) from exc

    async def _await_handle_response(self, response: Any) -> Any:
        """Normalize different Ray/Serve return shapes into awaited payload."""
        if inspect.isawaitable(response):
            return await response
        if hasattr(response, "result"):
            return response.result()
        return response

    def _parse_proxy_config(self, target: RuntimeTarget) -> ProxyConfig:
        raw = target.runtime_context.get("proxy_config") or {}
        try:
            return ProxyConfig.model_validate(raw)
        except Exception as exc:
            raise RuntimeNotConnectedError(
                f"Proxy config for model '{target.model_name}' is invalid: {exc}",
                code="backend_misconfigured",
            ) from exc

    def _build_proxy_headers(self, proxy_config: ProxyConfig) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if proxy_config.headers_policy.pass_request_id:
            headers["X-Request-ID"] = uuid4().hex
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
    ) -> httpx.Response:
        upstream = proxy_config.upstream_base_url.rstrip("/")
        timeout = httpx.Timeout(
            connect=proxy_config.timeout.connect_seconds,
            read=proxy_config.timeout.read_seconds,
            write=proxy_config.timeout.write_seconds,
            pool=proxy_config.timeout.pool_seconds,
        )
        url = f"{upstream}/{path.lstrip('/')}"
        headers = self._build_proxy_headers(proxy_config)
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

    def _to_json_response(self, response: httpx.Response) -> JSONResponse:
        try:
            body = response.json()
        except ValueError:
            body = {"error": {"message": response.text or "upstream returned non-json error"}}
        return JSONResponse(status_code=response.status_code, content=body)

    async def _execute_proxy_chat(
        self,
        *,
        target: RuntimeTarget,
        request: ChatCompletionsRequest,
    ) -> ChatCompletionsResponse | Response:
        proxy_config = self._parse_proxy_config(target)
        payload = self._proxy_payload(
            model_name=request.model,
            payload=request.model_dump(mode="json"),
            proxy_config=proxy_config,
        )
        if request.stream:
            if not proxy_config.streaming.enabled or not proxy_config.streaming.passthrough_sse:
                raise RuntimeNotConnectedError(
                    "proxy model does not allow stream passthrough",
                    code="unsupported_parameter",
                )
            return await self._execute_proxy_stream(proxy_config=proxy_config, path="/chat/completions", payload=payload)
        response = await self._request_proxy(
            proxy_config=proxy_config,
            path="/chat/completions",
            payload=payload,
        )
        if response.status_code >= 400:
            return self._to_json_response(response)
        return ChatCompletionsResponse.model_validate(response.json())

    async def _execute_proxy_stream(
        self,
        *,
        proxy_config: ProxyConfig,
        path: str,
        payload: dict[str, Any],
    ) -> Response:
        upstream = proxy_config.upstream_base_url.rstrip("/")
        timeout = httpx.Timeout(
            connect=proxy_config.timeout.connect_seconds,
            read=proxy_config.timeout.read_seconds,
            write=proxy_config.timeout.write_seconds,
            pool=proxy_config.timeout.pool_seconds,
        )
        url = f"{upstream}/{path.lstrip('/')}"
        headers = self._build_proxy_headers(proxy_config)
        client = httpx.AsyncClient(timeout=timeout)
        request = client.build_request("POST", url, json=payload, headers=headers)
        response = await client.send(request, stream=True)
        if response.status_code >= 400:
            await response.aread()
            await response.aclose()
            await client.aclose()
            return self._to_json_response(response)

        async def iterator() -> Any:
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return StreamingResponse(
            iterator(),
            media_type=response.headers.get("content-type", "text/event-stream"),
            status_code=response.status_code,
        )

    async def _execute_proxy_embedding(
        self,
        *,
        target: RuntimeTarget,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse | Response:
        proxy_config = self._parse_proxy_config(target)
        response = await self._request_proxy(
            proxy_config=proxy_config,
            path="/embeddings",
            payload=self._proxy_payload(
                model_name=request.model,
                payload=request.model_dump(mode="json"),
                proxy_config=proxy_config,
            ),
        )
        if response.status_code >= 400:
            return self._to_json_response(response)
        return EmbeddingResponse.model_validate(response.json())

    async def _execute_proxy_rerank(
        self,
        *,
        target: RuntimeTarget,
        request: RerankRequest,
    ) -> RerankResponse | Response:
        proxy_config = self._parse_proxy_config(target)
        response = await self._request_proxy(
            proxy_config=proxy_config,
            path="/rerank",
            payload=self._proxy_payload(
                model_name=request.model,
                payload=request.model_dump(mode="json"),
                proxy_config=proxy_config,
            ),
        )
        if response.status_code >= 400:
            return self._to_json_response(response)
        return RerankResponse.model_validate(response.json())

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
