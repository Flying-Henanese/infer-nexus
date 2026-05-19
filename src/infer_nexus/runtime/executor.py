"""Execution adapter that bridges request schemas with runtime invocation paths."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from time import time
from typing import Any
from uuid import uuid4

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
    ) -> ChatCompletionsResponse:
        """执行聊天请求并转换为统一响应结构。"""
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
    ) -> EmbeddingResponse:
        """执行向量化请求并转换为统一响应结构。"""
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
    ) -> RerankResponse:
        """执行 rerank 请求并转换为统一响应结构。"""
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
            handle = self.handle_resolver.get_handle(target.deployment_name)
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
