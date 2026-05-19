"""vLLM 后端适配实现。"""

import base64
import inspect
import re
import struct
from time import time
from typing import Any
from uuid import uuid4

from infer_nexus.backends.base import InferenceBackend
from infer_nexus.catalog.models import ModelConfig
from infer_nexus.core.errors import BackendConfigurationError, BackendRequestValidationError
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest


class VLLMBackend(InferenceBackend):
    """vLLM 后端适配器，负责 runtime spec 校验、引擎生命周期与请求转换。"""

    TASK_TO_MODE = {
        "chat": "generate",
        "embedding": "embed",
        "rerank": "score",
    }

    def __init__(self, runtime_spec: dict[str, Any]) -> None:
        """初始化后端实例。"""
        self.runtime_spec = runtime_spec
        self.engine: Any | None = None
        self.engine_state: str = "created"

    def validate_runtime_spec(self, runtime_spec: dict[str, Any], runtime_context: dict[str, Any]) -> None:
        """校验模型任务与 runtime spec 的后端参数一致性。"""
        task = getattr(runtime_context.get("task"), "value", runtime_context.get("task"))
        if task not in self.TASK_TO_MODE:
            raise BackendConfigurationError(
                f"vLLM backend does not support task '{task}'."
            )

        expected_mode = self.TASK_TO_MODE[task]
        actual_mode = runtime_spec.get("task_mode")
        if actual_mode != expected_mode:
            raise BackendConfigurationError(
                f"vLLM runtime_spec task_mode mismatch for model '{runtime_context.get('model_name')}'. "
                f"Expected '{expected_mode}' for task '{task}', got '{actual_mode}'."
            )

        if runtime_spec.get("backend") != "vllm":
            raise BackendConfigurationError(
                f"Runtime spec backend mismatch for model '{runtime_context.get('model_name')}'. "
                f"Expected 'vllm', got '{runtime_spec.get('backend')}'."
            )

    def startup(self) -> None:
        """启动 vLLM 引擎；stub 模式下不加载真实模型。"""
        init_mode = self.runtime_spec.get("backend_init_mode", "stub")
        if init_mode != "real":
            self.engine = None
            self.engine_state = "stub"
            return

        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Install the 'vllm' extra or switch runtime.backend_init_mode to 'stub'."
            ) from exc

        llm_kwargs: dict[str, Any] = {
            "model": self.runtime_spec["model_path"],
            "tensor_parallel_size": self.runtime_spec["tensor_parallel_size"],
            "dtype": self.runtime_spec.get("dtype") or "auto",
        }
        loading_config = self.runtime_spec.get("model_loading_config") or {}
        if loading_config.get("revision"):
            llm_kwargs["revision"] = loading_config["revision"]
        llm_kwargs.update(self.runtime_spec.get("engine_kwargs") or {})
        gpu_memory_utilization = self.runtime_spec.get("gpu_memory_utilization")
        max_model_len = self.runtime_spec.get("max_model_len")
        requested_mode = self.runtime_spec.get("task_mode")

        # 核心兼容逻辑：
        # 不同 vLLM 版本构造参数不一致，仅在当前版本签名支持时才传入对应参数。
        try:
            llm_signature = inspect.signature(LLM.__init__)
            llm_init_args = llm_signature.parameters
            accepts_var_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in llm_init_args.values()
            )

            if gpu_memory_utilization is not None and (
                "gpu_memory_utilization" in llm_init_args or accepts_var_kwargs
            ):
                llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
            if max_model_len is not None and ("max_model_len" in llm_init_args or accepts_var_kwargs):
                llm_kwargs["max_model_len"] = max_model_len
            # Keep task strict: vLLM 0.18.1 forwards unknown kwargs and
            # EngineArgs will reject unsupported `task`.
            if "task" in llm_init_args:
                llm_kwargs["task"] = requested_mode or "auto"
        except (TypeError, ValueError):
            # If introspection fails, avoid passing version-sensitive args.
            pass

        self.engine = LLM(**llm_kwargs)
        supported_tasks = getattr(self.engine, "supported_tasks", None)
        if not supported_tasks:
            engine_task = getattr(self.engine, "task", None)
            if engine_task:
                supported_tasks = [engine_task]
        if supported_tasks and requested_mode not in supported_tasks:
            raise BackendConfigurationError(
                f"Loaded vLLM model does not support requested task_mode '{requested_mode}'. "
                f"Supported tasks: {sorted(supported_tasks)}."
            )
        self.engine_state = "ready"

    def shutdown(self) -> None:
        """关闭引擎并清理状态。"""
        self.engine = None
        self.engine_state = "stopped"

    def build_runtime_spec(self, model: ModelConfig, resolved_model_reference: str) -> dict[str, Any]:
        """将模型声明转换为 vLLM 可消费的 runtime spec。"""
        task_mode = self.TASK_TO_MODE.get(model.task.value)
        if task_mode is None:
            raise BackendConfigurationError(
                f"vLLM backend does not support model task '{model.task.value}' for model '{model.name}'."
            )

        return {
            "backend": "vllm",
            "model_path": resolved_model_reference,
            "tensor_parallel_size": model.tensor_parallel_size,
            "dtype": model.dtype,
            "max_model_len": model.max_model_len,
            "gpu_per_replica": model.gpu_per_replica,
            "gpu_memory_utilization": model.gpu_memory_utilization,
            "cpu_per_replica": model.cpu_per_replica,
            "task_mode": task_mode,
            "capabilities": list(model.capabilities),
            "engine_kwargs": dict(model.engine_kwargs),
            "model_loading_config": model.model_loading_config.model_dump(mode="json"),
            "served_model_name": model.served_model_name or model.alias or model.name,
        }

    def _normalize_embedding_inputs(self, request: EmbeddingRequest) -> list[str]:
        """归一化 embedding 输入为字符串列表。"""
        inputs = request.input if isinstance(request.input, list) else [request.input]
        if not inputs:
            raise BackendRequestValidationError(
                "Embedding requests must include at least one input.",
                code="invalid_input",
            )
        return inputs

    def _encode_embedding_base64(self, embedding: list[float]) -> str:
        """将浮点向量编码为 OpenAI 兼容 base64 格式。"""
        packed = struct.pack(f"<{len(embedding)}f", *embedding)
        return base64.b64encode(packed).decode("ascii")

    def _build_embedding_stub_response(
        self,
        request: EmbeddingRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        inputs: list[str],
    ) -> dict[str, Any]:
        """构建 embedding 的 stub 响应。"""
        data = []
        for index, item in enumerate(inputs):
            embedding = [
                float(len(item)),
                float(index),
                float(len(runtime_spec["backend"])),
            ]
            if request.encoding_format == "base64":
                value: list[float] | str = self._encode_embedding_base64(embedding)
            else:
                value = embedding
            data.append({"index": index, "embedding": value})

        return {
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": len(inputs),
                "completion_tokens": 0,
                "total_tokens": len(inputs),
            },
            "backend": runtime_spec["backend"],
            "deployment": runtime_context["deployment_name"],
            "raw": {
                "input_count": len(inputs),
                "engine_state": self.engine_state,
            },
        }

    def _convert_embedding_result(
        self,
        *,
        request: EmbeddingRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        result: Any,
    ) -> dict[str, Any]:
        """将 vLLM embedding 原始结果转换为 API 响应。"""
        if not result:
            raise RuntimeError("vLLM embed returned no result")

        data = []
        prompt_tokens = 0
        for index, item in enumerate(result):
            outputs = getattr(item, "outputs", None)
            if outputs is None:
                raise RuntimeError("vLLM embedding result contained no outputs")

            embedding = getattr(outputs, "embedding", None)
            if embedding is None:
                raise RuntimeError("vLLM embedding output contained no embedding vector")

            prompt_tokens += len(getattr(item, "prompt_token_ids", None) or [])
            vector = list(embedding)
            if request.encoding_format == "base64":
                value: list[float] | str = self._encode_embedding_base64(vector)
            else:
                value = vector
            data.append({"index": index, "embedding": value})

        return {
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": prompt_tokens,
            },
            "backend": runtime_spec["backend"],
            "deployment": runtime_context["deployment_name"],
        }

    def _normalize_rerank_documents(self, request: RerankRequest) -> list[str]:
        """归一化 rerank 文档输入为字符串列表。"""
        documents = request.documents if isinstance(request.documents, list) else [request.documents]
        if not documents:
            raise BackendRequestValidationError(
                "Rerank requests must include at least one document.",
                code="invalid_input",
            )
        return documents

    def _score_stub_document(self, query: str, document: str) -> float:
        """基于词项重叠计算简单 stub 相关度分数。"""
        query_terms = {term for term in query.lower().split() if term}
        document_terms = {term for term in document.lower().split() if term}
        overlap = len(query_terms & document_terms)
        length_penalty = max(len(document_terms), 1)
        return overlap + (overlap / length_penalty)

    def _build_rerank_stub_response(
        self,
        request: RerankRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        documents: list[str],
    ) -> dict[str, Any]:
        """构建 rerank 的 stub 响应。"""
        scored = [
            {
                "index": index,
                "document": {"text": document},
                "relevance_score": float(self._score_stub_document(request.query, document)),
            }
            for index, document in enumerate(documents)
        ]
        scored.sort(key=lambda item: item["relevance_score"], reverse=True)
        if request.top_n > 0:
            scored = scored[: request.top_n]

        return {
            "id": f"rerank-{uuid4().hex}",
            "model": request.model,
            "usage": {"total_tokens": 1 + len(documents)},
            "results": scored,
            "backend": runtime_spec["backend"],
            "deployment": runtime_context["deployment_name"],
            "raw": {
                "document_count": len(documents),
                "engine_state": self.engine_state,
            },
        }

    def _convert_rerank_result(
        self,
        *,
        request: RerankRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        documents: list[str],
        result: Any,
    ) -> dict[str, Any]:
        """将 vLLM score 原始结果转换为 rerank 响应。"""
        if not result:
            raise RuntimeError("vLLM score returned no result")

        scored = []
        total_tokens = 0
        for index, item in enumerate(result):
            outputs = getattr(item, "outputs", None)
            if outputs is None or getattr(outputs, "score", None) is None:
                raise RuntimeError("vLLM score output contained no score")

            total_tokens += len(getattr(item, "prompt_token_ids", None) or [])
            scored.append(
                {
                    "index": index,
                    "document": {"text": documents[index]},
                    "relevance_score": float(outputs.score),
                }
            )

        scored.sort(key=lambda item: item["relevance_score"], reverse=True)
        if request.top_n > 0:
            scored = scored[: request.top_n]

        return {
            "id": f"rerank-{uuid4().hex}",
            "model": request.model,
            "usage": {"total_tokens": total_tokens},
            "results": scored,
            "backend": runtime_spec["backend"],
            "deployment": runtime_context["deployment_name"],
        }

    def _build_sampling_params(self, request: ChatCompletionsRequest) -> dict[str, Any]:
        """从 chat 请求提取采样参数并补充默认值。"""
        extra = dict(request.extra_body or {})
        model_extra = getattr(request, "model_extra", None) or {}
        max_tokens = request.max_tokens or extra.pop("max_tokens", None) or model_extra.get(
            "max_completion_tokens"
        ) or 512
        params: dict[str, Any] = {
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "top_p": request.top_p if request.top_p is not None else 1.0,
            "max_tokens": max_tokens,
        }
        optional_params = {
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "repetition_penalty": request.repetition_penalty,
            "stop": request.stop,
            "n": request.n,
            "seed": request.seed,
            "logprobs": request.top_logprobs if request.logprobs else None,
        }
        for key, value in optional_params.items():
            if value is not None:
                params[key] = value

        allowed_extra_keys = {
            "best_of",
            "top_k",
            "min_p",
            "min_tokens",
            "ignore_eos",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "include_stop_str_in_output",
            "truncate_prompt_tokens",
            "prompt_logprobs",
        }
        for key in allowed_extra_keys:
            if key in extra and extra[key] is not None:
                params[key] = extra[key]

        return params

    def _invoke_vllm_chat(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> Any:
        """Call vLLM chat across versions with different method signatures."""
        if self.engine is None:
            raise RuntimeError("vLLM engine is not initialized")

        try:
            from vllm import SamplingParams
        except ImportError:
            SamplingParams = None  # type: ignore[assignment]

        if SamplingParams is not None:
            try:
                return self.engine.chat(messages, sampling_params=SamplingParams(**sampling_params))
            except TypeError:
                pass

        try:
            return self.engine.chat(messages, **sampling_params)
        except TypeError:
            if SamplingParams is not None:
                return self.engine.chat(messages, SamplingParams(**sampling_params))
            raise

    def _supports_multimodal(self, runtime_context: dict[str, Any] | None = None) -> bool:
        """Return whether the current model runtime is allowed to accept image blocks."""
        capabilities = self.runtime_spec.get("capabilities")
        if capabilities is None and runtime_context is not None:
            capabilities = runtime_context.get("capabilities", [])
        return "vision" in (capabilities or [])

    def _normalize_data_url(self, url: str) -> str:
        """Normalize base64 data URLs into the stricter form expected by vLLM."""
        if not url.startswith("data:") or ";base64," not in url:
            return url

        prefix, payload = url.split(",", 1)
        normalized = re.sub(r"\s+", "", payload)
        normalized = normalized.replace("-", "+").replace("_", "/")
        padding = len(normalized) % 4
        if padding:
            normalized += "=" * (4 - padding)
        return f"{prefix},{normalized}"

    def _serialize_content_block(self, block: Any) -> dict[str, Any]:
        """Serialize one multimodal content block with minimal normalization."""
        payload = block.model_dump(mode="json")
        if payload.get("type") == "image_url":
            image_url = payload.get("image_url") or {}
            url = image_url.get("url")
            if isinstance(url, str):
                image_url["url"] = self._normalize_data_url(url)
        return payload

    def _serialize_message_content(
        self,
        content: Any,
        *,
        allow_multimodal: bool,
    ) -> str | list[dict[str, Any]]:
        """Normalize text or multimodal content into the payload expected by vLLM."""
        if isinstance(content, str):
            return content

        if not allow_multimodal:
            raise BackendRequestValidationError(
                "This model does not support multimodal chat content.",
                code="unsupported_message_content",
            )

        if not content:
            raise BackendRequestValidationError(
                "Multimodal chat content must include at least one content block.",
                code="invalid_input",
            )

        return [self._serialize_content_block(block) for block in content]

    def _build_chat_messages(
        self,
        request: ChatCompletionsRequest,
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """将 chat 消息转换为 vLLM 输入格式，并校验阶段一约束。"""
        if request.stream:
            raise BackendRequestValidationError(
                "Streaming chat completions are not supported in Phase 1.",
                code="unsupported_parameter",
            )

        allow_multimodal = self._supports_multimodal(runtime_context)
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            payload = {
                "role": message.role,
                "content": self._serialize_message_content(
                    message.content,
                    allow_multimodal=allow_multimodal,
                ),
            }
            if message.name:
                payload["name"] = message.name
            if message.tool_call_id:
                payload["tool_call_id"] = message.tool_call_id
            messages.append(payload)
        return messages

    def _build_chat_stub_response(
        self,
        request: ChatCompletionsRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        sampling_params: dict[str, Any],
    ) -> dict[str, Any]:
        """构建 chat 的 stub 响应。"""
        content = (
            f"backend stub response from {runtime_spec['backend']} "
            f"for deployment '{runtime_context['deployment_name']}' "
            f"(engine_state={self.engine_state}, "
            f"temperature={sampling_params['temperature']}, "
            f"top_p={sampling_params['top_p']}, "
            f"max_tokens={sampling_params['max_tokens']})"
        )
        prompt_tokens = len(request.messages)
        completion_tokens = 8
        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "created": int(time()),
            "model": request.model,
            "content": content,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "backend": runtime_spec["backend"],
            "deployment": runtime_context["deployment_name"],
            "raw": {
                "messages": self._build_chat_messages(request, runtime_context=runtime_context),
                "sampling_params": sampling_params,
            },
        }

    def _convert_chat_result(
        self,
        *,
        request: ChatCompletionsRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        result: Any,
        sampling_params: dict[str, Any],
    ) -> dict[str, Any]:
        """将 vLLM chat 原始结果转换为 OpenAI 风格响应。"""
        if not result:
            raise RuntimeError("vLLM chat returned no result")

        first = result[0]
        outputs = getattr(first, "outputs", None) or []
        if not outputs:
            raise RuntimeError("vLLM chat result contained no outputs")

        output = outputs[0]
        prompt_token_ids = getattr(first, "prompt_token_ids", None) or []
        output_token_ids = getattr(output, "token_ids", None) or []
        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "created": int(time()),
            "model": request.model,
            "content": getattr(output, "text", ""),
            "finish_reason": getattr(output, "finish_reason", "stop"),
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(output_token_ids),
                "total_tokens": len(prompt_token_ids) + len(output_token_ids),
            },
            "backend": runtime_spec["backend"],
            "deployment": runtime_context["deployment_name"],
            "raw": {
                "sampling_params": sampling_params,
            },
        }

    async def chat_completion(
        self,
        runtime_spec: dict[str, Any],
        request: ChatCompletionsRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """执行 chat completion。"""
        # 先做请求归一化和参数确定，再根据引擎是否就绪选择真实推理或 stub 降级。
        messages = self._build_chat_messages(request, runtime_context=runtime_context)
        sampling_params = self._build_sampling_params(request)
        if self.engine is None:
            return self._build_chat_stub_response(
                request,
                runtime_spec,
                runtime_context,
                sampling_params,
            )

        result = self._invoke_vllm_chat(messages, sampling_params)
        return self._convert_chat_result(
            request=request,
            runtime_spec=runtime_spec,
            runtime_context=runtime_context,
            result=result,
            sampling_params=sampling_params,
        )

    async def embedding(
        self,
        runtime_spec: dict[str, Any],
        request: EmbeddingRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """执行 embedding。"""
        inputs = self._normalize_embedding_inputs(request)
        if self.engine is None:
            return self._build_embedding_stub_response(
                request,
                runtime_spec,
                runtime_context,
                inputs,
            )

        result = self.engine.embed(inputs)
        return self._convert_embedding_result(
            request=request,
            runtime_spec=runtime_spec,
            runtime_context=runtime_context,
            result=result,
        )

    async def rerank(
        self,
        runtime_spec: dict[str, Any],
        request: RerankRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """执行 rerank。"""
        documents = self._normalize_rerank_documents(request)
        if self.engine is None:
            return self._build_rerank_stub_response(
                request,
                runtime_spec,
                runtime_context,
                documents,
            )

        result = self.engine.score(request.query, documents)
        return self._convert_rerank_result(
            request=request,
            runtime_spec=runtime_spec,
            runtime_context=runtime_context,
            documents=documents,
            result=result,
        )
