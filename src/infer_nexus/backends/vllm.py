import base64
import inspect
from pathlib import Path
import struct
from time import time
from typing import Any
from uuid import uuid4

from infer_nexus.backends.base import InferenceBackend
from infer_nexus.catalog.models import ModelConfig
from infer_nexus.core.errors import BackendConfigurationError, BackendRequestValidationError
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest


class VLLMBackend(InferenceBackend):
    TASK_TO_MODE = {
        "chat": "generate",
        "embedding": "embed",
        "rerank": "score",
    }

    def __init__(self, runtime_spec: dict[str, Any]) -> None:
        self.runtime_spec = runtime_spec
        self.engine: Any | None = None
        self.engine_state: str = "created"

    def validate_runtime_spec(self, runtime_spec: dict[str, Any], runtime_context: dict[str, Any]) -> None:
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
        gpu_memory_utilization = self.runtime_spec.get("gpu_memory_utilization")
        max_model_len = self.runtime_spec.get("max_model_len")
        requested_mode = self.runtime_spec.get("task_mode")

        # vLLM constructor args vary across versions.
        # Only pass version-sensitive kwargs when the current LLM signature supports them.
        try:
            llm_signature = inspect.signature(LLM.__init__)
            llm_init_args = llm_signature.parameters

            if gpu_memory_utilization is not None and "gpu_memory_utilization" in llm_init_args:
                llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
            if max_model_len is not None and "max_model_len" in llm_init_args:
                llm_kwargs["max_model_len"] = max_model_len
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
        self.engine = None
        self.engine_state = "stopped"

    def build_runtime_spec(self, model: ModelConfig, resolved_model_path: Path) -> dict[str, Any]:
        task_mode = self.TASK_TO_MODE.get(model.task.value)
        if task_mode is None:
            raise BackendConfigurationError(
                f"vLLM backend does not support model task '{model.task.value}' for model '{model.name}'."
            )

        return {
            "backend": "vllm",
            "model_path": str(resolved_model_path),
            "tensor_parallel_size": model.tensor_parallel_size,
            "dtype": model.dtype,
            "max_model_len": model.max_model_len,
            "gpu_per_replica": model.gpu_per_replica,
            "gpu_memory_utilization": model.gpu_memory_utilization,
            "cpu_per_replica": model.cpu_per_replica,
            "task_mode": task_mode,
        }

    def _normalize_embedding_inputs(self, request: EmbeddingRequest) -> list[str]:
        inputs = request.input if isinstance(request.input, list) else [request.input]
        if not inputs:
            raise BackendRequestValidationError(
                "Embedding requests must include at least one input.",
                code="invalid_input",
            )
        return inputs

    def _encode_embedding_base64(self, embedding: list[float]) -> str:
        packed = struct.pack(f"<{len(embedding)}f", *embedding)
        return base64.b64encode(packed).decode("ascii")

    def _build_embedding_stub_response(
        self,
        request: EmbeddingRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        inputs: list[str],
    ) -> dict[str, Any]:
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
        documents = request.documents if isinstance(request.documents, list) else [request.documents]
        if not documents:
            raise BackendRequestValidationError(
                "Rerank requests must include at least one document.",
                code="invalid_input",
            )
        return documents

    def _score_stub_document(self, query: str, document: str) -> float:
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
        return {
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "top_p": request.top_p if request.top_p is not None else 1.0,
            "max_tokens": request.max_tokens or 512,
        }

    def _build_chat_messages(self, request: ChatCompletionsRequest) -> list[dict[str, Any]]:
        if request.stream:
            raise BackendRequestValidationError(
                "Streaming chat completions are not supported in Phase 1.",
                code="unsupported_parameter",
            )

        messages: list[dict[str, Any]] = []
        for message in request.messages:
            if not isinstance(message.content, str):
                raise BackendRequestValidationError(
                    "Only text chat messages are supported in Phase 1.",
                    code="unsupported_message_content",
                )

            payload = {"role": message.role, "content": message.content}
            if message.name:
                payload["name"] = message.name
            messages.append(payload)
        return messages

    def _build_chat_stub_response(
        self,
        request: ChatCompletionsRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        sampling_params: dict[str, Any],
    ) -> dict[str, Any]:
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
                "messages": self._build_chat_messages(request),
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
        messages = self._build_chat_messages(request)
        sampling_params = self._build_sampling_params(request)
        if self.engine is None:
            return self._build_chat_stub_response(
                request,
                runtime_spec,
                runtime_context,
                sampling_params,
            )

        result = self.engine.chat(messages, **sampling_params)
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
