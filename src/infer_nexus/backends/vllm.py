"""vLLM backend adapter implementation."""

from __future__ import annotations

import importlib
import inspect
import logging
import re
import struct
from collections.abc import AsyncIterator
from time import time
from typing import Any
from uuid import uuid4

from infer_nexus.backends.base import InferenceBackend
from infer_nexus.backends.vllm_native import (
    DynamicVLLMOpenAIChatServingAdapter,
    DynamicVLLMOpenAIEmbeddingServingAdapter,
    OpenAIChatServingAdapter,
    OpenAIEmbeddingServingAdapter,
    OpenAIServingEngineClientCompatProxy,
    ResolvedOpenAIServingImports,
)
from infer_nexus.backends.vllm_local_best_effort import LocalBestEffortVLLMExecutor
from infer_nexus.catalog.models import ModelConfig
from infer_nexus.core.enums import CompatibilityMode
from infer_nexus.core.errors import BackendConfigurationError, BackendRequestValidationError
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest


logger = logging.getLogger(__name__)


class VLLMBackend(InferenceBackend):
    """Adapt local vLLM engine lifecycle and request handling."""

    TASK_TO_MODE = {
        "chat": "generate",
        "embedding": "embed",
        "rerank": "score",
    }
    REQUEST_DEFAULT_SAMPLING_KEYS = {
        "best_of",
        "frequency_penalty",
        "ignore_eos",
        "include_stop_str_in_output",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "min_p",
        "min_tokens",
        "n",
        "presence_penalty",
        "prompt_logprobs",
        "repetition_penalty",
        "seed",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "stop",
        "temperature",
        "top_k",
        "top_logprobs",
        "top_p",
        "truncate_prompt_tokens",
        "vllm_xargs",
    }
    TOOL_REQUEST_KEYS = {"parallel_tool_calls"}
    OPENAI_SERVING_ONLY_ENGINE_KEYS = {
        "enable_auto_tool_choice",
        "tool_call_parser",
    }
    REASONING_REQUEST_KEYS = {
        "chat_template",
        "chat_template_kwargs",
        "enable_reasoning",
        "enable_thinking",
        "reasoning",
        "reasoning_effort",
        "thinking",
    }
    UNSUPPORTED_KWARG_PATTERN = re.compile(r"Unexpected keyword argument '([^']+)'")

    def __init__(self, runtime_spec: dict[str, Any]) -> None:
        self.runtime_spec = runtime_spec
        self.engine: Any | None = None
        self.openai_serving_chat_adapter: OpenAIChatServingAdapter | None = None
        self.openai_serving_embedding_adapter: OpenAIEmbeddingServingAdapter | None = None
        self.openai_serving_adapter_init_error: str | None = None
        self.openai_serving_embedding_adapter_init_error: str | None = None
        self.async_engine_init_error: str | None = None
        self.engine_kind: str = "created"
        self.engine_state: str = "created"
        self.local_best_effort_executor = LocalBestEffortVLLMExecutor(self)

    def validate_runtime_spec(self, runtime_spec: dict[str, Any], runtime_context: dict[str, Any]) -> None:
        task = getattr(runtime_context.get("task"), "value", runtime_context.get("task"))
        if task not in self.TASK_TO_MODE:
            raise BackendConfigurationError(f"vLLM backend does not support task '{task}'.")

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

        if self._requires_openai_serving_adapter(runtime_spec):
            openai_serving = runtime_spec.get("openai_serving") or {}
            if runtime_spec.get("task_mode") == "generate" and not openai_serving.get("enabled"):
                raise BackendConfigurationError(
                    f"vllm_native local vLLM chat model '{runtime_context.get('model_name')}' "
                    "requires openai_serving.enabled=true."
                )
        if self._requires_openai_embedding_serving_adapter(runtime_spec):
            openai_serving = runtime_spec.get("openai_serving") or {}
            if runtime_spec.get("task_mode") == "embed" and not openai_serving.get("enabled"):
                raise BackendConfigurationError(
                    f"vllm_native local vLLM embedding model '{runtime_context.get('model_name')}' "
                    "requires openai_serving.enabled=true."
                )
        if self._is_vllm_native(runtime_spec) and runtime_spec.get("task_mode") == "score":
            raise BackendConfigurationError("vllm_native is not supported for vLLM rerank models.")

    def _filter_kwargs_for_callable(self, callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return dict(kwargs)

        parameters = signature.parameters
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return dict(kwargs)

        allowed_names = {
            name
            for name, parameter in parameters.items()
            if name != "self"
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        return {key: value for key, value in kwargs.items() if key in allowed_names}

    def _create_async_vllm_engine(self, llm_kwargs: dict[str, Any]) -> Any:
        try:
            try:
                from vllm.engine.arg_utils import AsyncEngineArgs
                from vllm.engine.async_llm_engine import AsyncLLMEngine
            except ImportError:
                from vllm import AsyncEngineArgs, AsyncLLMEngine  # type: ignore[attr-defined]
        except ImportError as exc:
            raise RuntimeError("This vLLM installation does not expose AsyncLLMEngine.") from exc

        engine_args_kwargs = self._filter_kwargs_for_callable(AsyncEngineArgs, llm_kwargs)
        engine_args = AsyncEngineArgs(**engine_args_kwargs)
        from_engine_args = getattr(AsyncLLMEngine, "from_engine_args", None)
        if callable(from_engine_args):
            return from_engine_args(engine_args)
        return AsyncLLMEngine(engine_args)

    def startup(self) -> None:
        init_mode = self.runtime_spec.get("backend_init_mode", "stub")
        if init_mode != "real":
            self.engine = None
            self.openai_serving_chat_adapter = None
            self.openai_serving_embedding_adapter = None
            self.openai_serving_adapter_init_error = None
            self.openai_serving_embedding_adapter_init_error = None
            self.async_engine_init_error = None
            self.engine_kind = "stub"
            self.engine_state = "stub"
            return

        llm_kwargs: dict[str, Any] = {
            "model": self.runtime_spec["model_path"],
            "tensor_parallel_size": self.runtime_spec["tensor_parallel_size"],
            "dtype": self.runtime_spec.get("dtype") or "auto",
        }
        loading_config = self.runtime_spec.get("model_loading_config") or {}
        if loading_config.get("revision"):
            llm_kwargs["revision"] = loading_config["revision"]
        engine_kwargs = dict(self.runtime_spec.get("engine_kwargs") or {})
        llm_kwargs.update(
            {
                key: value
                for key, value in engine_kwargs.items()
                if key not in self.OPENAI_SERVING_ONLY_ENGINE_KEYS
            }
        )
        gpu_memory_utilization = self.runtime_spec.get("gpu_memory_utilization")
        max_model_len = self.runtime_spec.get("max_model_len")
        requested_mode = self.runtime_spec.get("task_mode")

        if requested_mode == "generate":
            async_llm_kwargs = dict(llm_kwargs)
            if gpu_memory_utilization is not None:
                async_llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
            if max_model_len is not None:
                async_llm_kwargs["max_model_len"] = max_model_len
            async_llm_kwargs["task"] = requested_mode or "auto"
            try:
                self.engine = self._create_async_vllm_engine(async_llm_kwargs)
                self.async_engine_init_error = None
                self.engine_kind = "async"
                self.engine_state = "ready"
                self.openai_serving_chat_adapter = self._initialize_openai_serving_chat_adapter()
                self.openai_serving_embedding_adapter = None
                if self.openai_serving_chat_adapter is not None:
                    logger.info("Initialized vLLM OpenAI serving adapter for async engine.")
                elif self._requires_openai_serving_adapter():
                    self._raise_openai_serving_unavailable()
                elif self._should_use_openai_serving_adapter():
                    logger.warning(
                        "vLLM OpenAI serving adapter is disabled for async engine: %s",
                        self.openai_serving_adapter_init_error,
                    )
                return
            except BackendConfigurationError:
                raise
            except Exception as exc:
                self.async_engine_init_error = str(exc)

        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Install the 'vllm' extra or switch runtime.backend_init_mode to 'stub'."
            ) from exc

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
            if "task" in llm_init_args:
                llm_kwargs["task"] = requested_mode or "auto"
            elif (
                requested_mode in {"embed", "score"}
                and ("runner" in llm_init_args or accepts_var_kwargs)
            ):
                # Older vLLM releases use runner="pooling" for embedding / scoring
                # models instead of the newer task=... API.
                llm_kwargs["runner"] = "pooling"
        except (TypeError, ValueError):
            pass

        self.engine = LLM(**llm_kwargs)
        self.engine_kind = "sync"
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
        self.openai_serving_chat_adapter = self._initialize_openai_serving_chat_adapter()
        if self.openai_serving_chat_adapter is None and self._requires_openai_serving_adapter():
            self._raise_openai_serving_unavailable()
        if self.openai_serving_chat_adapter is not None:
            logger.info("Initialized vLLM OpenAI serving adapter for sync engine.")
        elif self._should_use_openai_serving_adapter():
            logger.warning(
                "vLLM OpenAI serving adapter is disabled for sync engine: %s",
                self.openai_serving_adapter_init_error,
            )

        self.openai_serving_embedding_adapter = self._initialize_openai_serving_embedding_adapter()
        if (
            self.openai_serving_embedding_adapter is None
            and self._requires_openai_embedding_serving_adapter()
        ):
            self._raise_openai_embedding_serving_unavailable()
        if self.openai_serving_embedding_adapter is not None:
            logger.info("Initialized vLLM OpenAI embeddings serving adapter for sync engine.")
        elif self._should_use_openai_serving_embedding_adapter():
            logger.warning(
                "vLLM OpenAI embeddings serving adapter is disabled for sync engine: %s",
                self.openai_serving_embedding_adapter_init_error,
            )
        self.engine_state = "ready"

    def shutdown(self) -> None:
        self.engine = None
        self.openai_serving_chat_adapter = None
        self.openai_serving_embedding_adapter = None
        self.openai_serving_adapter_init_error = None
        self.openai_serving_embedding_adapter_init_error = None
        self.async_engine_init_error = None
        self.engine_kind = "stopped"
        self.engine_state = "stopped"

    def build_runtime_spec(self, model: ModelConfig, resolved_model_reference: str) -> dict[str, Any]:
        task_mode = self.TASK_TO_MODE.get(model.task.value)
        if task_mode is None:
            raise BackendConfigurationError(
                f"vLLM backend does not support model task '{model.task.value}' for model '{model.name}'."
            )
        openai_serving_config = model.vllm.openai_serving.model_dump(mode="json")
        engine_kwargs = dict(model.vllm.engine_kwargs)
        if openai_serving_config.get("enabled"):
            if openai_serving_config.get("reasoning_parser"):
                engine_kwargs.setdefault("reasoning_parser", openai_serving_config["reasoning_parser"])
            if openai_serving_config.get("enable_reasoning"):
                engine_kwargs.setdefault("enable_reasoning", True)

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
            "engine_kwargs": engine_kwargs,
            "request_defaults": dict(model.vllm.request_defaults),
            "request_policy": model.vllm.request_policy.model_dump(mode="json"),
            "openai_serving": openai_serving_config,
            "compat_mode": model.compat_mode.value,
            "model_loading_config": model.model_loading_config.model_dump(mode="json"),
            "served_model_name": model.served_model_name or model.alias or model.name,
        }

    def _request_defaults(self, runtime_spec: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = runtime_spec or self.runtime_spec
        return dict(spec.get("request_defaults") or {})

    def _request_policy(self, runtime_spec: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = runtime_spec or self.runtime_spec
        return dict(spec.get("request_policy") or {})

    def _compat_mode(self, runtime_spec: dict[str, Any] | None = None) -> str:
        spec = runtime_spec or self.runtime_spec
        return str(spec.get("compat_mode") or CompatibilityMode.LOCAL_BEST_EFFORT.value)

    def _is_strict_openai(self, runtime_spec: dict[str, Any] | None = None) -> bool:
        return self._compat_mode(runtime_spec) == CompatibilityMode.STRICT_OPENAI.value

    def _is_vllm_native(self, runtime_spec: dict[str, Any] | None = None) -> bool:
        return self._compat_mode(runtime_spec) in {
            CompatibilityMode.VLLM_NATIVE.value,
            CompatibilityMode.STRICT_OPENAI.value,
        }

    def _requires_openai_serving_adapter(
        self,
        runtime_spec: dict[str, Any] | None = None,
    ) -> bool:
        spec = runtime_spec or self.runtime_spec
        return self._is_vllm_native(spec) and spec.get("task_mode") == "generate"

    def _requires_openai_embedding_serving_adapter(
        self,
        runtime_spec: dict[str, Any] | None = None,
    ) -> bool:
        spec = runtime_spec or self.runtime_spec
        return self._is_vllm_native(spec) and spec.get("task_mode") == "embed"

    def _raise_openai_serving_unavailable(self, reason: str | None = None) -> None:
        detail = reason or self.openai_serving_adapter_init_error or "adapter is not initialized"
        raise BackendConfigurationError(
            "vllm_native vLLM chat requires the replica-local vLLM OpenAI serving adapter, "
            f"but it is unavailable: {detail}"
        )

    def _raise_openai_embedding_serving_unavailable(self, reason: str | None = None) -> None:
        detail = (
            reason
            or self.openai_serving_embedding_adapter_init_error
            or "adapter is not initialized"
        )
        raise BackendConfigurationError(
            "vllm_native vLLM embedding requires the replica-local vLLM OpenAI embeddings "
            f"serving adapter, but it is unavailable: {detail}"
        )

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

    def _merge_request_extras(
        self,
        request: ChatCompletionsRequest,
        *,
        runtime_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.local_best_effort_executor._merge_request_extras(
            request,
            runtime_spec=runtime_spec,
        )

    def _build_sampling_params(
        self,
        request: ChatCompletionsRequest,
        *,
        runtime_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.local_best_effort_executor._build_sampling_params(
            request,
            runtime_spec=runtime_spec,
        )

    def _build_chat_kwargs(
        self,
        request: ChatCompletionsRequest,
        *,
        runtime_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.local_best_effort_executor._build_chat_kwargs(
            request,
            runtime_spec=runtime_spec,
        )

    def _build_sampling_params_instance(
        self,
        sampling_params: dict[str, Any],
        *,
        sampling_params_cls: type[Any],
    ) -> Any:
        return self.local_best_effort_executor._build_sampling_params_instance(
            sampling_params,
            sampling_params_cls=sampling_params_cls,
        )

    def _filter_sampling_params_for_vllm(
        self,
        sampling_params: dict[str, Any],
        *,
        sampling_params_cls: type[Any],
    ) -> dict[str, Any]:
        return self.local_best_effort_executor._filter_sampling_params_for_vllm(
            sampling_params,
            sampling_params_cls=sampling_params_cls,
        )

    def _filter_chat_kwargs_for_vllm(self, chat_kwargs: dict[str, Any]) -> dict[str, Any]:
        return self.local_best_effort_executor._filter_chat_kwargs_for_vllm(chat_kwargs)

    def _is_async_engine(self) -> bool:
        return self.local_best_effort_executor._is_async_engine()

    async def _maybe_await(self, value: Any) -> Any:
        return await self.local_best_effort_executor._maybe_await(value)

    async def _get_async_engine_tokenizer(self) -> Any:
        return await self.local_best_effort_executor._get_async_engine_tokenizer()

    def _load_async_engine_image_asset(self, image_url: str) -> Any:
        return self.local_best_effort_executor._load_async_engine_image_asset(image_url)

    def _build_async_engine_multi_modal_data(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        return self.local_best_effort_executor._build_async_engine_multi_modal_data(messages)

    def _build_async_engine_generate_input(
        self,
        generate: Any,
        prompt: str,
        multi_modal_data: dict[str, Any] | None,
    ) -> tuple[Any, dict[str, Any]]:
        return self.local_best_effort_executor._build_async_engine_generate_input(
            generate,
            prompt,
            multi_modal_data,
        )

    async def _build_async_engine_chat_prompt(
        self,
        messages: list[dict[str, Any]],
        chat_kwargs: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        return await self.local_best_effort_executor._build_async_engine_chat_prompt(
            messages,
            chat_kwargs,
        )

    def _build_async_engine_sampling_params_instance(self, sampling_params: dict[str, Any]) -> Any:
        return self.local_best_effort_executor._build_async_engine_sampling_params_instance(
            sampling_params
        )

    async def _invoke_async_engine_chat_stream(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        *,
        chat_kwargs: dict[str, Any],
        request_id: str,
    ) -> Any:
        return await self.local_best_effort_executor._invoke_async_engine_chat_stream(
            messages,
            sampling_params,
            chat_kwargs=chat_kwargs,
            request_id=request_id,
        )

    def _supports_multimodal(self, runtime_context: dict[str, Any] | None = None) -> bool:
        return self.local_best_effort_executor._supports_multimodal(runtime_context)

    def _normalize_data_url(self, url: str) -> str:
        return self.local_best_effort_executor._normalize_data_url(url)

    def _serialize_content_block(self, block: Any) -> dict[str, Any]:
        return self.local_best_effort_executor._serialize_content_block(block)

    def _is_text_only_content(self, content: Any) -> bool:
        return self.local_best_effort_executor._is_text_only_content(content)

    def _collapse_text_only_content(self, content: list[Any]) -> str:
        return self.local_best_effort_executor._collapse_text_only_content(content)

    def _serialize_message_content(
        self,
        content: Any,
        *,
        allow_multimodal: bool,
    ) -> str | list[dict[str, Any]]:
        return self.local_best_effort_executor._serialize_message_content(
            content,
            allow_multimodal=allow_multimodal,
        )

    def _build_chat_messages(
        self,
        request: ChatCompletionsRequest,
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.local_best_effort_executor._build_chat_messages(
            request,
            runtime_context=runtime_context,
        )

    def _should_use_openai_serving_adapter(
        self,
        runtime_spec: dict[str, Any] | None = None,
    ) -> bool:
        spec = runtime_spec or self.runtime_spec
        openai_serving = spec.get("openai_serving") or {}
        return spec.get("task_mode") == "generate" and bool(openai_serving.get("enabled"))

    def _should_use_openai_serving_embedding_adapter(
        self,
        runtime_spec: dict[str, Any] | None = None,
    ) -> bool:
        spec = runtime_spec or self.runtime_spec
        openai_serving = spec.get("openai_serving") or {}
        return spec.get("task_mode") == "embed" and bool(openai_serving.get("enabled"))

    def _import_vllm_symbol(self, module_path: str, symbol_name: str) -> Any:
        module = importlib.import_module(module_path)
        return getattr(module, symbol_name)

    def _resolve_openai_serving_imports(self) -> ResolvedOpenAIServingImports:
        candidates = [
            {
                "chat_request": (
                    "vllm.entrypoints.openai.chat_completion.protocol",
                    "ChatCompletionRequest",
                ),
                "serving_chat": (
                    "vllm.entrypoints.openai.chat_completion.serving",
                    "OpenAIServingChat",
                ),
                "serving_models": (
                    "vllm.entrypoints.openai.models.serving",
                    "OpenAIServingModels",
                ),
                "base_model_path": (
                    "vllm.entrypoints.openai.models.protocol",
                    "BaseModelPath",
                ),
                "serving_render": (
                    "vllm.entrypoints.serve.render.serving",
                    "OpenAIServingRender",
                ),
            },
            {
                "chat_request": ("vllm.entrypoints.openai.protocol", "ChatCompletionRequest"),
                "serving_chat": ("vllm.entrypoints.openai.serving_chat", "OpenAIServingChat"),
                "serving_models": ("vllm.entrypoints.openai.serving_models", "OpenAIServingModels"),
                "base_model_path": ("vllm.entrypoints.openai.serving_models", "BaseModelPath"),
                "serving_render": None,
            },
        ]

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                serving_render_cls = None
                if candidate["serving_render"] is not None:
                    serving_render_cls = self._import_vllm_symbol(*candidate["serving_render"])
                return ResolvedOpenAIServingImports(
                    chat_request_cls=self._import_vllm_symbol(*candidate["chat_request"]),
                    serving_chat_cls=self._import_vllm_symbol(*candidate["serving_chat"]),
                    serving_models_cls=self._import_vllm_symbol(*candidate["serving_models"]),
                    base_model_path_cls=self._import_vllm_symbol(*candidate["base_model_path"]),
                    serving_render_cls=serving_render_cls,
                )
            except (ImportError, AttributeError) as exc:
                last_error = exc

        if last_error is None:
            raise ImportError("No supported vLLM OpenAI serving import layout was found.")
        raise ImportError(
            f"Unable to resolve supported vLLM OpenAI serving imports: {last_error}"
        ) from last_error

    def _resolve_openai_serving_engine_client(self) -> Any | None:
        def _is_compatible(client: Any) -> bool:
            return hasattr(client, "model_config")

        candidates = [
            getattr(self.engine, "engine_client", None),
            getattr(self.engine, "async_engine_client", None),
            getattr(self.engine, "llm_engine", None),
            getattr(self.engine, "engine", None),
            getattr(self.engine, "_engine", None),
            self.engine,
        ]
        unique_candidates: list[Any] = []
        seen_ids: set[int] = set()
        for candidate in candidates:
            if candidate is None:
                continue
            candidate_id = id(candidate)
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            unique_candidates.append(candidate)

        for candidate in unique_candidates:
            if _is_compatible(candidate):
                if hasattr(candidate, "errored") and hasattr(candidate, "generate"):
                    return candidate
                fallbacks = [item for item in unique_candidates if item is not candidate]
                return OpenAIServingEngineClientCompatProxy(candidate, fallbacks)
        return None

    def _build_openai_serving_base_model_path(self, base_model_path_cls: type[Any]) -> Any:
        served_model_name = (
            self.runtime_spec.get("served_model_name")
            or self.runtime_spec.get("model_name")
            or self.runtime_spec.get("model_path")
        )
        model_path = self.runtime_spec["model_path"]
        kwargs: dict[str, Any] = {}
        try:
            signature = inspect.signature(base_model_path_cls)
        except (TypeError, ValueError):
            signature = None

        if signature is None:
            return base_model_path_cls(name=served_model_name, model_path=model_path)

        for parameter_name in signature.parameters:
            if parameter_name == "self":
                continue
            if parameter_name == "name":
                kwargs[parameter_name] = served_model_name
            elif parameter_name in {"model_path", "path", "root"}:
                kwargs[parameter_name] = model_path

        return base_model_path_cls(**kwargs)

    def _build_openai_serving_models(
        self,
        *,
        serving_models_cls: type[Any],
        base_model_path_cls: type[Any],
        engine_client: Any,
    ) -> Any:
        return serving_models_cls(
            engine_client,
            [self._build_openai_serving_base_model_path(base_model_path_cls)],
        )

    def _build_openai_serving_render(
        self,
        *,
        serving_render_cls: type[Any],
        engine_client: Any,
        serving_models: Any,
    ) -> Any:
        model_registry = getattr(serving_models, "registry", None)
        if model_registry is None:
            raise RuntimeError("OpenAIServingModels did not expose a model registry.")

        engine_kwargs = self.runtime_spec.get("engine_kwargs") or {}
        openai_serving_config = self.runtime_spec.get("openai_serving") or {}
        tool_call_parser = engine_kwargs.get("tool_call_parser")
        enable_auto_tools = bool(engine_kwargs.get("enable_auto_tool_choice"))
        reasoning_parser = (
            engine_kwargs.get("reasoning_parser")
            or openai_serving_config.get("reasoning_parser")
            or None
        )

        init_signature = inspect.signature(serving_render_cls)
        parameters = init_signature.parameters
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        kwargs = {
            "model_config": getattr(engine_client, "model_config"),
            "renderer": getattr(engine_client, "renderer"),
            "io_processor": getattr(engine_client, "io_processor", None),
            "model_registry": model_registry,
            "request_logger": None,
            "chat_template": None,
            "chat_template_content_format": "auto",
            "trust_request_chat_template": False,
            "enable_auto_tools": enable_auto_tools,
            "tool_parser": tool_call_parser,
            "reasoning_parser": reasoning_parser,
            "default_chat_template_kwargs": self._request_defaults().get("chat_template_kwargs"),
        }
        if not accepts_var_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in parameters}

        return serving_render_cls(**kwargs)

    def _build_openai_serving_chat(
        self,
        *,
        serving_chat_cls: type[Any],
        engine_client: Any,
        serving_models: Any,
        serving_render: Any | None,
    ) -> Any:
        openai_serving_config = self.runtime_spec.get("openai_serving") or {}
        engine_kwargs = self.runtime_spec.get("engine_kwargs") or {}
        enable_auto_tool_choice = engine_kwargs.get("enable_auto_tool_choice")
        tool_call_parser = engine_kwargs.get("tool_call_parser")
        reasoning_parser = (
            engine_kwargs.get("reasoning_parser")
            or openai_serving_config.get("reasoning_parser")
            or ""
        )
        init_signature = inspect.signature(serving_chat_cls)
        parameters = init_signature.parameters
        kwargs: dict[str, Any] = {}
        args: list[Any] = []

        if "engine_client" in parameters:
            args.append(engine_client)
        if "model_config" in parameters:
            args.append(getattr(engine_client, "model_config"))
        if "models" in parameters:
            args.append(serving_models)
        if "response_role" in parameters:
            args.append("assistant")

        optional_kwargs = {
            "openai_serving_render": serving_render,
            "request_logger": None,
            "chat_template": None,
            "chat_template_content_format": "auto",
            "trust_request_chat_template": False,
            "return_tokens_as_token_ids": False,
            "enable_auto_tool_choice": enable_auto_tool_choice,
            "enable_auto_tools": enable_auto_tool_choice,
            "tool_call_parser": tool_call_parser,
            "tool_parser": tool_call_parser,
            "reasoning_parser": reasoning_parser,
            "default_chat_template_kwargs": self._request_defaults().get("chat_template_kwargs"),
        }
        skip_if_none = {
            "enable_auto_tool_choice",
            "enable_auto_tools",
            "tool_call_parser",
            "tool_parser",
            "default_chat_template_kwargs",
        }
        for key, value in optional_kwargs.items():
            if key in parameters and (value is not None or key not in skip_if_none):
                kwargs[key] = value

        return serving_chat_cls(*args, **kwargs)

    def _initialize_openai_serving_chat_adapter(self) -> OpenAIChatServingAdapter | None:
        if not self._should_use_openai_serving_adapter():
            self.openai_serving_adapter_init_error = None
            return None
        if self.engine is None:
            self.openai_serving_adapter_init_error = (
                "vLLM engine is not initialized, so OpenAI serving cannot be constructed."
            )
            return None

        try:
            imports = self._resolve_openai_serving_imports()
            engine_client = self._resolve_openai_serving_engine_client()
            if engine_client is None:
                raise RuntimeError("No compatible engine client was found on the initialized vLLM engine.")
            serving_models = self._build_openai_serving_models(
                serving_models_cls=imports.serving_models_cls,
                base_model_path_cls=imports.base_model_path_cls,
                engine_client=engine_client,
            )
            serving_render = None
            if imports.serving_render_cls is not None:
                serving_render = self._build_openai_serving_render(
                    serving_render_cls=imports.serving_render_cls,
                    engine_client=engine_client,
                    serving_models=serving_models,
                )
            serving_chat = self._build_openai_serving_chat(
                serving_chat_cls=imports.serving_chat_cls,
                engine_client=engine_client,
                serving_models=serving_models,
                serving_render=serving_render,
            )
        except Exception as exc:
            self.openai_serving_adapter_init_error = str(exc)
            logger.warning(
                "Failed to initialize vLLM OpenAI serving adapter for model_path='%s' "
                "served_model_name='%s' engine_kind='%s': %s",
                self.runtime_spec.get("model_path"),
                self.runtime_spec.get("served_model_name"),
                self.engine_kind,
                exc,
            )
            return None

        self.openai_serving_adapter_init_error = None
        return DynamicVLLMOpenAIChatServingAdapter(
            chat_request_cls=imports.chat_request_cls,
            serving_chat=serving_chat,
        )

    def _resolve_openai_serving_embedding_imports(self) -> tuple[type[Any], type[Any]]:
        candidates = [
            (
                ("vllm.entrypoints.pooling.embed.protocol", "EmbeddingCompletionRequest"),
                ("vllm.entrypoints.pooling.embed.serving", "ServingEmbedding"),
            ),
            (
                ("vllm.entrypoints.openai.protocol", "EmbeddingCompletionRequest"),
                ("vllm.entrypoints.pooling.embed.serving", "ServingEmbedding"),
            ),
        ]

        last_error: Exception | None = None
        for request_candidate, serving_candidate in candidates:
            try:
                return (
                    self._import_vllm_symbol(*request_candidate),
                    self._import_vllm_symbol(*serving_candidate),
                )
            except (ImportError, AttributeError) as exc:
                last_error = exc

        if last_error is None:
            raise ImportError("No supported vLLM embeddings serving import layout was found.")
        raise ImportError(
            f"Unable to resolve supported vLLM embeddings serving imports: {last_error}"
        ) from last_error

    def _build_openai_serving_embedding(
        self,
        *,
        serving_embedding_cls: type[Any],
        engine_client: Any,
        serving_models: Any,
    ) -> Any:
        init_signature = inspect.signature(serving_embedding_cls)
        parameters = init_signature.parameters
        kwargs: dict[str, Any] = {}
        args: list[Any] = []

        if "engine_client" in parameters:
            args.append(engine_client)
        if "models" in parameters:
            args.append(serving_models)

        optional_kwargs = {
            "supported_tasks": ("embed",),
            "request_logger": None,
            "chat_template": None,
            "chat_template_content_format": "auto",
            "trust_request_chat_template": False,
            "return_tokens_as_token_ids": False,
            "log_error_stack": False,
        }
        for key, value in optional_kwargs.items():
            if key in parameters:
                kwargs[key] = value

        return serving_embedding_cls(*args, **kwargs)

    def _initialize_openai_serving_embedding_adapter(self) -> OpenAIEmbeddingServingAdapter | None:
        if not self._should_use_openai_serving_embedding_adapter():
            self.openai_serving_embedding_adapter_init_error = None
            return None
        if self.engine is None:
            self.openai_serving_embedding_adapter_init_error = (
                "vLLM engine is not initialized, so OpenAI embeddings serving cannot be constructed."
            )
            return None

        try:
            embedding_request_cls, serving_embedding_cls = (
                self._resolve_openai_serving_embedding_imports()
            )
            imports = self._resolve_openai_serving_imports()
            engine_client = self._resolve_openai_serving_engine_client()
            if engine_client is None:
                raise RuntimeError("No compatible engine client was found on the initialized vLLM engine.")
            serving_models = self._build_openai_serving_models(
                serving_models_cls=imports.serving_models_cls,
                base_model_path_cls=imports.base_model_path_cls,
                engine_client=engine_client,
            )
            serving_embedding = self._build_openai_serving_embedding(
                serving_embedding_cls=serving_embedding_cls,
                engine_client=engine_client,
                serving_models=serving_models,
            )
        except Exception as exc:
            self.openai_serving_embedding_adapter_init_error = str(exc)
            logger.warning(
                "Failed to initialize vLLM OpenAI embeddings serving adapter for "
                "model_path='%s' served_model_name='%s' engine_kind='%s': %s",
                self.runtime_spec.get("model_path"),
                self.runtime_spec.get("served_model_name"),
                self.engine_kind,
                exc,
            )
            return None

        self.openai_serving_embedding_adapter_init_error = None
        return DynamicVLLMOpenAIEmbeddingServingAdapter(
            embedding_request_cls=embedding_request_cls,
            serving_embedding=serving_embedding,
        )

    def _build_openai_serving_request_payload(
        self,
        request: ChatCompletionsRequest,
        *,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        payload = request.model_dump(
            mode="json",
            exclude_none=True,
            exclude={"extra_body", "messages"},
        )
        payload["messages"] = []
        for message in request.messages:
            serialized_message = message.model_dump(mode="json", exclude_none=True)
            if message.content is None:
                serialized_message["content"] = None
            payload["messages"].append(serialized_message)
        payload.update(request.extra_body or {})
        payload["model"] = (
            runtime_context.get("served_model_name")
            or runtime_spec.get("served_model_name")
            or request.model
        )
        return payload

    def _build_openai_serving_embedding_request_payload(
        self,
        request: EmbeddingRequest,
        *,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        payload["model"] = (
            runtime_context.get("served_model_name")
            or runtime_spec.get("served_model_name")
            or request.model
        )
        return payload

    async def _call_openai_serving_chat_completion(
        self,
        request_payload: dict[str, Any],
    ) -> dict[str, Any]:
        adapter = self.openai_serving_chat_adapter
        if adapter is None:
            raise RuntimeError("vLLM OpenAI serving adapter is not initialized")

        result = adapter.chat_completion(request_payload)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _iter_openai_serving_stream(
        self,
        request_payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
        adapter = self.openai_serving_chat_adapter
        if adapter is None:
            raise RuntimeError("vLLM OpenAI serving adapter is not initialized")

        stream = adapter.chat_completion_stream(request_payload)
        if inspect.isawaitable(stream):
            stream = await stream

        if hasattr(stream, "__aiter__"):
            async for chunk in stream:
                yield chunk
            return

        if isinstance(stream, bytes | str | dict):
            yield stream
            return

        for chunk in stream:
            yield chunk

    async def _call_openai_serving_embedding(
        self,
        request_payload: dict[str, Any],
    ) -> dict[str, Any]:
        adapter = self.openai_serving_embedding_adapter
        if adapter is None:
            raise RuntimeError("vLLM OpenAI embeddings serving adapter is not initialized")

        result = adapter.embedding(request_payload)
        if inspect.isawaitable(result):
            result = await result
        return result

    def _build_chat_stub_response(
        self,
        request: ChatCompletionsRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        sampling_params: dict[str, Any],
        chat_kwargs: dict[str, Any],
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
                "messages": self._build_chat_messages(request, runtime_context=runtime_context),
                "sampling_params": sampling_params,
                "chat_kwargs": chat_kwargs,
            },
        }

    # NOTE: Temporarily commented out because this helper is currently unused.
    # Keeping the code here for short-term rollback safety during module cleanup.
    #
    # def _build_chat_stub_stream_chunks(
    #     self,
    #     request: ChatCompletionsRequest,
    #     runtime_spec: dict[str, Any],
    #     runtime_context: dict[str, Any],
    #     sampling_params: dict[str, Any],
    #     chat_kwargs: dict[str, Any],
    # ) -> list[dict[str, Any]]:
    #     response = self._build_chat_stub_response(
    #         request,
    #         runtime_spec,
    #         runtime_context,
    #         sampling_params,
    #         chat_kwargs,
    #     )
    #     created = response["created"]
    #     response_id = response["id"]
    #     model = response["model"]
    #     return [
    #         {
    #             "id": response_id,
    #             "object": "chat.completion.chunk",
    #             "created": created,
    #             "model": model,
    #             "choices": [
    #                 {
    #                     "index": 0,
    #                     "delta": {
    #                         "role": "assistant",
    #                         "content": response["content"],
    #                     },
    #                     "finish_reason": None,
    #                 }
    #             ],
    #         },
    #         {
    #             "id": response_id,
    #             "object": "chat.completion.chunk",
    #             "created": created,
    #             "model": model,
    #             "choices": [
    #                 {
    #                     "index": 0,
    #                     "delta": {},
    #                     "finish_reason": response["finish_reason"],
    #                 }
    #             ],
    #         },
    #     ]

    def _convert_chat_result(
        self,
        *,
        request: ChatCompletionsRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        result: Any,
        sampling_params: dict[str, Any],
        chat_kwargs: dict[str, Any],
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
                "chat_kwargs": chat_kwargs,
            },
        }

    async def chat_completion(
        self,
        runtime_spec: dict[str, Any],
        request: ChatCompletionsRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        vllm_native = self._is_vllm_native(runtime_spec)
        if self.openai_serving_chat_adapter is not None:
            request_payload = self._build_openai_serving_request_payload(
                request,
                runtime_spec=runtime_spec,
                runtime_context=runtime_context,
            )
            try:
                return await self._call_openai_serving_chat_completion(request_payload)
            except Exception as exc:
                if vllm_native:
                    logger.exception(
                        "Native vLLM OpenAI serving chat invocation failed for model '%s' "
                        "served_model_name '%s'.",
                        runtime_context.get("model_name"),
                        runtime_context.get("served_model_name") or runtime_spec.get("served_model_name"),
                    )
                    raise
                self.openai_serving_adapter_init_error = str(exc)
                self.openai_serving_chat_adapter = None
        elif vllm_native:
            self._raise_openai_serving_unavailable()
        return await self.local_best_effort_executor.chat_completion(
            runtime_spec,
            request,
            runtime_context,
        )

    async def chat_completion_stream(
        self,
        runtime_spec: dict[str, Any],
        request: ChatCompletionsRequest,
        runtime_context: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
        vllm_native = self._is_vllm_native(runtime_spec)
        if self.openai_serving_chat_adapter is not None:
            request_payload = self._build_openai_serving_request_payload(
                request,
                runtime_spec=runtime_spec,
                runtime_context=runtime_context,
            )
            try:
                async for chunk in self._iter_openai_serving_stream(request_payload):
                    yield chunk
                return
            except Exception as exc:
                if vllm_native:
                    logger.exception(
                        "Native vLLM OpenAI serving stream invocation failed for model '%s' "
                        "served_model_name '%s'.",
                        runtime_context.get("model_name"),
                        runtime_context.get("served_model_name") or runtime_spec.get("served_model_name"),
                    )
                    raise
                self.openai_serving_adapter_init_error = str(exc)
                self.openai_serving_chat_adapter = None
        elif vllm_native:
            self._raise_openai_serving_unavailable()
        async for event in self.local_best_effort_executor.chat_completion_stream(
            runtime_spec,
            request,
            runtime_context,
        ):
            yield event

    async def embedding(
        self,
        runtime_spec: dict[str, Any],
        request: EmbeddingRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        if self._is_vllm_native(runtime_spec):
            if self.openai_serving_embedding_adapter is None:
                self._raise_openai_embedding_serving_unavailable()
            request_payload = self._build_openai_serving_embedding_request_payload(
                request,
                runtime_spec=runtime_spec,
                runtime_context=runtime_context,
            )
            return await self._call_openai_serving_embedding(request_payload)

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
        if self._is_vllm_native(runtime_spec):
            raise BackendConfigurationError("vllm_native is not supported for VLLM rerank.")

        documents = self._normalize_rerank_documents(request)
        if self.engine is None:
            return self._build_rerank_stub_response(
                request,
                runtime_spec,
                runtime_context,
                documents,
            )

        score_kwargs: dict[str, Any] = {}
        score_signature = inspect.signature(self.engine.score)
        score_args = score_signature.parameters
        score_accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in score_args.values()
        )

        chat_template = (runtime_spec.get("engine_kwargs") or {}).get("chat_template")
        if chat_template and ("chat_template" in score_args or score_accepts_var_kwargs):
            score_kwargs["chat_template"] = chat_template

        result = self.engine.score(request.query, documents, **score_kwargs)
        return self._convert_rerank_result(
            request=request,
            runtime_spec=runtime_spec,
            runtime_context=runtime_context,
            documents=documents,
            result=result,
        )
