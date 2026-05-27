"""vLLM backend adapter implementation."""

from __future__ import annotations

import base64
import importlib
import inspect
import logging
import re
import struct
from io import BytesIO
from pathlib import Path
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from time import time
from typing import Any, Protocol
from uuid import uuid4
from urllib.parse import urlparse
from urllib.request import urlopen

from infer_nexus.backends.base import InferenceBackend
from infer_nexus.catalog.models import ModelConfig
from infer_nexus.core.errors import BackendConfigurationError, BackendRequestValidationError
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest


logger = logging.getLogger(__name__)


class OpenAIChatServingAdapter(Protocol):
    """Backend-local contract for vLLM OpenAI serving passthrough."""

    async def chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Return a full OpenAI-compatible chat completion payload."""

    async def chat_completion_stream(
        self,
        request_payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
        """Yield OpenAI-compatible chat completion chunks."""


@dataclass(frozen=True)
class ResolvedOpenAIServingImports:
    """Resolved vLLM serving imports for one supported internal layout."""

    chat_request_cls: type[Any]
    serving_chat_cls: type[Any]
    serving_models_cls: type[Any]
    base_model_path_cls: type[Any]
    serving_render_cls: type[Any] | None = None


class DynamicVLLMOpenAIChatServingAdapter:
    """Wrap native vLLM serving objects behind the local adapter contract."""

    def __init__(
        self,
        *,
        chat_request_cls: type[Any],
        serving_chat: Any,
    ) -> None:
        self.chat_request_cls = chat_request_cls
        self.serving_chat = serving_chat

    def _build_request(self, request_payload: dict[str, Any]) -> Any:
        if hasattr(self.chat_request_cls, "model_validate"):
            return self.chat_request_cls.model_validate(request_payload)
        return self.chat_request_cls(**request_payload)

    def _normalize_payload(self, payload: Any) -> dict[str, Any] | bytes | str:
        if isinstance(payload, dict | bytes | str):
            return payload
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="json", exclude_none=True)
        return payload

    async def chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        request = self._build_request(request_payload)
        result = self.serving_chat.create_chat_completion(request, raw_request=None)
        if inspect.isawaitable(result):
            result = await result
        if hasattr(result, "__aiter__"):
            raise RuntimeError(
                "vLLM OpenAI serving returned a stream for a non-streaming chat request."
            )
        normalized = self._normalize_payload(result)
        if not isinstance(normalized, dict):
            raise RuntimeError(
                "vLLM OpenAI serving returned an unexpected non-streaming chat payload type."
            )
        return normalized

    async def chat_completion_stream(
        self,
        request_payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
        request = self._build_request(request_payload)
        result = self.serving_chat.create_chat_completion(request, raw_request=None)
        if inspect.isawaitable(result):
            result = await result

        if hasattr(result, "__aiter__"):
            async for chunk in result:
                yield self._normalize_payload(chunk)
            return

        yield self._normalize_payload(result)


class OpenAIServingEngineClientCompatProxy:
    """Compatibility proxy that augments engine clients with required serving attributes."""

    def __init__(self, primary_client: Any, fallback_clients: list[Any]) -> None:
        self._client = primary_client
        self._fallback_clients = [candidate for candidate in fallback_clients if candidate is not None]

    @property
    def errored(self) -> bool:
        for candidate in [self._client, *self._fallback_clients]:
            if hasattr(candidate, "errored"):
                return bool(getattr(candidate, "errored"))
        return False

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        def _wrap_iterable(iterable: Iterable[Any]) -> AsyncIterator[Any]:
            async def iterator() -> AsyncIterator[Any]:
                for item in iterable:
                    yield item

            return iterator()

        def _wrap_awaitable(awaitable: Any) -> AsyncIterator[Any]:
            async def iterator() -> AsyncIterator[Any]:
                resolved = await awaitable
                if hasattr(resolved, "__aiter__"):
                    async for item in resolved:
                        yield item
                    return
                if isinstance(resolved, Iterable) and not isinstance(resolved, (bytes, str, dict)):
                    for item in resolved:
                        yield item
                    return
                yield resolved

            return iterator()

        for candidate in [self._client, *self._fallback_clients]:
            method = getattr(candidate, "generate", None)
            if callable(method):
                call_args = args
                call_kwargs = kwargs
                try:
                    signature = inspect.signature(method)
                except (TypeError, ValueError):
                    signature = None

                if signature is not None:
                    parameters = signature.parameters
                    accepts_var_args = any(
                        parameter.kind == inspect.Parameter.VAR_POSITIONAL
                        for parameter in parameters.values()
                    )
                    if not accepts_var_args:
                        positional_limit = sum(
                            1
                            for parameter in parameters.values()
                            if parameter.kind
                            in (
                                inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            )
                        )
                        call_args = args[:positional_limit]

                    accepts_var_kwargs = any(
                        parameter.kind == inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters.values()
                    )
                    if not accepts_var_kwargs:
                        allowed_names = {
                            name
                            for name, parameter in parameters.items()
                            if parameter.kind
                            in (
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                inspect.Parameter.KEYWORD_ONLY,
                            )
                        }
                        call_kwargs = {
                            key: value
                            for key, value in kwargs.items()
                            if key in allowed_names
                        }

                result = method(*call_args, **call_kwargs)
                if hasattr(result, "__aiter__"):
                    return result
                if inspect.isawaitable(result):
                    return _wrap_awaitable(result)
                if isinstance(result, Iterable) and not isinstance(result, (bytes, str, dict)):
                    return _wrap_iterable(result)
                return result
        raise AttributeError("No compatible 'generate' method found on engine client candidates.")

    def __getattr__(self, name: str) -> Any:
        if hasattr(self._client, name):
            return getattr(self._client, name)
        for candidate in self._fallback_clients:
            if hasattr(candidate, name):
                return getattr(candidate, name)
        raise AttributeError(name)


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
        self.openai_serving_adapter_init_error: str | None = None
        self.async_engine_init_error: str | None = None
        self.engine_kind: str = "created"
        self.engine_state: str = "created"

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
            self.openai_serving_adapter_init_error = None
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
                if self.openai_serving_chat_adapter is not None:
                    logger.info("Initialized vLLM OpenAI serving adapter for async engine.")
                elif self._should_use_openai_serving_adapter():
                    logger.warning(
                        "vLLM OpenAI serving adapter is disabled for async engine: %s",
                        self.openai_serving_adapter_init_error,
                    )
                return
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
        self.engine_state = "ready"

    def shutdown(self) -> None:
        self.engine = None
        self.openai_serving_chat_adapter = None
        self.openai_serving_adapter_init_error = None
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
            "model_loading_config": model.model_loading_config.model_dump(mode="json"),
            "served_model_name": model.served_model_name or model.alias or model.name,
        }

    def _request_defaults(self, runtime_spec: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = runtime_spec or self.runtime_spec
        return dict(spec.get("request_defaults") or {})

    def _request_policy(self, runtime_spec: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = runtime_spec or self.runtime_spec
        return dict(spec.get("request_policy") or {})

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
        merged = dict(self._request_defaults(runtime_spec))
        model_extra = getattr(request, "model_extra", None) or {}
        merged.update(model_extra)
        merged.update(request.extra_body or {})
        return merged

    def _build_sampling_params(
        self,
        request: ChatCompletionsRequest,
        *,
        runtime_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged = self._merge_request_extras(request, runtime_spec=runtime_spec)
        max_tokens = (
            request.max_tokens
            or merged.pop("max_tokens", None)
            or request.max_completion_tokens
            or merged.get("max_completion_tokens")
            or 512
        )
        params: dict[str, Any] = {
            "temperature": (
                request.temperature if request.temperature is not None else merged.get("temperature", 0.7)
            ),
            "top_p": request.top_p if request.top_p is not None else merged.get("top_p", 1.0),
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
            fallback = merged.get(key)
            if value is not None:
                params[key] = value
            elif fallback is not None:
                params[key] = fallback

        for key in self.REQUEST_DEFAULT_SAMPLING_KEYS:
            if key in {"max_tokens", "max_completion_tokens", "temperature", "top_p", "vllm_xargs"}:
                continue
            value = merged.get(key)
            if value is not None and key not in params:
                params[key] = value

        vllm_xargs = merged.get("vllm_xargs")
        if isinstance(vllm_xargs, dict):
            no_repeat_ngram_size = vllm_xargs.get("no_repeat_ngram_size")
            if no_repeat_ngram_size is not None:
                params["no_repeat_ngram_size"] = no_repeat_ngram_size

        return params

    def _build_chat_kwargs(
        self,
        request: ChatCompletionsRequest,
        *,
        runtime_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged = self._merge_request_extras(request, runtime_spec=runtime_spec)
        policy = self._request_policy(runtime_spec)
        allow_tools = bool(policy.get("allow_tools"))
        allow_reasoning = bool(policy.get("allow_reasoning"))
        passthrough_unknown = bool(policy.get("passthrough_unknown_openai_fields"))

        if not allow_tools and (request.tools is not None or request.tool_choice is not None):
            raise BackendRequestValidationError(
                "This model does not allow tool-calling request fields.",
                code="unsupported_parameter",
            )

        reasoning_keys_present = [
            key for key in self.REASONING_REQUEST_KEYS if merged.get(key) is not None
        ]
        if not allow_reasoning and reasoning_keys_present:
            raise BackendRequestValidationError(
                "This model does not allow reasoning request fields.",
                code="unsupported_parameter",
            )

        chat_kwargs: dict[str, Any] = {}
        if request.tools is not None:
            chat_kwargs["tools"] = request.tools
        elif allow_tools and merged.get("tools") is not None:
            chat_kwargs["tools"] = merged.get("tools")

        if request.tool_choice is not None:
            chat_kwargs["tool_choice"] = request.tool_choice
        elif allow_tools and merged.get("tool_choice") is not None:
            chat_kwargs["tool_choice"] = merged.get("tool_choice")

        if request.response_format is not None:
            chat_kwargs["response_format"] = request.response_format

        if allow_tools:
            if request.parallel_tool_calls is not None:
                chat_kwargs["parallel_tool_calls"] = request.parallel_tool_calls
            for key in self.TOOL_REQUEST_KEYS:
                value = merged.get(key)
                if value is not None:
                    chat_kwargs[key] = value

        if allow_reasoning:
            for key in self.REASONING_REQUEST_KEYS:
                value = merged.get(key)
                if value is not None:
                    chat_kwargs[key] = value

        if passthrough_unknown:
            excluded_keys = (
                self.REQUEST_DEFAULT_SAMPLING_KEYS
                | self.TOOL_REQUEST_KEYS
                | self.REASONING_REQUEST_KEYS
                | {"response_format", "tool_choice", "tools"}
            )
            for key, value in merged.items():
                if value is None or key in excluded_keys:
                    continue
                chat_kwargs.setdefault(key, value)

        return chat_kwargs

    def _build_sampling_params_instance(
        self,
        sampling_params: dict[str, Any],
        *,
        sampling_params_cls: type[Any],
    ) -> Any:
        filtered_sampling_params = self._filter_sampling_params_for_vllm(
            sampling_params,
            sampling_params_cls=sampling_params_cls,
        )

        while True:
            try:
                return sampling_params_cls(**filtered_sampling_params)
            except TypeError as exc:
                match = self.UNSUPPORTED_KWARG_PATTERN.search(str(exc))
                if match is None:
                    raise
                unsupported_key = match.group(1)
                if unsupported_key not in filtered_sampling_params:
                    raise
                filtered_sampling_params = {
                    key: value
                    for key, value in filtered_sampling_params.items()
                    if key != unsupported_key
                }

    def _filter_sampling_params_for_vllm(
        self,
        sampling_params: dict[str, Any],
        *,
        sampling_params_cls: type[Any],
    ) -> dict[str, Any]:
        try:
            signature = inspect.signature(sampling_params_cls.__init__)
        except (TypeError, ValueError):
            return dict(sampling_params)

        parameters = signature.parameters
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_var_kwargs:
            return dict(sampling_params)

        supported_keys = {
            name
            for name, parameter in parameters.items()
            if name != "self"
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        return {
            key: value
            for key, value in sampling_params.items()
            if key in supported_keys
        }

    def _filter_chat_kwargs_for_vllm(self, chat_kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.engine is None:
            return dict(chat_kwargs)
        try:
            signature = inspect.signature(self.engine.chat)
        except (TypeError, ValueError):
            return dict(chat_kwargs)

        parameters = signature.parameters
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_var_kwargs:
            return dict(chat_kwargs)

        supported_keys = {
            name
            for name, parameter in parameters.items()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        supported_keys.discard("self")
        supported_keys.discard("messages")
        return {
            key: value
            for key, value in chat_kwargs.items()
            if key in supported_keys
        }

    def _is_async_engine(self) -> bool:
        return self.engine is not None and self.engine_kind == "async"

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _get_async_engine_tokenizer(self) -> Any:
        if self.engine is None:
            raise RuntimeError("vLLM engine is not initialized")

        for candidate in (
            self.engine,
            getattr(self.engine, "engine", None),
            getattr(self.engine, "engine_client", None),
            getattr(self.engine, "llm_engine", None),
        ):
            if candidate is None:
                continue
            get_tokenizer = getattr(candidate, "get_tokenizer", None)
            if callable(get_tokenizer):
                return await self._maybe_await(get_tokenizer())
            tokenizer = getattr(candidate, "tokenizer", None)
            if tokenizer is not None:
                return tokenizer

        raise BackendRequestValidationError(
            "This vLLM async engine does not expose a tokenizer for chat templating.",
            code="unsupported_parameter",
        )

    def _load_async_engine_image_asset(self, image_url: str) -> Any:
        """Load a vision asset into a vLLM-friendly in-memory image object."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise BackendRequestValidationError(
                "Async vLLM multimodal chat requires Pillow for image decoding.",
                code="unsupported_parameter",
            ) from exc

        normalized_url = self._normalize_data_url(image_url)
        parsed = urlparse(normalized_url)

        try:
            if normalized_url.startswith("data:") and ";base64," in normalized_url:
                _, payload = normalized_url.split(",", 1)
                image_bytes = base64.b64decode(payload)
                return Image.open(BytesIO(image_bytes)).convert("RGB")

            if parsed.scheme in {"http", "https"}:
                with urlopen(normalized_url, timeout=10) as response:
                    image_bytes = response.read()
                return Image.open(BytesIO(image_bytes)).convert("RGB")

            candidate_path = Path(normalized_url)
            if candidate_path.exists():
                return Image.open(candidate_path).convert("RGB")
        except Exception as exc:
            raise BackendRequestValidationError(
                f"Async vLLM multimodal chat could not load image input: {exc}",
                code="unsupported_parameter",
            ) from exc

        raise BackendRequestValidationError(
            "Async vLLM multimodal chat requires image_url values that are data URLs, "
            "HTTP(S) URLs, or readable local file paths.",
            code="unsupported_parameter",
        )

    def _build_async_engine_multi_modal_data(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        image_assets: list[Any] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "image_url":
                    continue
                image_url = ((block.get("image_url") or {}).get("url"))
                if not isinstance(image_url, str):
                    raise BackendRequestValidationError(
                        "Async vLLM multimodal chat requires image_url.url to be a string.",
                        code="unsupported_parameter",
                    )
                image_assets.append(self._load_async_engine_image_asset(image_url))

        if not image_assets:
            return None
        return {"image": image_assets}

    def _build_async_engine_generate_input(
        self,
        generate: Any,
        prompt: str,
        multi_modal_data: dict[str, Any] | None,
    ) -> tuple[Any, dict[str, Any]]:
        if multi_modal_data is None:
            return prompt, {}

        try:
            signature = inspect.signature(generate)
        except (TypeError, ValueError):
            signature = None

        accepts_multi_modal_kwarg = False
        if signature is not None:
            accepts_multi_modal_kwarg = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                or name == "multi_modal_data"
                for name, parameter in signature.parameters.items()
            )

        if accepts_multi_modal_kwarg:
            return prompt, {"multi_modal_data": multi_modal_data}

        return {"prompt": prompt, "multi_modal_data": multi_modal_data}, {}

    async def _build_async_engine_chat_prompt(
        self,
        messages: list[dict[str, Any]],
        chat_kwargs: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        tokenizer = await self._get_async_engine_tokenizer()
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise BackendRequestValidationError(
                "This vLLM async engine tokenizer does not support chat templates.",
                code="unsupported_parameter",
            )

        template_kwargs = dict(chat_kwargs.get("chat_template_kwargs") or {})
        for key in ("enable_thinking", "reasoning", "thinking"):
            if key in chat_kwargs and key not in template_kwargs:
                template_kwargs[key] = chat_kwargs[key]
        prompt = apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        if not isinstance(prompt, str):
            raise BackendRequestValidationError(
                "This vLLM async engine produced a non-text chat template prompt.",
                code="unsupported_parameter",
            )
        multi_modal_data = None
        if any(isinstance(message.get("content"), list) for message in messages):
            multi_modal_data = self._build_async_engine_multi_modal_data(messages)
        return prompt, multi_modal_data

    def _build_async_engine_sampling_params_instance(self, sampling_params: dict[str, Any]) -> Any:
        try:
            from vllm import SamplingParams
        except ImportError:
            return dict(sampling_params)
        return self._build_sampling_params_instance(
            sampling_params,
            sampling_params_cls=SamplingParams,
        )

    async def _invoke_async_engine_chat_stream(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        *,
        chat_kwargs: dict[str, Any],
        request_id: str,
    ) -> Any:
        if self.engine is None:
            raise RuntimeError("vLLM engine is not initialized")
        generate = getattr(self.engine, "generate", None)
        if not callable(generate):
            raise BackendRequestValidationError(
                "This vLLM async engine does not expose generate().",
                code="unsupported_parameter",
            )

        prompt, multi_modal_data = await self._build_async_engine_chat_prompt(messages, chat_kwargs)
        sampling_params_instance = self._build_async_engine_sampling_params_instance(sampling_params)
        generate_input, generate_kwargs = self._build_async_engine_generate_input(
            generate,
            prompt,
            multi_modal_data,
        )
        stream = generate(
            generate_input,
            sampling_params_instance,
            request_id,
            **generate_kwargs,
        )
        return await self._maybe_await(stream)

    async def _collect_async_engine_chat_completion(
        self,
        request: ChatCompletionsRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        sampling_params: dict[str, Any],
        chat_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        messages = self._build_chat_messages(request, runtime_context=runtime_context)
        request_id = f"chatcmpl-{uuid4().hex}"
        stream = await self._invoke_async_engine_chat_stream(
            messages,
            sampling_params,
            chat_kwargs=chat_kwargs,
            request_id=request_id,
        )
        final_item = None
        async for item in self._iter_chat_stream_outputs(stream):
            final_item = item
        if final_item is None:
            raise RuntimeError("vLLM async chat returned no result")
        return self._convert_chat_result(
            request=request,
            runtime_spec=runtime_spec,
            runtime_context=runtime_context,
            result=[final_item],
            sampling_params=sampling_params,
            chat_kwargs=chat_kwargs,
        )

    async def _collect_sync_chat_completion(
        self,
        request: ChatCompletionsRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        sampling_params: dict[str, Any],
        chat_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if self.engine is None:
            return self._build_chat_stub_response(
                request,
                runtime_spec,
                runtime_context,
                sampling_params,
                chat_kwargs,
            )
        if self._is_async_engine():
            raise BackendRequestValidationError(
                "This vLLM async runtime does not have a sync chat fallback engine.",
                code="unsupported_parameter",
            )

        messages = self._build_chat_messages(request, runtime_context=runtime_context)
        result = self._invoke_vllm_chat(
            messages,
            sampling_params,
            chat_kwargs=chat_kwargs,
        )
        return self._convert_chat_result(
            request=request,
            runtime_spec=runtime_spec,
            runtime_context=runtime_context,
            result=result,
            sampling_params=sampling_params,
            chat_kwargs=chat_kwargs,
        )

    def _invoke_chat_callable(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling_params_instance: Any | None,
        sampling_params: dict[str, Any],
        chat_kwargs: dict[str, Any],
    ) -> Any:
        if self.engine is None:
            raise RuntimeError("vLLM engine is not initialized")
        current_chat_kwargs = self._filter_chat_kwargs_for_vllm(chat_kwargs)

        while True:
            try:
                if sampling_params_instance is not None:
                    return self.engine.chat(
                        messages,
                        sampling_params=sampling_params_instance,
                        **current_chat_kwargs,
                    )
                return self.engine.chat(messages, **sampling_params, **current_chat_kwargs)
            except TypeError as exc:
                match = self.UNSUPPORTED_KWARG_PATTERN.search(str(exc))
                if match is None:
                    raise
                unsupported_key = match.group(1)
                if unsupported_key not in current_chat_kwargs:
                    raise
                current_chat_kwargs = {
                    key: value
                    for key, value in current_chat_kwargs.items()
                    if key != unsupported_key
                }

    def _invoke_vllm_chat(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        *,
        chat_kwargs: dict[str, Any],
    ) -> Any:
        if self.engine is None:
            raise RuntimeError("vLLM engine is not initialized")

        try:
            from vllm import SamplingParams
        except ImportError:
            SamplingParams = None  # type: ignore[assignment]

        if SamplingParams is not None:
            sampling_params_instance = self._build_sampling_params_instance(
                sampling_params,
                sampling_params_cls=SamplingParams,
            )
            try:
                return self._invoke_chat_callable(
                    messages,
                    sampling_params_instance=sampling_params_instance,
                    sampling_params=sampling_params,
                    chat_kwargs=chat_kwargs,
                )
            except TypeError:
                pass

        return self._invoke_chat_callable(
            messages,
            sampling_params_instance=None,
            sampling_params=sampling_params,
            chat_kwargs=chat_kwargs,
        )

    def _chat_method_accepts_stream(self) -> bool:
        if self.engine is None:
            return False
        chat_method = getattr(self.engine, "chat", None)
        if not callable(chat_method):
            return False
        try:
            signature = inspect.signature(chat_method)
        except (TypeError, ValueError):
            return False
        parameters = signature.parameters
        return "stream" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    def _invoke_vllm_chat_stream(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        *,
        chat_kwargs: dict[str, Any],
    ) -> Any:
        if self.engine is None:
            raise RuntimeError("vLLM engine is not initialized")
        if not self._chat_method_accepts_stream():
            raise BackendRequestValidationError(
                "This vLLM runtime does not expose a native streaming chat API.",
                code="unsupported_parameter",
            )

        try:
            from vllm import SamplingParams
        except ImportError:
            SamplingParams = None  # type: ignore[assignment]

        sampling_params_instance = (
            self._build_sampling_params_instance(
                sampling_params,
                sampling_params_cls=SamplingParams,
            )
            if SamplingParams is not None
            else None
        )
        current_chat_kwargs = self._filter_chat_kwargs_for_vllm(chat_kwargs)

        while True:
            try:
                if sampling_params_instance is not None:
                    return self.engine.chat(
                        messages,
                        sampling_params=sampling_params_instance,
                        stream=True,
                        **current_chat_kwargs,
                    )
                return self.engine.chat(
                    messages,
                    stream=True,
                    **sampling_params,
                    **current_chat_kwargs,
                )
            except TypeError as exc:
                match = self.UNSUPPORTED_KWARG_PATTERN.search(str(exc))
                if match is None:
                    raise
                unsupported_key = match.group(1)
                if unsupported_key == "stream":
                    raise BackendRequestValidationError(
                        "This vLLM runtime does not expose a native streaming chat API.",
                        code="unsupported_parameter",
                    ) from exc
                if unsupported_key not in current_chat_kwargs:
                    raise
                current_chat_kwargs = {
                    key: value
                    for key, value in current_chat_kwargs.items()
                    if key != unsupported_key
                }

    async def _iter_chat_stream_outputs(self, result: Any) -> AsyncIterator[Any]:
        if inspect.isawaitable(result):
            result = await result
        if hasattr(result, "__aiter__"):
            async for item in result:
                yield item
            return
        if isinstance(result, Iterable) and not isinstance(result, (bytes, str, dict, list, tuple)):
            for item in result:
                yield item
            return
        raise BackendRequestValidationError(
            "This vLLM runtime did not return an incremental chat stream.",
            code="unsupported_parameter",
        )

    def _extract_stream_output_text_and_finish(self, item: Any) -> tuple[str, str | None]:
        if isinstance(item, dict):
            if isinstance(item.get("delta_text"), str):
                return item["delta_text"], item.get("finish_reason")
            if isinstance(item.get("text"), str):
                return item["text"], item.get("finish_reason")
            outputs = item.get("outputs") or []
        else:
            if isinstance(getattr(item, "delta_text", None), str):
                return getattr(item, "delta_text"), getattr(item, "finish_reason", None)
            if isinstance(getattr(item, "text", None), str):
                return getattr(item, "text"), getattr(item, "finish_reason", None)
            outputs = getattr(item, "outputs", None) or []

        if not outputs:
            return "", None
        output = outputs[0]
        if isinstance(output, dict):
            return str(output.get("text") or ""), output.get("finish_reason")
        return str(getattr(output, "text", "") or ""), getattr(output, "finish_reason", None)

    async def _iter_chat_deltas(
        self,
        request: ChatCompletionsRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        sampling_params: dict[str, Any],
        chat_kwargs: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        response_id = f"chatcmpl-{uuid4().hex}"
        created = int(time())
        model = runtime_context.get("served_model_name") or runtime_spec.get("served_model_name") or request.model

        if self.engine is None:
            response = self._build_chat_stub_response(
                request,
                runtime_spec,
                runtime_context,
                sampling_params,
                chat_kwargs,
            )
            yield {
                "type": "chat_delta",
                "id": response["id"],
                "created": response["created"],
                "model": response["model"],
                "delta_text": response["content"],
                "finish_reason": None,
            }
            yield {
                "type": "chat_delta",
                "id": response["id"],
                "created": response["created"],
                "model": response["model"],
                "delta_text": "",
                "finish_reason": response.get("finish_reason", "stop"),
            }
            return

        messages = self._build_chat_messages(request, runtime_context=runtime_context)
        if self._is_async_engine():
            stream = await self._invoke_async_engine_chat_stream(
                messages,
                sampling_params,
                chat_kwargs=chat_kwargs,
                request_id=response_id,
            )
        else:
            stream = self._invoke_vllm_chat_stream(
                messages,
                sampling_params,
                chat_kwargs=chat_kwargs,
            )
        previous_text = ""
        async for item in self._iter_chat_stream_outputs(stream):
            text, finish_reason = self._extract_stream_output_text_and_finish(item)
            if text.startswith(previous_text):
                delta_text = text[len(previous_text):]
            else:
                delta_text = text
            previous_text = text
            if delta_text:
                yield {
                    "type": "chat_delta",
                    "id": response_id,
                    "created": created,
                    "model": model,
                    "delta_text": delta_text,
                    "finish_reason": None,
                }
            if finish_reason:
                yield {
                    "type": "chat_delta",
                    "id": response_id,
                    "created": created,
                    "model": model,
                    "delta_text": "",
                    "finish_reason": finish_reason,
                }
                return

    async def _iter_chat_completion_deltas(
        self,
        request: ChatCompletionsRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        sampling_params: dict[str, Any],
        chat_kwargs: dict[str, Any],
        *,
        force_sync: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Expose a full non-streaming chat completion as an SSE-compatible stream.

        This is a compatibility path for vLLM runtimes backed by sync LLM, which
        cannot produce true incremental tokens but can still satisfy OpenAI
        stream clients with one content chunk followed by a terminal chunk.
        """
        if self.engine is None:
            response = self._build_chat_stub_response(
                request,
                runtime_spec,
                runtime_context,
                sampling_params,
                chat_kwargs,
            )
        elif self._is_async_engine() and not force_sync:
            response = await self._collect_async_engine_chat_completion(
                request,
                runtime_spec,
                runtime_context,
                sampling_params,
                chat_kwargs,
            )
        else:
            response = await self._collect_sync_chat_completion(
                request,
                runtime_spec,
                runtime_context,
                sampling_params,
                chat_kwargs,
            )

        response_id = response.get("id") or f"chatcmpl-{uuid4().hex}"
        created = response.get("created") or int(time())
        model = response.get("model") or (
            runtime_context.get("served_model_name") or runtime_spec.get("served_model_name") or request.model
        )
        content = response.get("content") or ""
        finish_reason = response.get("finish_reason") or "stop"
        if content:
            yield {
                "type": "chat_delta",
                "id": response_id,
                "created": created,
                "model": model,
                "delta_text": content,
                "finish_reason": None,
            }
        yield {
            "type": "chat_delta",
            "id": response_id,
            "created": created,
            "model": model,
            "delta_text": "",
            "finish_reason": finish_reason,
        }

    def _supports_multimodal(self, runtime_context: dict[str, Any] | None = None) -> bool:
        capabilities = self.runtime_spec.get("capabilities")
        if capabilities is None and runtime_context is not None:
            capabilities = runtime_context.get("capabilities", [])
        return "vision" in (capabilities or [])

    def _normalize_data_url(self, url: str) -> str:
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
        payload = block.model_dump(mode="json", exclude_none=True)
        if payload.get("type") == "image_url":
            image_url = payload.get("image_url") or {}
            url = image_url.get("url")
            if isinstance(url, str):
                image_url["url"] = self._normalize_data_url(url)
        return payload

    def _is_text_only_content(self, content: Any) -> bool:
        if not isinstance(content, list) or not content:
            return False
        for block in content:
            payload = self._serialize_content_block(block)
            if payload.get("type") != "text":
                return False
        return True

    def _collapse_text_only_content(self, content: list[Any]) -> str:
        parts: list[str] = []
        for block in content:
            payload = self._serialize_content_block(block)
            text = payload.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    def _serialize_message_content(
        self,
        content: Any,
        *,
        allow_multimodal: bool,
    ) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content

        if self._is_text_only_content(content):
            return self._collapse_text_only_content(content)

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
            if message.tool_calls is not None:
                payload["tool_calls"] = message.tool_calls
            if message.function_call is not None:
                payload["function_call"] = message.function_call
            messages.append(payload)
        return messages

    def _should_use_openai_serving_adapter(
        self,
        runtime_spec: dict[str, Any] | None = None,
    ) -> bool:
        spec = runtime_spec or self.runtime_spec
        openai_serving = spec.get("openai_serving") or {}
        return spec.get("task_mode") == "generate" and bool(openai_serving.get("enabled"))

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

        openai_serving_config = self.runtime_spec.get("openai_serving") or {}
        engine_kwargs = self.runtime_spec.get("engine_kwargs") or {}
        tool_call_parser = (
            openai_serving_config.get("tool_call_parser")
            or engine_kwargs.get("tool_call_parser")
        )
        enable_auto_tools = bool(
            openai_serving_config.get("enable_auto_tool_choice")
            or engine_kwargs.get("enable_auto_tool_choice")
        )
        reasoning_parser = (
            openai_serving_config.get("reasoning_parser")
            or engine_kwargs.get("reasoning_parser")
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
        tool_call_parser = (
            openai_serving_config.get("tool_call_parser")
            or engine_kwargs.get("tool_call_parser")
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
            "reasoning_parser": openai_serving_config.get("reasoning_parser") or "",
            "enable_auto_tools": bool(
                openai_serving_config.get("enable_auto_tool_choice")
                or engine_kwargs.get("enable_auto_tool_choice")
            ),
            "tool_parser": tool_call_parser,
            "default_chat_template_kwargs": self._request_defaults().get("chat_template_kwargs"),
        }
        for key, value in optional_kwargs.items():
            if key in parameters:
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
            logger.warning("Failed to initialize vLLM OpenAI serving adapter: %s", exc)
            return None

        self.openai_serving_adapter_init_error = None
        return DynamicVLLMOpenAIChatServingAdapter(
            chat_request_cls=imports.chat_request_cls,
            serving_chat=serving_chat,
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

        serialized_messages: list[dict[str, Any]] = []
        allow_multimodal = self._supports_multimodal(runtime_context)
        for message in request.messages:
            serialized = message.model_dump(mode="json", exclude_none=True)
            if message.content is None:
                serialized["content"] = None
            else:
                serialized["content"] = self._serialize_message_content(
                    message.content,
                    allow_multimodal=allow_multimodal,
                )
            serialized_messages.append(serialized)

        payload["messages"] = serialized_messages
        payload["model"] = (
            runtime_context.get("served_model_name")
            or runtime_spec.get("served_model_name")
            or request.model
        )
        payload.update(request.extra_body or {})
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

    def _build_chat_stub_stream_chunks(
        self,
        request: ChatCompletionsRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        sampling_params: dict[str, Any],
        chat_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        response = self._build_chat_stub_response(
            request,
            runtime_spec,
            runtime_context,
            sampling_params,
            chat_kwargs,
        )
        created = response["created"]
        response_id = response["id"]
        model = response["model"]
        return [
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": response["content"],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": response["finish_reason"],
                    }
                ],
            },
        ]

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
        if self.openai_serving_chat_adapter is not None:
            request_payload = self._build_openai_serving_request_payload(
                request,
                runtime_spec=runtime_spec,
                runtime_context=runtime_context,
            )
            try:
                return await self._call_openai_serving_chat_completion(request_payload)
            except Exception as exc:
                self.openai_serving_adapter_init_error = str(exc)
                self.openai_serving_chat_adapter = None

        messages = self._build_chat_messages(request, runtime_context=runtime_context)
        sampling_params = self._build_sampling_params(request, runtime_spec=runtime_spec)
        chat_kwargs = self._build_chat_kwargs(request, runtime_spec=runtime_spec)
        if self.engine is None:
            return self._build_chat_stub_response(
                request,
                runtime_spec,
                runtime_context,
                sampling_params,
                chat_kwargs,
            )
        if self._is_async_engine():
            return await self._collect_async_engine_chat_completion(
                request,
                runtime_spec,
                runtime_context,
                sampling_params,
                chat_kwargs,
            )

        return await self._collect_sync_chat_completion(
            request,
            runtime_spec,
            runtime_context,
            sampling_params,
            chat_kwargs,
        )

    async def chat_completion_stream(
        self,
        runtime_spec: dict[str, Any],
        request: ChatCompletionsRequest,
        runtime_context: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
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
                self.openai_serving_adapter_init_error = str(exc)
                self.openai_serving_chat_adapter = None

        sampling_params = self._build_sampling_params(request, runtime_spec=runtime_spec)
        chat_kwargs = self._build_chat_kwargs(request, runtime_spec=runtime_spec)
        if self.engine is None:
            async for event in self._iter_chat_completion_deltas(
                request,
                runtime_spec,
                runtime_context,
                sampling_params,
                chat_kwargs,
                force_sync=True,
            ):
                yield event
            return
        try:
            async for event in self._iter_chat_deltas(
                request,
                runtime_spec,
                runtime_context,
                sampling_params,
                chat_kwargs,
            ):
                yield event
            return
        except BackendRequestValidationError as exc:
            if exc.code != "unsupported_parameter" or self._is_async_engine():
                raise

        async for event in self._iter_chat_completion_deltas(
            request,
            runtime_spec,
            runtime_context,
            sampling_params,
            chat_kwargs,
            force_sync=True,
        ):
            yield event

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
