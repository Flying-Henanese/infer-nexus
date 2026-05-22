"""vLLM backend adapter implementation."""

from __future__ import annotations

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
        loading_config = self.runtime_spec.get("model_loading_config") or {}
        if loading_config.get("revision"):
            llm_kwargs["revision"] = loading_config["revision"]
        llm_kwargs.update(self.runtime_spec.get("engine_kwargs") or {})
        gpu_memory_utilization = self.runtime_spec.get("gpu_memory_utilization")
        max_model_len = self.runtime_spec.get("max_model_len")
        requested_mode = self.runtime_spec.get("task_mode")

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
        except (TypeError, ValueError):
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

    def build_runtime_spec(self, model: ModelConfig, resolved_model_reference: str) -> dict[str, Any]:
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
            "engine_kwargs": dict(model.vllm.engine_kwargs),
            "request_defaults": dict(model.vllm.request_defaults),
            "request_policy": model.vllm.request_policy.model_dump(mode="json"),
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

    def _serialize_message_content(
        self,
        content: Any,
        *,
        allow_multimodal: bool,
    ) -> str | list[dict[str, Any]]:
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
