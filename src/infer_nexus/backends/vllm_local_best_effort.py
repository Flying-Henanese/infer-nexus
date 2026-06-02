"""Local best-effort chat execution path for the vLLM backend."""

from __future__ import annotations

import base64
import inspect
import re
from io import BytesIO
from pathlib import Path
from collections.abc import AsyncIterator, Iterable
from time import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from urllib.request import urlopen
from uuid import uuid4

from infer_nexus.core.errors import BackendRequestValidationError
from infer_nexus.core.schemas import ChatCompletionsRequest

if TYPE_CHECKING:
    from infer_nexus.backends.vllm import VLLMBackend


class LocalBestEffortVLLMExecutor:
    """Encapsulate fallback local chat behavior for non-native vLLM modes."""

    def __init__(self, backend: VLLMBackend) -> None:
        self.backend = backend

    def _merge_request_extras(
        self,
        request: ChatCompletionsRequest,
        *,
        runtime_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged = dict(self.backend._request_defaults(runtime_spec))
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

        for key in self.backend.REQUEST_DEFAULT_SAMPLING_KEYS:
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
        policy = self.backend._request_policy(runtime_spec)
        allow_tools = bool(policy.get("allow_tools"))
        allow_reasoning = bool(policy.get("allow_reasoning"))
        passthrough_unknown = bool(policy.get("passthrough_unknown_openai_fields"))

        if not allow_tools and (request.tools is not None or request.tool_choice is not None):
            raise BackendRequestValidationError(
                "This model does not allow tool-calling request fields.",
                code="unsupported_parameter",
            )

        reasoning_keys_present = [
            key for key in self.backend.REASONING_REQUEST_KEYS if merged.get(key) is not None
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
            for key in self.backend.TOOL_REQUEST_KEYS:
                value = merged.get(key)
                if value is not None:
                    chat_kwargs[key] = value

        if allow_reasoning:
            for key in self.backend.REASONING_REQUEST_KEYS:
                value = merged.get(key)
                if value is not None:
                    chat_kwargs[key] = value

        if passthrough_unknown:
            excluded_keys = (
                self.backend.REQUEST_DEFAULT_SAMPLING_KEYS
                | self.backend.TOOL_REQUEST_KEYS
                | self.backend.REASONING_REQUEST_KEYS
                | {"response_format", "tool_choice", "tools"}
            )
            for key, value in merged.items():
                if value is None or key in excluded_keys:
                    continue
                chat_kwargs.setdefault(key, value)

        return chat_kwargs

    def _supports_multimodal(self, runtime_context: dict[str, Any] | None = None) -> bool:
        capabilities = self.backend.runtime_spec.get("capabilities")
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
                match = self.backend.UNSUPPORTED_KWARG_PATTERN.search(str(exc))
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
        if self.backend.engine is None:
            return dict(chat_kwargs)
        try:
            signature = inspect.signature(self.backend.engine.chat)
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
        return self.backend.engine is not None and self.backend.engine_kind == "async"

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _get_async_engine_tokenizer(self) -> Any:
        if self.backend.engine is None:
            raise RuntimeError("vLLM engine is not initialized")

        for candidate in (
            self.backend.engine,
            getattr(self.backend.engine, "engine", None),
            getattr(self.backend.engine, "engine_client", None),
            getattr(self.backend.engine, "llm_engine", None),
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
                image_assets.append(self.backend._load_async_engine_image_asset(image_url))

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
        if self.backend.engine is None:
            raise RuntimeError("vLLM engine is not initialized")
        generate = getattr(self.backend.engine, "generate", None)
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
        messages = self.backend._build_chat_messages(request, runtime_context=runtime_context)
        request_id = f"chatcmpl-{uuid4().hex}"
        stream = await self.backend._invoke_async_engine_chat_stream(
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
        return self.backend._convert_chat_result(
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
        if self.backend.engine is None:
            return self.backend._build_chat_stub_response(
                request,
                runtime_spec,
                runtime_context,
                sampling_params,
                chat_kwargs,
            )
        if self.backend._is_async_engine():
            raise BackendRequestValidationError(
                "This vLLM async runtime does not have a sync chat fallback engine.",
                code="unsupported_parameter",
            )

        messages = self.backend._build_chat_messages(request, runtime_context=runtime_context)
        result = self._invoke_vllm_chat(
            messages,
            sampling_params,
            chat_kwargs=chat_kwargs,
        )
        return self.backend._convert_chat_result(
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
        if self.backend.engine is None:
            raise RuntimeError("vLLM engine is not initialized")
        current_chat_kwargs = self._filter_chat_kwargs_for_vllm(chat_kwargs)

        while True:
            try:
                if sampling_params_instance is not None:
                    return self.backend.engine.chat(
                        messages,
                        sampling_params=sampling_params_instance,
                        **current_chat_kwargs,
                    )
                return self.backend.engine.chat(messages, **sampling_params, **current_chat_kwargs)
            except TypeError as exc:
                match = self.backend.UNSUPPORTED_KWARG_PATTERN.search(str(exc))
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
        if self.backend.engine is None:
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
        if self.backend.engine is None:
            return False
        chat_method = getattr(self.backend.engine, "chat", None)
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
        if self.backend.engine is None:
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
                    return self.backend.engine.chat(
                        messages,
                        sampling_params=sampling_params_instance,
                        stream=True,
                        **current_chat_kwargs,
                    )
                return self.backend.engine.chat(
                    messages,
                    stream=True,
                    **sampling_params,
                    **current_chat_kwargs,
                )
            except TypeError as exc:
                match = self.backend.UNSUPPORTED_KWARG_PATTERN.search(str(exc))
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

        if self.backend.engine is None:
            response = self.backend._build_chat_stub_response(
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

        messages = self.backend._build_chat_messages(request, runtime_context=runtime_context)
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
        """Expose a full non-streaming chat completion as an SSE-compatible stream."""
        if self.backend.engine is None:
            response = self.backend._build_chat_stub_response(
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

    async def chat_completion(
        self,
        runtime_spec: dict[str, Any],
        request: ChatCompletionsRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        sampling_params = self.backend._build_sampling_params(request, runtime_spec=runtime_spec)
        chat_kwargs = self.backend._build_chat_kwargs(request, runtime_spec=runtime_spec)
        if self.backend.engine is None:
            return self.backend._build_chat_stub_response(
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
        sampling_params = self.backend._build_sampling_params(request, runtime_spec=runtime_spec)
        chat_kwargs = self.backend._build_chat_kwargs(request, runtime_spec=runtime_spec)
        if self.backend.engine is None:
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
