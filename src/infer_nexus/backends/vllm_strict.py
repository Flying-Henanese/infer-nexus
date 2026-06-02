"""Strict/native vLLM OpenAI serving execution paths."""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from infer_nexus.backends.vllm_native import (
    DynamicVLLMOpenAIChatServingAdapter,
    DynamicVLLMOpenAIEmbeddingServingAdapter,
    OpenAIChatServingAdapter,
    OpenAIEmbeddingServingAdapter,
    OpenAIServingEngineClientCompatProxy,
    ResolvedOpenAIServingImports,
)
from infer_nexus.core.errors import BackendConfigurationError
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest

if TYPE_CHECKING:
    from infer_nexus.backends.vllm import VLLMBackend


logger = logging.getLogger(__name__)


class StrictNativeVLLMExecutor:
    """Encapsulate replica-local native vLLM OpenAI serving behavior."""

    def __init__(self, backend: VLLMBackend) -> None:
        self.backend = backend

    def _should_use_openai_serving_adapter(
        self,
        runtime_spec: dict[str, Any] | None = None,
    ) -> bool:
        spec = runtime_spec or self.backend.runtime_spec
        openai_serving = spec.get("openai_serving") or {}
        return spec.get("task_mode") == "generate" and bool(openai_serving.get("enabled"))

    def _should_use_openai_serving_embedding_adapter(
        self,
        runtime_spec: dict[str, Any] | None = None,
    ) -> bool:
        spec = runtime_spec or self.backend.runtime_spec
        openai_serving = spec.get("openai_serving") or {}
        return spec.get("task_mode") == "embed" and bool(openai_serving.get("enabled"))

    def _raise_openai_serving_unavailable(self, reason: str | None = None) -> None:
        detail = reason or self.backend.openai_serving_adapter_init_error or "adapter is not initialized"
        raise BackendConfigurationError(
            "vllm_native vLLM chat requires the replica-local vLLM OpenAI serving adapter, "
            f"but it is unavailable: {detail}"
        )

    def _raise_openai_embedding_serving_unavailable(self, reason: str | None = None) -> None:
        detail = (
            reason
            or self.backend.openai_serving_embedding_adapter_init_error
            or "adapter is not initialized"
        )
        raise BackendConfigurationError(
            "vllm_native vLLM embedding requires the replica-local vLLM OpenAI embeddings "
            f"serving adapter, but it is unavailable: {detail}"
        )

    def _import_vllm_symbol(self, module_path: str, symbol_name: str) -> Any:
        from infer_nexus.backends import vllm as vllm_module

        module = vllm_module.importlib.import_module(module_path)
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

    def _resolve_openai_serving_engine_client(self) -> Any | None:
        def _is_compatible(client: Any) -> bool:
            return hasattr(client, "model_config")

        candidates = [
            getattr(self.backend.engine, "engine_client", None),
            getattr(self.backend.engine, "async_engine_client", None),
            getattr(self.backend.engine, "llm_engine", None),
            getattr(self.backend.engine, "engine", None),
            getattr(self.backend.engine, "_engine", None),
            self.backend.engine,
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
            self.backend.runtime_spec.get("served_model_name")
            or self.backend.runtime_spec.get("model_name")
            or self.backend.runtime_spec.get("model_path")
        )
        model_path = self.backend.runtime_spec["model_path"]
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

        engine_kwargs = self.backend.runtime_spec.get("engine_kwargs") or {}
        openai_serving_config = self.backend.runtime_spec.get("openai_serving") or {}
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
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
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
            "default_chat_template_kwargs": self.backend._request_defaults().get("chat_template_kwargs"),
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
        openai_serving_config = self.backend.runtime_spec.get("openai_serving") or {}
        engine_kwargs = self.backend.runtime_spec.get("engine_kwargs") or {}
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
            "default_chat_template_kwargs": self.backend._request_defaults().get("chat_template_kwargs"),
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

    def _initialize_openai_serving_chat_adapter(self) -> OpenAIChatServingAdapter | None:
        if not self._should_use_openai_serving_adapter():
            self.backend.openai_serving_adapter_init_error = None
            return None
        if self.backend.engine is None:
            self.backend.openai_serving_adapter_init_error = (
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
            self.backend.openai_serving_adapter_init_error = str(exc)
            logger.warning(
                "Failed to initialize vLLM OpenAI serving adapter for model_path='%s' "
                "served_model_name='%s' engine_kind='%s': %s",
                self.backend.runtime_spec.get("model_path"),
                self.backend.runtime_spec.get("served_model_name"),
                self.backend.engine_kind,
                exc,
            )
            return None

        self.backend.openai_serving_adapter_init_error = None
        return DynamicVLLMOpenAIChatServingAdapter(
            chat_request_cls=imports.chat_request_cls,
            serving_chat=serving_chat,
        )

    def _initialize_openai_serving_embedding_adapter(self) -> OpenAIEmbeddingServingAdapter | None:
        if not self._should_use_openai_serving_embedding_adapter():
            self.backend.openai_serving_embedding_adapter_init_error = None
            return None
        if self.backend.engine is None:
            self.backend.openai_serving_embedding_adapter_init_error = (
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
            self.backend.openai_serving_embedding_adapter_init_error = str(exc)
            logger.warning(
                "Failed to initialize vLLM OpenAI embeddings serving adapter for "
                "model_path='%s' served_model_name='%s' engine_kind='%s': %s",
                self.backend.runtime_spec.get("model_path"),
                self.backend.runtime_spec.get("served_model_name"),
                self.backend.engine_kind,
                exc,
            )
            return None

        self.backend.openai_serving_embedding_adapter_init_error = None
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
        adapter = self.backend.openai_serving_chat_adapter
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
        adapter = self.backend.openai_serving_chat_adapter
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
        adapter = self.backend.openai_serving_embedding_adapter
        if adapter is None:
            raise RuntimeError("vLLM OpenAI embeddings serving adapter is not initialized")

        result = adapter.embedding(request_payload)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def chat_completion(
        self,
        runtime_spec: dict[str, Any],
        request: ChatCompletionsRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        vllm_native = self.backend._is_vllm_native(runtime_spec)
        if self.backend.openai_serving_chat_adapter is not None:
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
                self.backend.openai_serving_adapter_init_error = str(exc)
                self.backend.openai_serving_chat_adapter = None
        elif vllm_native:
            self._raise_openai_serving_unavailable()
        return await self.backend.local_best_effort_executor.chat_completion(
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
        vllm_native = self.backend._is_vllm_native(runtime_spec)
        if self.backend.openai_serving_chat_adapter is not None:
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
                self.backend.openai_serving_adapter_init_error = str(exc)
                self.backend.openai_serving_chat_adapter = None
        elif vllm_native:
            self._raise_openai_serving_unavailable()
        async for event in self.backend.local_best_effort_executor.chat_completion_stream(
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
        if self.backend.openai_serving_embedding_adapter is None:
            self._raise_openai_embedding_serving_unavailable()
        request_payload = self._build_openai_serving_embedding_request_payload(
            request,
            runtime_spec=runtime_spec,
            runtime_context=runtime_context,
        )
        return await self._call_openai_serving_embedding(request_payload)
