"""vLLM 后端适配器实现。

本模块负责 vLLM 后端的生命周期、runtime spec 校验和请求分发衔接。
兼容模式相关的执行细节会委托给严格/原生执行器和本地 best-effort
执行器，同时保留 infer-nexus 其他部分依赖的公共后端接口。
"""

from __future__ import annotations

import importlib
import inspect
import logging
import re
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
from infer_nexus.backends.vllm_strict import StrictNativeVLLMExecutor
from infer_nexus.catalog.models import ModelConfig
from infer_nexus.core.enums import CompatibilityMode
from infer_nexus.core.errors import BackendConfigurationError
from infer_nexus.core.schemas import ChatCompletionsRequest, EmbeddingRequest, RerankRequest


logger = logging.getLogger(__name__)


class VLLMBackend(InferenceBackend):
    """适配本地 vLLM 引擎生命周期和请求处理。"""

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
        """根据已解析的 runtime spec 创建 vLLM 后端。"""
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
        self.strict_executor = StrictNativeVLLMExecutor(self)

    def validate_runtime_spec(self, runtime_spec: dict[str, Any], runtime_context: dict[str, Any]) -> None:
        """校验 runtime spec 是否能够服务当前请求任务。"""
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
        """移除 callable 签名不支持的关键字参数。"""
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
        """在受支持的 vLLM 版本中创建异步 vLLM 引擎。"""
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
        """初始化配置指定的 vLLM 引擎和可选 serving adapter。"""
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
            if "task" in llm_init_args or accepts_var_kwargs:
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
        """释放该后端持有的引擎和 adapter 引用。"""
        self.engine = None
        self.openai_serving_chat_adapter = None
        self.openai_serving_embedding_adapter = None
        self.openai_serving_adapter_init_error = None
        self.openai_serving_embedding_adapter_init_error = None
        self.async_engine_init_error = None
        self.engine_kind = "stopped"
        self.engine_state = "stopped"

    def build_runtime_spec(self, model: ModelConfig, resolved_model_reference: str) -> dict[str, Any]:
        """为 catalog 模型构建 vLLM runtime spec。"""
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
        """返回 runtime spec 中的请求默认值。"""
        spec = runtime_spec or self.runtime_spec
        return dict(spec.get("request_defaults") or {})

    def _request_policy(self, runtime_spec: dict[str, Any] | None = None) -> dict[str, Any]:
        """返回 runtime spec 中的请求策略配置。"""
        spec = runtime_spec or self.runtime_spec
        return dict(spec.get("request_policy") or {})

    def _compat_mode(self, runtime_spec: dict[str, Any] | None = None) -> str:
        """返回已配置的兼容模式。"""
        spec = runtime_spec or self.runtime_spec
        return str(spec.get("compat_mode") or CompatibilityMode.LOCAL_BEST_EFFORT.value)

    def _is_strict_openai(self, runtime_spec: dict[str, Any] | None = None) -> bool:
        """返回运行时是否使用严格 OpenAI 兼容模式。"""
        return self._compat_mode(runtime_spec) == CompatibilityMode.STRICT_OPENAI.value

    def _is_vllm_native(self, runtime_spec: dict[str, Any] | None = None) -> bool:
        """返回运行时是否需要原生 vLLM serving。"""
        return self._compat_mode(runtime_spec) in {
            CompatibilityMode.VLLM_NATIVE.value,
            CompatibilityMode.STRICT_OPENAI.value,
        }

    def _requires_openai_serving_adapter(
        self,
        runtime_spec: dict[str, Any] | None = None,
    ) -> bool:
        """返回聊天任务是否需要 OpenAI serving adapter。"""
        spec = runtime_spec or self.runtime_spec
        return self._is_vllm_native(spec) and spec.get("task_mode") == "generate"

    def _requires_openai_embedding_serving_adapter(
        self,
        runtime_spec: dict[str, Any] | None = None,
    ) -> bool:
        """返回 embedding 任务是否需要 OpenAI serving adapter。"""
        spec = runtime_spec or self.runtime_spec
        return self._is_vllm_native(spec) and spec.get("task_mode") == "embed"

    def _raise_openai_serving_unavailable(self, reason: str | None = None) -> None:
        """抛出严格执行器的 chat serving 不可用错误。"""
        self.strict_executor._raise_openai_serving_unavailable(reason)

    def _raise_openai_embedding_serving_unavailable(self, reason: str | None = None) -> None:
        """抛出严格执行器的 embedding serving 不可用错误。"""
        self.strict_executor._raise_openai_embedding_serving_unavailable(reason)

    def _normalize_embedding_inputs(self, request: EmbeddingRequest) -> list[str]:
        """Deprecated: delegate local embedding input normalization to the local executor."""
        return self.local_best_effort_executor._normalize_embedding_inputs(request)

    def _encode_embedding_base64(self, embedding: list[float]) -> str:
        """Deprecated: delegate local embedding encoding to the local executor."""
        return self.local_best_effort_executor._encode_embedding_base64(embedding)

    def _build_embedding_stub_response(
        self,
        request: EmbeddingRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        inputs: list[str],
    ) -> dict[str, Any]:
        """Deprecated: delegate local embedding stub shaping to the local executor."""
        return self.local_best_effort_executor._build_embedding_stub_response(
            request,
            runtime_spec,
            runtime_context,
            inputs,
        )

    def _convert_embedding_result(
        self,
        *,
        request: EmbeddingRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        result: Any,
    ) -> dict[str, Any]:
        """Deprecated: delegate local embedding result shaping to the local executor."""
        return self.local_best_effort_executor._convert_embedding_result(
            request=request,
            runtime_spec=runtime_spec,
            runtime_context=runtime_context,
            result=result,
        )

    def _normalize_rerank_documents(self, request: RerankRequest) -> list[str]:
        """Deprecated: delegate local rerank input normalization to the local executor."""
        return self.local_best_effort_executor._normalize_rerank_documents(request)

    def _score_stub_document(self, query: str, document: str) -> float:
        """Deprecated: delegate local rerank stub scoring to the local executor."""
        return self.local_best_effort_executor._score_stub_document(query, document)

    def _build_rerank_stub_response(
        self,
        request: RerankRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        documents: list[str],
    ) -> dict[str, Any]:
        """Deprecated: delegate local rerank stub shaping to the local executor."""
        return self.local_best_effort_executor._build_rerank_stub_response(
            request,
            runtime_spec,
            runtime_context,
            documents,
        )

    def _convert_rerank_result(
        self,
        *,
        request: RerankRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        documents: list[str],
        result: Any,
    ) -> dict[str, Any]:
        """Deprecated: delegate local rerank result shaping to the local executor."""
        return self.local_best_effort_executor._convert_rerank_result(
            request=request,
            runtime_spec=runtime_spec,
            runtime_context=runtime_context,
            documents=documents,
            result=result,
        )

    def _merge_request_extras(
        self,
        request: ChatCompletionsRequest,
        *,
        runtime_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """通过本地 best-effort 执行器合并聊天请求扩展参数。"""
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
        """通过本地 best-effort 执行器构建采样参数。"""
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
        """通过本地 best-effort 执行器构建 chat kwargs。"""
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
        """通过本地执行器实例化 vLLM sampling params。"""
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
        """通过本地执行器过滤 vLLM 采样参数。"""
        return self.local_best_effort_executor._filter_sampling_params_for_vllm(
            sampling_params,
            sampling_params_cls=sampling_params_cls,
        )

    def _filter_chat_kwargs_for_vllm(self, chat_kwargs: dict[str, Any]) -> dict[str, Any]:
        """通过本地执行器过滤 vLLM chat kwargs。"""
        return self.local_best_effort_executor._filter_chat_kwargs_for_vllm(chat_kwargs)

    def _is_async_engine(self) -> bool:
        """返回当前引擎是否为异步引擎。"""
        return self.local_best_effort_executor._is_async_engine()

    async def _maybe_await(self, value: Any) -> Any:
        """仅在值可等待时执行 await。"""
        return await self.local_best_effort_executor._maybe_await(value)

    async def _get_async_engine_tokenizer(self) -> Any:
        """从异步 vLLM 引擎获取 tokenizer。"""
        return await self.local_best_effort_executor._get_async_engine_tokenizer()

    def _load_async_engine_image_asset(self, image_url: str) -> Any:
        """为异步引擎多模态输入加载图片资源。"""
        return self.local_best_effort_executor._load_async_engine_image_asset(image_url)

    def _build_async_engine_multi_modal_data(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """为异步引擎生成构建多模态数据。"""
        return self.local_best_effort_executor._build_async_engine_multi_modal_data(messages)

    def _build_async_engine_generate_input(
        self,
        generate: Any,
        prompt: str,
        multi_modal_data: dict[str, Any] | None,
    ) -> tuple[Any, dict[str, Any]]:
        """构建异步引擎 generate 输入和 kwargs。"""
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
        """把聊天消息渲染为异步引擎 prompt。"""
        return await self.local_best_effort_executor._build_async_engine_chat_prompt(
            messages,
            chat_kwargs,
        )

    def _build_async_engine_sampling_params_instance(self, sampling_params: dict[str, Any]) -> Any:
        """为异步引擎生成构建 sampling params。"""
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
        """通过本地执行器调用异步引擎聊天流。"""
        return await self.local_best_effort_executor._invoke_async_engine_chat_stream(
            messages,
            sampling_params,
            chat_kwargs=chat_kwargs,
            request_id=request_id,
        )

    def _supports_multimodal(self, runtime_context: dict[str, Any] | None = None) -> bool:
        """返回运行时上下文是否支持多模态输入。"""
        return self.local_best_effort_executor._supports_multimodal(runtime_context)

    def _normalize_data_url(self, url: str) -> str:
        """规范化供多模态加载使用的 data URL。"""
        return self.local_best_effort_executor._normalize_data_url(url)

    def _serialize_content_block(self, block: Any) -> dict[str, Any]:
        """为 vLLM 序列化聊天内容块。"""
        return self.local_best_effort_executor._serialize_content_block(block)

    def _is_text_only_content(self, content: Any) -> bool:
        """返回消息内容是否只包含文本。"""
        return self.local_best_effort_executor._is_text_only_content(content)

    def _collapse_text_only_content(self, content: list[Any]) -> str:
        """把纯文本内容块折叠为字符串。"""
        return self.local_best_effort_executor._collapse_text_only_content(content)

    def _serialize_message_content(
        self,
        content: Any,
        *,
        allow_multimodal: bool,
    ) -> str | list[dict[str, Any]]:
        """为本地 best-effort 聊天序列化消息内容。"""
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
        """从 OpenAI 风格请求构建 vLLM 聊天消息。"""
        return self.local_best_effort_executor._build_chat_messages(
            request,
            runtime_context=runtime_context,
        )

    def _should_use_openai_serving_adapter(
        self,
        runtime_spec: dict[str, Any] | None = None,
    ) -> bool:
        """返回聊天是否应使用严格 OpenAI serving。"""
        return self.strict_executor._should_use_openai_serving_adapter(runtime_spec)

    def _should_use_openai_serving_embedding_adapter(
        self,
        runtime_spec: dict[str, Any] | None = None,
    ) -> bool:
        """返回 embeddings 是否应使用严格 OpenAI serving。"""
        return self.strict_executor._should_use_openai_serving_embedding_adapter(runtime_spec)

    def _import_vllm_symbol(self, module_path: str, symbol_name: str) -> Any:
        """通过严格执行器导入 vLLM 符号。"""
        return self.strict_executor._import_vllm_symbol(module_path, symbol_name)

    def _resolve_openai_serving_imports(self) -> ResolvedOpenAIServingImports:
        """通过严格执行器解析 OpenAI serving 导入。"""
        return self.strict_executor._resolve_openai_serving_imports()

    def _resolve_openai_serving_engine_client(self) -> Any | None:
        """通过严格执行器解析 OpenAI serving engine client。"""
        return self.strict_executor._resolve_openai_serving_engine_client()

    def _build_openai_serving_base_model_path(self, base_model_path_cls: type[Any]) -> Any:
        """通过严格执行器构建 vLLM serving base model path。"""
        return self.strict_executor._build_openai_serving_base_model_path(base_model_path_cls)

    def _build_openai_serving_models(
        self,
        *,
        serving_models_cls: type[Any],
        base_model_path_cls: type[Any],
        engine_client: Any,
    ) -> Any:
        """通过严格执行器构建 OpenAI serving models。"""
        return self.strict_executor._build_openai_serving_models(
            serving_models_cls=serving_models_cls,
            base_model_path_cls=base_model_path_cls,
            engine_client=engine_client,
        )

    def _build_openai_serving_render(
        self,
        *,
        serving_render_cls: type[Any],
        engine_client: Any,
        serving_models: Any,
    ) -> Any:
        """通过严格执行器构建 OpenAI serving render 支持。"""
        return self.strict_executor._build_openai_serving_render(
            serving_render_cls=serving_render_cls,
            engine_client=engine_client,
            serving_models=serving_models,
        )

    def _build_openai_serving_chat(
        self,
        *,
        serving_chat_cls: type[Any],
        engine_client: Any,
        serving_models: Any,
        serving_render: Any | None,
    ) -> Any:
        """通过严格执行器构建 OpenAI chat serving。"""
        return self.strict_executor._build_openai_serving_chat(
            serving_chat_cls=serving_chat_cls,
            engine_client=engine_client,
            serving_models=serving_models,
            serving_render=serving_render,
        )

    def _initialize_openai_serving_chat_adapter(self) -> OpenAIChatServingAdapter | None:
        """通过严格执行器初始化 chat serving adapter。"""
        return self.strict_executor._initialize_openai_serving_chat_adapter()

    def _resolve_openai_serving_embedding_imports(self) -> tuple[type[Any], type[Any]]:
        """通过严格执行器解析 embedding serving 导入。"""
        return self.strict_executor._resolve_openai_serving_embedding_imports()

    def _build_openai_serving_embedding(
        self,
        *,
        serving_embedding_cls: type[Any],
        engine_client: Any,
        serving_models: Any,
    ) -> Any:
        """通过严格执行器构建 OpenAI embedding serving。"""
        return self.strict_executor._build_openai_serving_embedding(
            serving_embedding_cls=serving_embedding_cls,
            engine_client=engine_client,
            serving_models=serving_models,
        )

    def _initialize_openai_serving_embedding_adapter(self) -> OpenAIEmbeddingServingAdapter | None:
        """通过严格执行器初始化 embedding serving adapter。"""
        return self.strict_executor._initialize_openai_serving_embedding_adapter()

    def _build_openai_serving_request_payload(
        self,
        request: ChatCompletionsRequest,
        *,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """通过严格执行器构建 chat serving 请求 payload。"""
        return self.strict_executor._build_openai_serving_request_payload(
            request,
            runtime_spec=runtime_spec,
            runtime_context=runtime_context,
        )

    def _build_openai_serving_embedding_request_payload(
        self,
        request: EmbeddingRequest,
        *,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """通过严格执行器构建 embedding serving 请求 payload。"""
        return self.strict_executor._build_openai_serving_embedding_request_payload(
            request,
            runtime_spec=runtime_spec,
            runtime_context=runtime_context,
        )

    async def _call_openai_serving_chat_completion(
        self,
        request_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """通过严格执行器调用 chat serving。"""
        return await self.strict_executor._call_openai_serving_chat_completion(request_payload)

    async def _iter_openai_serving_stream(
        self,
        request_payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any] | bytes | str]:
        """通过严格执行器迭代 chat serving 流式分块。"""
        async for chunk in self.strict_executor._iter_openai_serving_stream(request_payload):
            yield chunk

    async def _call_openai_serving_embedding(
        self,
        request_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """通过严格执行器调用 embedding serving。"""
        return await self.strict_executor._call_openai_serving_embedding(request_payload)

    def _build_chat_stub_response(
        self,
        request: ChatCompletionsRequest,
        runtime_spec: dict[str, Any],
        runtime_context: dict[str, Any],
        sampling_params: dict[str, Any],
        chat_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """构建确定性的 stub chat completion 响应。"""
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
        """把 vLLM chat 输出转换为后端响应结构。"""
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
        """执行非流式 chat completion。"""
        return await self.strict_executor.chat_completion(
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
        """流式执行 chat completion。"""
        async for event in self.strict_executor.chat_completion_stream(
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
        """执行 embedding 请求。"""
        if self._is_vllm_native(runtime_spec):
            return await self.strict_executor.embedding(runtime_spec, request, runtime_context)
        return await self.local_best_effort_executor.embedding(runtime_spec, request, runtime_context)

    async def rerank(
        self,
        runtime_spec: dict[str, Any],
        request: RerankRequest,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """执行 rerank 请求；rerank 当前仅支持 local_best_effort。"""
        if self._is_vllm_native(runtime_spec):
            raise BackendConfigurationError("vllm_native is not supported for VLLM rerank.")
        return await self.local_best_effort_executor.rerank(runtime_spec, request, runtime_context)
