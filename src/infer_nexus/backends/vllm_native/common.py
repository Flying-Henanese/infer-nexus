"""Shared helpers for wrapping vLLM OpenAI serving internals."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResolvedOpenAIServingImports:
    """Resolved vLLM serving imports for one supported internal layout."""

    chat_request_cls: type[Any]
    serving_chat_cls: type[Any]
    serving_models_cls: type[Any]
    base_model_path_cls: type[Any]
    serving_render_cls: type[Any] | None = None


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
