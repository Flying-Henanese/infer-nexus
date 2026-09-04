"""Real-Ray acceptance POC for cross-application Serve streaming.

Run explicitly against the pinned Ray runtime:

    INFER_NEXUS_RUN_RAY_INTEGRATION=1 pytest -q tests/test_serve_gateway_poc.py
"""

from __future__ import annotations

import asyncio
import os
import socket
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from ray import serve
except ModuleNotFoundError:  # pragma: no cover - the integration test is skipped without Ray.
    serve = None


pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


MODEL_APPLICATION_NAME = "poc-model-app"
MODEL_DEPLOYMENT_NAME = "poc-model"
GATEWAY_APPLICATION_NAME = "poc-gateway-app"


if serve is not None:
    @serve.deployment(name=MODEL_DEPLOYMENT_NAME, num_replicas=1)
    class POCStreamingModel:
        def __init__(self, version: str = "before") -> None:
            self.version = version
            self.cancelled = False
            self.finalized = False

        async def unary(self) -> dict[str, str]:
            return {"value": self.version}

        async def stream(self, mode: str = "success") -> AsyncIterator[str]:
            if mode == "before_error":
                raise ValueError("before first chunk")
            yield "one"
            if mode == "after_error":
                raise ValueError("after first chunk")
            if mode == "wait_for_cancel":
                try:
                    await asyncio.sleep(30)
                    yield "unexpected"
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
                finally:
                    self.finalized = True
                return
            yield "two"

        async def reset_cancellation(self) -> None:
            self.cancelled = False
            self.finalized = False

        async def cancellation_state(self) -> dict[str, bool]:
            return {"cancelled": self.cancelled, "finalized": self.finalized}


    poc_api = FastAPI()

    @serve.deployment(name="poc-gateway", num_replicas=1)
    @serve.ingress(poc_api)
    class POCGateway:
        def __init__(self) -> None:
            self.model = serve.get_deployment_handle(
                MODEL_DEPLOYMENT_NAME,
                app_name=MODEL_APPLICATION_NAME,
            )

        @poc_api.get("/unary")
        async def unary(self) -> JSONResponse:
            return JSONResponse(content=await self.model.unary.remote())

        @poc_api.get("/stream")
        async def stream(self, mode: str = "success") -> StreamingResponse:
            stream = self.model.stream.options(stream=True).remote(mode)
            iterator = stream.__aiter__()
            try:
                first_chunk = await iterator.__anext__()
            except StopAsyncIteration:
                first_chunk = None
            except Exception as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

            async def sse() -> AsyncIterator[str]:
                if first_chunk is not None:
                    yield f"data: {first_chunk}\n\n"
                async for chunk in iterator:
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        @poc_api.post("/cancellation/reset")
        async def reset_cancellation(self) -> JSONResponse:
            await self.model.reset_cancellation.remote()
            return JSONResponse(content={"ok": True})

        @poc_api.get("/cancellation/state")
        async def cancellation_state(self) -> JSONResponse:
            return JSONResponse(content=await self.model.cancellation_state.remote())


@pytest.mark.skipif(
    os.getenv("INFER_NEXUS_RUN_RAY_INTEGRATION") != "1",
    reason="Set INFER_NEXUS_RUN_RAY_INTEGRATION=1 to run the real Ray Serve POC.",
)
def test_cross_application_streaming_reaches_http_ingress() -> None:
    """A Serve ingress must consume another application's streaming handle internally."""
    ray = pytest.importorskip("ray", reason="Ray Serve is required for this integration POC.")
    import httpx

    assert serve is not None
    proxy_port = _free_port()

    async def exercise() -> None:
        base_url = f"http://127.0.0.1:{proxy_port}/poc"
        async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
            unary = await client.get("/unary")
            assert unary.status_code == 200
            assert unary.json() == {"value": "before"}

            stream = await client.get("/stream")
            assert stream.status_code == 200
            assert stream.headers["content-type"].startswith("text/event-stream")
            assert stream.text == "data: one\n\ndata: two\n\ndata: [DONE]\n\n"

            before_error = await client.get("/stream", params={"mode": "before_error"})
            assert before_error.status_code == 503
            assert "before first chunk" in before_error.text

            async with client.stream("GET", "/stream", params={"mode": "after_error"}) as after_error:
                assert after_error.status_code == 200
                assert await anext(after_error.aiter_lines()) == "data: one"

            assert (await client.post("/cancellation/reset")).status_code == 200
            cancellation_stream = await client.send(
                client.build_request("GET", "/stream", params={"mode": "wait_for_cancel"}),
                stream=True,
            )
            assert cancellation_stream.status_code == 200
            assert await anext(cancellation_stream.aiter_lines()) == "data: one"
            await cancellation_stream.aclose()

            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                state = (await client.get("/cancellation/state")).json()
                if state["finalized"]:
                    break
                await asyncio.sleep(0.2)
            assert state == {"cancelled": True, "finalized": True}

            serve.run(POCStreamingModel.bind("after"), name=MODEL_APPLICATION_NAME, route_prefix=None)
            redeployed = await client.get("/unary")
            assert redeployed.status_code == 200
            assert redeployed.json() == {"value": "after"}

    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        _temp_dir=f"/tmp/infer-nexus-serve-poc-{os.getpid()}",
    )
    try:
        serve.start(
            proxy_location="HeadOnly",
            http_options={"host": "127.0.0.1", "port": proxy_port},
        )
        serve.run(POCStreamingModel.bind(), name=MODEL_APPLICATION_NAME, route_prefix=None)
        serve.run(POCGateway.bind(), name=GATEWAY_APPLICATION_NAME, route_prefix="/poc")
        asyncio.run(exercise())
        status = serve.status()
        assert MODEL_APPLICATION_NAME in str(status)
        assert GATEWAY_APPLICATION_NAME in str(status)
    finally:
        serve.shutdown()
        ray.shutdown()


@pytest.mark.skipif(
    os.getenv("INFER_NEXUS_RUN_RAY_INTEGRATION") != "1",
    reason="Set INFER_NEXUS_RUN_RAY_INTEGRATION=1 to run the real Ray Serve POC.",
)
def test_real_gateway_ingress_reuses_fastapi_without_ray_client(tmp_path: Path) -> None:
    """The production binding must serve normal routes from inside a Serve replica."""
    ray = pytest.importorskip("ray", reason="Ray Serve is required for this integration POC.")
    import httpx
    from ray import serve

    from infer_nexus.catalog.loader import load_model_catalog
    from infer_nexus.catalog.registry import ModelRegistry
    from infer_nexus.core.config import Settings
    from infer_nexus.model_store import LocalModelStore
    from infer_nexus.runtime.serve_app import ServeApplicationBuilder

    catalog_path = tmp_path / "models.yaml"
    model_path = tmp_path / "poc-model"
    model_path.mkdir()
    catalog_path.write_text(
        "models:\n"
        "  - name: poc-model\n"
        "    alias: poc-model\n"
        "    task: chat\n"
        "    backend: vllm\n"
        f"    model_path: {model_path}\n"
        "    tensor_parallel_size: 1\n"
        "    cpu_per_replica: 1\n"
        "    gpu_per_replica: 0\n"
        "    min_replicas: 1\n"
        "    max_replicas: 1\n",
        encoding="utf-8",
    )
    settings = Settings()
    settings.catalog.models_path = str(catalog_path)
    settings.model_store.root_dir = str(tmp_path / "models")
    settings.runtime.execution_mode = "serve"
    settings.runtime.backend_init_mode = "stub"
    settings.runtime.gateway_ingress.application_name = "poc-real-gateway"
    settings.runtime.gateway_ingress.route_prefix = "/poc"
    proxy_port = _free_port()
    registry = ModelRegistry(load_model_catalog(catalog_path))
    builder = ServeApplicationBuilder(
        model_store=LocalModelStore(settings.model_store.root_dir),
        backend_init_mode=settings.runtime.backend_init_mode,
        service_name=settings.service.name,
    )

    ray.init(
        address="local",
        # The stub model reserves one CPU and the ingress reserves 0.5 CPU.
        # Leave headroom so this validates both deployments rather than
        # waiting forever for the bounded ingress placement.
        num_cpus=2,
        include_dashboard=False,
        _temp_dir=f"/tmp/infer-nexus-real-ingress-poc-{os.getpid()}",
    )
    try:
        serve.start(
            proxy_location="HeadOnly",
            http_options={"host": "127.0.0.1", "port": proxy_port},
        )
        spec = builder.build_gateway_spec(registry, settings=settings)
        model_bindings = builder.build_serve_bindings(registry, serve=serve)
        serve.run(
            model_bindings["poc-model"],
            name=builder.build_application_name("poc-model"),
            route_prefix=None,
        )
        serve.run(
            builder.build_gateway_binding(registry, settings=settings, serve=serve),
            name=spec.application_name,
            route_prefix=spec.route_prefix,
        )
        with httpx.Client(base_url=f"http://127.0.0.1:{proxy_port}/poc", timeout=10) as client:
            assert client.get("/healthz").status_code == 200
            assert client.get("/readyz").status_code == 200
            models = client.get("/v1/models")
            assert models.status_code == 200
            assert models.json()["data"][0]["id"] == "poc-model"
            completion = client.post(
                "/v1/chat/completions",
                json={"model": "poc-model", "messages": [{"role": "user", "content": "hello"}]},
            )
            assert completion.status_code == 200
            assert completion.json()["object"] == "chat.completion"
            streamed = client.post(
                "/v1/chat/completions",
                json={
                    "model": "poc-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
            assert streamed.status_code == 200
            assert streamed.text.count("data: [DONE]") == 1
        assert spec.application_name in str(serve.status())
    finally:
        serve.shutdown()
        ray.shutdown()
