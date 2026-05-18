# Minimal Startup

This document defines the smallest practical startup flow for `infer-nexus` on a single Ubuntu + CUDA node.

Before using this flow on a real machine, go through [DEPLOYMENT_CHECKLIST.md](/Users/zhoushujian/Projects/GitHub/infer-nexus/DEPLOYMENT_CHECKLIST.md).

## Goal

Bring up:

1. a Ray Serve runtime that owns model deployments
2. a FastAPI gateway that exposes the northbound APIs

The two processes are separate for now:

- Ray Serve runtime hosts the model replicas and deployment handles
- FastAPI gateway receives `/v1/...` requests and dispatches to Serve handles

This is enough to validate the end-to-end control flow before doing more production packaging.

## Prerequisites

- Ubuntu x86_64
- NVIDIA driver and CUDA environment already working
- local model artifacts already downloaded under `model_store.root_dir`
- Python 3.11+
- `uv` installed

## Install

Install the runtime dependencies:

```bash
uv sync --extra serve --extra vllm
```

If you also need the offline download script:

```bash
uv sync --extra serve --extra vllm --extra artifacts
```

## Settings

For a first runtime validation, update [config/settings.yaml](/Users/zhoushujian/Projects/GitHub/infer-nexus/config/settings.yaml):

```yaml
runtime:
  execution_mode: serve
  backend_init_mode: real
```

Keep `model_store.root_dir` pointing to the local model directory.

## Resource Pool Boundary

Define the GPU pool boundary before starting Ray.

Example: expose only 4 A100s to `infer-nexus`:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
ray start --head --num-gpus=4
```

This follows Ray's recommended resource boundary pattern:

- `CUDA_VISIBLE_DEVICES` constrains what the Ray node can see
- `--num-gpus` tells Ray the logical GPU capacity
- deployments only request GPU quantities; they do not pin device ids

Ray docs:

- `ray start --head --num-gpus=<NUM_GPUS>`: https://docs.ray.io/en/latest/ray-core/configure.html
- `ray.serve.get_deployment_handle(..., app_name=...)`: https://docs.ray.io/en/latest/serve/api/doc/ray.serve.get_deployment_handle.html
- `serve.run(..., name=..., route_prefix=None)`: https://docs.ray.io/en/latest/serve/api/doc/ray.serve.run.html

## Start the Serve Runtime

Deploy model replicas into Ray Serve:

```bash
uv run python scripts/run_serve_runtime.py --ray-address auto
```

Notes:

- The script uses `serve.start(proxy_location="Disabled")` by default.
- This avoids starting a Serve HTTP proxy and prevents a port conflict with the gateway.
- The runtime is deployed as the Serve application named `infer-nexus`.

## Start the Gateway

Start the northbound API gateway:

```bash
uv run python scripts/run_gateway.py
```

Default bind:

- host: `0.0.0.0`
- port: `8000`

These come from [config/settings.yaml](/Users/zhoushujian/Projects/GitHub/infer-nexus/config/settings.yaml).

## Minimal Validation

Check model discovery:

```bash
curl http://127.0.0.1:8000/v1/models
```

Check health:

```bash
curl http://127.0.0.1:8000/healthz
```

Check embeddings:

```bash
curl -X POST http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bge-embedding",
    "input": "hello world"
  }'
```

Check rerank:

```bash
curl -X POST http://127.0.0.1:8000/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bge-rerank",
    "query": "capital of france",
    "documents": [
      "The capital of Brazil is Brasilia.",
      "The capital of France is Paris."
    ],
    "top_n": 1
  }'
```

Check chat:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-chat",
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  }'
```

## One-Click Script

To bring up the whole minimal flow with one command (install deps + start/attach Ray + deploy Serve runtime + start gateway), run:

```bash
scripts/start_minimal.sh --cuda-visible-devices 0,1,2,3 --num-gpus 4
```

Stop everything (including Ray) with:

```bash
scripts/stop_minimal.sh
```

## Current Limitations

- The gateway and Serve runtime are still separate processes.
- True single-port serving is not implemented yet.
- `stream=true` is still rejected in Phase 1.
- VLM is still intentionally unsupported.
- Some models may need vLLM-specific startup overrides such as score templates or `hf_overrides`.

## Convergence Direction

The current two-process shape is deliberate. It keeps the northbound gateway and the Ray Serve runtime separate while the platform is still proving out model loading, task dispatch, and backend initialization.

The intended convergence direction is:

1. keep a single external entrypoint for clients
2. move from "gateway process + separate runtime launcher" toward one operator-facing startup command
3. keep Ray Serve as the runtime owner of model replicas
4. avoid reintroducing per-model external ports

This means the long-term target is not "clients call Serve directly". The target is "one operator starts infer-nexus once, and clients still see one northbound API entrypoint".

## Shutdown

Stop the gateway process and the runtime process.

Then stop Ray:

```bash
ray stop
```
