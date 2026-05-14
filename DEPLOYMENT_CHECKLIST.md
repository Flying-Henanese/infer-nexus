# Deployment Checklist

This checklist is for the first real bring-up of `infer-nexus` on `Ubuntu x86_64 + NVIDIA CUDA`.

Use it before and during the first runtime validation. The goal is to detect environment and configuration faults early, before they look like application bugs.

## 1. Machine and Driver Baseline

- Confirm the target machine is `x86_64 Linux`.
- Confirm the NVIDIA driver is installed and the GPUs are visible:

```bash
nvidia-smi
```

- Confirm the intended GPU pool is actually present on the machine.
- Confirm enough GPU memory exists for the registered models and their minimum replica counts.

## 2. Python and Tooling

- Confirm Python is `3.11+`:

```bash
python3 --version
```

- Confirm `uv` is installed:

```bash
uv --version
```

- From the repository root, install runtime dependencies:

```bash
uv sync --extra serve --extra vllm
```

- If you also need the offline model download utility:

```bash
uv sync --extra serve --extra vllm --extra artifacts
```

## 3. Local Model Store

- Confirm `config/settings.yaml` points `model_store.root_dir` at the correct local model directory.
- Confirm every registered model path exists under the local model store root.
- Confirm the runtime will not need to download models from the network at startup.

Quick sanity check:

```bash
find models -maxdepth 3 -type d | sort
```

## 4. Registered Model Configuration

- Review `config/models.yaml`.
- Confirm each model has the correct:
  - `task`
  - `backend`
  - `model_path`
  - `tensor_parallel_size`
  - `gpu_per_replica`
  - `min_replicas`
  - `max_replicas`
- Confirm `model_path` is a local-path semantic value, not a Hugging Face repo id.
- Confirm the sum of `gpu_per_replica * min_replicas` across all models fits within the intended pool.

## 5. Runtime Settings

- For real Ray Serve validation, confirm:

```yaml
runtime:
  execution_mode: serve
  backend_init_mode: real
```

- Confirm `cluster.accelerator_type` still matches the actual machine type.
- Confirm `runtime.backend` is still `vllm`.

## 6. Resource Pool Boundary

- Decide which GPUs belong to `infer-nexus`.
- Export only those devices before starting Ray:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

- Start Ray with matching logical GPU count:

```bash
ray start --head --num-gpus=4
```

- If you expose 4 GPUs to Ray but your model minima require 6, startup failure is expected. Fix the configuration before going further.

## 7. Preflight Tests Before Runtime Bring-Up

- Run the local unit test suite:

```bash
uv run pytest -q
```

- If these tests fail, do not start debugging CUDA first. Fix the application-level issue before mixing in environment complexity.

## 8. Start Order

1. Start Ray.
2. Start the Serve runtime:

```bash
uv run python scripts/run_serve_runtime.py --ray-address auto
```

3. Start the gateway:

```bash
uv run python scripts/run_gateway.py
```

## 9. Smoke Validation

- Health:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

- Model discovery:

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/api/catalog/models
```

- Chat:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-chat",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

- Embeddings:

```bash
curl -X POST http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bge-embedding",
    "input": "hello world"
  }'
```

- Rerank:

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

## 10. Failure Interpretation

- `503 model_artifact_missing`
  - Local model directory is missing or `model_path` is wrong.
- `400 unsupported_task_type`
  - The request hit a model registered for the wrong task.
- `400 unsupported_parameter`
  - Phase 1 feature is not implemented, for example `stream=true`.
- `501 runtime_not_connected`
  - The gateway cannot resolve or invoke the Serve deployment handle.
- Startup crash during Serve runtime deployment
  - Usually means resource sizing, backend initialization, or model artifact issues.

## 11. Acceptance Criteria

The first deployment is good enough to continue only if:

- Ray starts with the intended GPU pool boundary.
- Serve runtime deploys without crashing.
- Gateway starts and stays healthy.
- `/v1/models` returns the expected registered aliases.
- At least one chat request succeeds.
- At least one embedding request succeeds.
- At least one rerank request succeeds.
- No request path falls back to unexpected `500` responses.
