# Next Session Handoff

## Current Snapshot (2026-05-18)

The project is running in real Ray Serve + vLLM mode on Ubuntu/CUDA and is past the initial scaffolding stage.

Current active model catalog:
- `Qwen3-VL-8B-Instruct` (`alias: qwen3-vl-chat-8b-instruct`)
- `Qwen3-8B` (`alias: qwen3-8b`)

Current startup defaults in `scripts/start_minimal.sh`:
- `CUDA_VISIBLE_DEVICES_VALUE="1,3"`
- Ray is started only when no existing cluster is reachable.

## What Was Recently Fixed

1. vLLM 0.18.1 compatibility in backend startup
- `src/infer_nexus/backends/vllm.py` now avoids passing unsupported `task` kwargs.
- Task support check falls back safely across versions.

2. Model-level memory/context knobs are now wired from config
- `gpu_memory_utilization` added and validated in model schema.
- `max_model_len` is now passed through runtime spec into `LLM(...)`.
- Confirmed in logs via `non-default args` and `max_seq_len=...`.

3. Chat call compatibility across vLLM versions
- `LLM.chat()` invocation now uses `SamplingParams`-compatible path for vLLM 0.18.1.
- Fixes `TypeError: unexpected keyword argument 'temperature'`.

4. Stop script hardening
- `scripts/stop_minimal.sh` now force-stops Ray (`ray stop -f`) and improves cleanup behavior for child/stray inference processes.

## Current Main Runtime Problem

`max_model_len` is now effective, but deployment can still fail due to GPU memory pressure during KV cache initialization:

- Typical failure:
  - `ValueError: No available memory for the cache blocks`
  - `Available KV cache memory: <negative GiB>`

This is not a config-parsing bug now. It is a scheduling/capacity issue when both models initialize under current GPU sharing behavior.

## Why It Fails

- Ray schedules using logical GPU resources (`gpu_per_replica`) rather than real-time free VRAM.
- With fractional GPU requests, replicas can still land on the same physical card.
- vLLM graph capture + weights + KV cache budget can exceed available memory on that card.

## Recommended Next Actions

1. Enforce model-to-device separation for stability-first validation
- Prefer `gpu_per_replica: 1` for both models initially.
- Set both models to `min_replicas: 1`, `max_replicas: 1`.
- Keep startup bounded with `--cuda-visible-devices 1,3 --num-gpus 2`.

2. If fractional GPU must be kept
- Add explicit placement/resource constraints in deployment actor options (code change required).
- Do not rely on fractional `num_gpus` alone for physical separation.

3. Keep memory knobs conservative while dual-model bring-up is unstable
- Reduce `max_model_len` first, then tune `gpu_memory_utilization`.
- Remember: in multi-model same-host scenarios, increasing `gpu_memory_utilization` can worsen contention.

4. Verify each rollout with log checkpoints
- `non-default args` shows expected `max_model_len` and `gpu_memory_utilization`.
- `max_seq_len=...` matches catalog config.
- No `No available memory for the cache blocks` in either model replica.

## Fast Verification Commands

Health:
```bash
curl http://127.0.0.1:8000/healthz
```

Model list:
```bash
curl http://127.0.0.1:8000/v1/models
```

Chat test (`qwen3-8b`):
```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "请回复: service ready"}],
    "temperature": 0.1,
    "max_tokens": 32,
    "stream": false
  }'
```

## Files Most Likely To Touch Next

- `config/models.yaml`
- `scripts/start_minimal.sh`
- `scripts/stop_minimal.sh`
- `src/infer_nexus/backends/vllm.py`
- `src/infer_nexus/runtime/deployments.py` (if adding placement constraints)
