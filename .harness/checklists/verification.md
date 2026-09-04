# Verification Checklist

Use this before saying a change is complete.

## For Documentation Or Harness Changes

- Inspect the diff for scope.
- Confirm links and referenced paths exist.
- Do not claim runtime behavior changed.

Suggested commands:

```bash
git diff --check
git status --short
```

## For Python Code Changes

Run the smallest relevant test set first, then broaden when shared behavior changed.

Common commands:

```bash
uv run --frozen pytest -q
python3 -m compileall src tests
git diff --check
```

If `uv run --frozen` is blocked by local environment state, report that clearly and use compile checks where useful.

## Current Repository Test Baseline

As observed on 2026-09-04, `uv run --frozen pytest -q` completes but is not green:
115 tests pass, 47 fail, and 2 are skipped.

Known contributors:

- shared fixtures and many API/dispatcher/runtime/script tests still expect the
  previous multi-model catalog and aliases, while `config/models.yaml` now
  enables only Qwen3.5-9B for Compose validation
- the installed `uv` warns that `tool.uv.extra-build-dependencies` is not recognized unless the relevant preview support/version is used

For unrelated work, run the smallest relevant tests and compare any full-suite failures with this baseline. Do not describe the repository as fully green, and do not treat every known baseline failure as caused by a documentation-only change.

## For Runtime/Gateway Changes

Check the relevant unit tests around:

- `tests/test_api.py`
- `tests/test_dispatcher.py`
- `tests/test_main.py`
- `tests/test_runtime.py`
- `tests/test_proxy_streaming.py`
- `tests/test_gateway_ingress.py`
- `tests/test_serve_gateway_poc.py` with `INFER_NEXUS_RUN_RAY_INTEGRATION=1` on the pinned Ray runtime
- In Serve mode, verify `/readyz` returns 503 while a configured model
  application is unavailable, then 200 only after every model app is healthy.
- For gateway queue saturation, confirm the Serve proxy returns HTTP 503 and
  inspect `serve_deployment_queued_queries`; this rejection happens before
  FastAPI metrics or worker admission.

For real Serve-mode validation, use the bring-up sequence in
`docs/DEPLOYMENT_CHECKLIST.md` together with
`.harness/workflows/remote-deploy-and-validate.md`. The deployment checklist's
chat example names disabled `qwen3.5-27b`; use active alias `qwen3.5-9b`.

For Compose validation, distinguish these two levels in the report:

- A request from the host to `127.0.0.1:8000` (mapped to `ray-head:8000`) is
  an end-to-end deployment check; it exercises the Serve proxy, Gateway
  ingress, model deployment, and vLLM runtime.
- A unit test or `docker compose config` render only checks local code or
  configuration. It does not validate an accelerator image, device mounts, or
  a real model.

The CUDA Compose path has completed the first level on A100 for the active
Qwen3.5-9B model. The Ascend path has only completed configuration and unit
checks, so retain an explicit Ascend end-to-end validation step before claiming
platform support.

## For Metrics Or Benchmark Changes

Check:

- `tests/test_benchmark_runner.py`
- `tests/test_benchmark_report.py`
- `tests/test_benchmark_metrics.py`
- `tests/test_api.py` for gateway metrics exposure

Live metric validation requires a running gateway/Ray Serve environment.
