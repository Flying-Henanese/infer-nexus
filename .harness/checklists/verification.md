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

As observed on 2026-09-12, `uv run --frozen pytest -q` completes but is not green: 114 tests pass, 51 fail, and 2 are skipped.

Known contributors:

- shared fixtures and many API/dispatcher/runtime/script tests still expect the previous three-model catalog and aliases, while `config/models.yaml` currently has five enabled models
- a small set of API tests attempts to replace slotted `RuntimeExecutor` instance methods and fails because those attributes are read-only
- `tests/test_dispatcher.py::test_gateway_runtime_metrics_render_after_serve_call`
  assumes a fixed Prometheus label-rendering order (`model`, then `method`),
  while the installed client renders the same labels in a different order
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
- `tests/test_kuberay_manifests.py`
- `tests/test_serve_gateway_poc.py` with `INFER_NEXUS_RUN_RAY_INTEGRATION=1` on the pinned Ray runtime
- For the Serve-native Gateway ingress, verify `/readyz` returns 503 when a
  configured model application is missing or reports a non-running application
  or deployment status. The current helper accepts a `RUNNING` application with
  an empty deployment-status mapping; do not claim that case proves deployment
  readiness until the implementation is tightened.
- For gateway queue saturation, confirm the Serve proxy returns HTTP 503 and
  inspect `serve_deployment_queued_queries`; this rejection happens before
  FastAPI metrics or worker admission.

For real serve-mode validation, use the deployment checklist in `docs/DEPLOYMENT_CHECKLIST.md`.

## For Metrics Or Benchmark Changes

Check:

- `tests/test_benchmark_runner.py`
- `tests/test_benchmark_report.py`
- `tests/test_benchmark_metrics.py`
- `tests/test_api.py` for gateway metrics exposure

Live metric validation requires a running gateway/Ray Serve environment.
