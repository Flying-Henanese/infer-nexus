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

## For Runtime/Gateway Changes

Check the relevant unit tests around:

- `tests/test_api.py`
- `tests/test_dispatcher.py`
- `tests/test_main.py`
- `tests/test_runtime.py`
- `tests/test_proxy_streaming.py`

For real serve-mode validation, use the deployment checklist in `docs/DEPLOYMENT_CHECKLIST.md`.

## For Metrics Or Benchmark Changes

Check:

- `tests/test_benchmark_runner.py`
- `tests/test_benchmark_report.py`
- `tests/test_benchmark_metrics.py`
- `tests/test_api.py` for gateway metrics exposure

Live metric validation requires a running gateway/Ray Serve environment.

