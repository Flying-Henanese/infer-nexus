# Gateway Worker Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add complete process-local, fail-fast admission for inference requests while retaining Uvicorn shared-port distribution.

**Architecture:** A framework-independent controller owns per-process capacity. A pure ASGI middleware holds a slot through the full HTTP or streaming lifecycle and emits a stable retryable 503 response at capacity; model-level guards remain inside `RuntimeExecutor`.

**Tech Stack:** Python 3.11+, FastAPI/Starlette ASGI, asyncio, prometheus-client, pytest

---

### Task 1: Process-local admission controller

**Files:**
- Create: `src/infer_nexus/control/worker_admission.py`
- Create: `tests/test_worker_admission.py`
- Modify: `src/infer_nexus/observability/metrics.py`

- [ ] Write controller tests for acquire, fail-fast capacity rejection, release, disabled limits, snapshots, and concurrent callers.
- [ ] Run `pytest tests/test_worker_admission.py -q` and confirm failure because the module does not exist.
- [ ] Implement `WorkerAdmissionController`, `WorkerAdmissionSnapshot`, and `WorkerOverloadedError` with an `asyncio.Lock` and no local queue.
- [ ] Add capacity, inflight, and rejection metric methods keyed by PID.
- [ ] Run `pytest tests/test_worker_admission.py -q` and confirm all tests pass.

### Task 2: Full-lifecycle ASGI middleware

**Files:**
- Create: `src/infer_nexus/api/worker_admission_middleware.py`
- Create: `tests/test_worker_admission_middleware.py`

- [ ] Write middleware tests for protected and bypassed paths, 503 error shape, `Retry-After`, normal release, streaming release, and exception release.
- [ ] Run `pytest tests/test_worker_admission_middleware.py -q` and confirm failure because the middleware does not exist.
- [ ] Implement pure ASGI middleware that resolves the controller from `scope["app"].state`, wraps `send`, and releases exactly once in `finally`.
- [ ] Run `pytest tests/test_worker_admission_middleware.py -q` and confirm all tests pass.

### Task 3: Application configuration and wiring

**Files:**
- Modify: `src/infer_nexus/core/config.py`
- Modify: `src/infer_nexus/main.py`
- Modify: `config/settings.yaml`
- Modify: `tests/test_main.py`
- Modify: `tests/test_api.py`

- [ ] Write failing tests for the retry-after setting, controller lifespan wiring, inference overload 503, and health bypass.
- [ ] Run the targeted tests and confirm the new expectations fail.
- [ ] Add `gateway_worker_retry_after_seconds`, install the middleware, and construct the per-worker controller during lifespan.
- [ ] Run the targeted tests and confirm they pass.

### Task 4: Remove duplicate runtime worker guard

**Files:**
- Modify: `src/infer_nexus/runtime/executor.py`
- Modify: `tests/test_dispatcher.py`

- [ ] Replace the executor-level worker guard test with coverage proving per-model guards still reject and circuit-break correctly.
- [ ] Remove `_GatewayWorkerGuard`, `_worker_guard`, `_get_worker_guard`, and worker acquire/release branches from Serve handle calls.
- [ ] Run `pytest tests/test_dispatcher.py -q` and confirm the fake-handle tests pass.

### Task 5: Startup semantics and verification

**Files:**
- Modify: `scripts/run_gateway.py`
- Modify: `tests/test_scripts.py`

- [ ] Add a failing script test for a startup log that distinguishes per-worker and theoretical aggregate capacity.
- [ ] Add the startup log without changing `uvicorn.run(..., workers=N)`.
- [ ] Run `pytest tests/test_worker_admission.py tests/test_worker_admission_middleware.py tests/test_main.py tests/test_api.py tests/test_dispatcher.py tests/test_scripts.py -q`.
- [ ] Run `python -m compileall -q src tests` and `git diff --check`.
