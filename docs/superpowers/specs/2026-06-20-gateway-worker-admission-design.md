# Gateway Worker Admission Design

## Scope

Keep Uvicorn's existing shared-listener, multi-process distribution and add complete process-local overload protection. This phase does not add a user-space gateway, independent worker endpoints, global admission, retry routing, or worker removal.

## Design

Each Uvicorn process owns one `WorkerAdmissionController`. A pure ASGI middleware acquires a process-local slot before selected inference paths enter FastAPI and releases it only after the final response body, application failure, or disconnect. This makes streaming requests occupy capacity for their complete lifetime.

The worker layer never queues. At capacity it returns `503 Service Unavailable`, an OpenAI-compatible `gateway_worker_overloaded` error, and `Retry-After`. Tenant/model admission remains `429`; the existing deployment guard retains per-model concurrency, bounded queuing, timeout, and circuit-breaker behavior.

Health, readiness, metrics, and model-list endpoints bypass worker admission so diagnostics remain available during overload. `/readyz` remains unchanged because requests to a shared listener cannot target a specific worker.

## Components

- `control/worker_admission.py`: framework-independent process-local controller and immutable snapshot.
- `api/worker_admission_middleware.py`: inference-path filtering, ASGI lifecycle handling, and overload response.
- `main.py`: constructs the controller per worker and installs middleware.
- `runtime/executor.py`: removes the old Serve-handle-only worker guard while retaining deployment guards.
- `observability/metrics.py`: records per-PID capacity, inflight, and rejection metrics.

## Configuration

`runtime.gateway_worker_max_inflight` remains backward compatible and explicitly means a per-process limit. `runtime.gateway_worker_retry_after_seconds` defaults to 1. A zero inflight limit disables worker admission.

## Testing

Tests use controller concurrency, a small ASGI application, and existing fake Serve handles. No Ray or vLLM process is required. Coverage includes fail-fast rejection, disabled admission, stream lifetime, exception cleanup, path bypass, API error mapping, and removal of duplicate executor-level worker accounting.

## Deferred Work

Independent worker endpoints, a global ingress, worker-aware balancing, retries, global limits, process removal, and Prometheus multiprocess aggregation are separate follow-up phases.
