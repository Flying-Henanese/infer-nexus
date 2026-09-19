# Logging Management Execution Plan

Status: historical plan, application-layer portions substantially implemented

Prepared: 2026-09-14
Target implementation start: 2026-09-15

> Do not execute this plan from the beginning. Verify current source first.
> The remaining Compose stdout, application event-file, local query, retention,
> and documentation work is specified in
> `logging-storage-query-redesign-plan.md`.

## Objective

Replace the current shallow logging setup with one deep logging module that
provides a small application-facing interface while hiding formatting,
context propagation, redaction, framework configuration, and serialization.

The completed design must provide:

- one-line JSON logs (JSONL) in Compose and KubeRay environments
- readable console logs for local development
- one canonical request ID across the Gateway, Ray Serve model handle, model
  replica, and proxy upstream path
- stable event names and queryable fields
- exactly one authoritative terminal application event per request
- explicit adapters for Uvicorn, Ray Serve, and vLLM logging behavior
- a real collection path for node-local Ray logs
- tests that cover the logging interface rather than formatter internals

## Scope

This plan covers application logs, request lifecycle logs, Ray Serve logging
configuration, vLLM logging configuration, and deployment-side log collection.

It does not add:

- distributed tracing or an OpenTelemetry collector
- prompt, message, embedding input, or rerank document logging
- a general-purpose pluggable log sink interface
- a Web UI
- changes to inference, admission, or scaling policy
- dynamic model registration or a new runtime topology

Metrics remain in Prometheus. Logs provide request-level explanation and
diagnostic context; they must not duplicate time-series data at high volume.

## Current Baseline

### Configuration and bootstrap

- `ServiceSettings.log_level` is the only application logging setting.
- `src/infer_nexus/observability/logging.py` calls `logging.basicConfig()` with
  a plain-text formatter.
- `GatewayRuntime.create()` is the only active logging bootstrap call.
- The Serve deployment driver and `ModelRuntimeReplica` do not execute the same
  bootstrap.
- `basicConfig()` is not a reliable ownership mechanism once Ray, Uvicorn, or
  vLLM has already installed handlers.

### Application events

There are currently 15 explicit application logger calls across four source
files:

- `api/openai_routes.py`: one request error log
- `runtime/executor.py`: four Serve/SSE error logs
- `backends/vllm.py`: six backend initialization/fallback logs
- `backends/vllm_strict.py`: four initialization/invocation logs

Fields such as model, deployment, request ID, and error reason are interpolated
into message strings rather than emitted as stable structured fields.

The current error flow can record the same failure at the backend, executor,
and route layers. Some SSE iterators catch an error, log it, and return cleanly,
which can produce an error log while the outer stream metric records success.

### Request identity

- The route observation `ContextVar` only carries metric state.
- The executor creates unrelated UUIDs for proxy calls and streaming response
  headers.
- Request identity is not explicitly propagated as a separate argument over a
  Ray Serve deployment handle.
- Direct Uvicorn and Serve-ingress execution do not share a canonical request
  identity source.

### Storage and collection

- Local bootstrap scripts write the `ray start` command output to
  `.infer-nexus/logs/ray.log` and the one-shot deployment driver output to
  `.infer-nexus/logs/serve_runtime.log`.
- Actual Ray system, worker, actor, and Serve logs are managed under the Ray
  session directory, normally `/tmp/ray/session_*/logs`.
- Compose mounts `/app/.infer-nexus/logs`, but current Compose processes do not
  write their Ray session logs there.
- The KubeRay POC has no shared `/tmp/ray` volume or log collector.

## Architectural Decisions

### AD-1: Standard logging remains the underlying mechanism

Use Python `logging` and `logging.config.dictConfig` as the implementation
foundation. Ray, Uvicorn, and vLLM already use standard logging, so introducing
another public logging ecosystem would increase integration cost.

Use `python-json-logger` as a direct dependency if it satisfies the contract.
Do not rely on its current transitive installation through vLLM. If it cannot
provide deterministic exception and field serialization, implement a private
JSON formatter behind the same module interface.

### AD-2: Application logs use JSONL in deployed environments

Each application log record is one complete JSON object followed by one
newline. Stack traces must be serialized inside a JSON string or structured
exception object and must not create raw continuation lines.

Profiles:

- local development: `console`
- tests: `json` or an in-memory handler selected by the test
- Compose: `json`
- KubeRay: `json`

### AD-3: The application interface stays small

The public interface of the logging module should be limited to:

```python
configure_logging(settings, process_identity)
get_logger(module_name)
bind_log_context(**fields)
```

Expected use:

```python
logger = get_logger(__name__)

with bind_log_context(request_id=request_id, model=model, task=task):
    logger.info(
        "request.completed",
        outcome="success",
        status_code=200,
        duration_ms=duration_ms,
    )
```

The implementation owns:

- handler and formatter configuration
- context merging and cleanup
- JSON serialization fallbacks
- UTC timestamp formatting
- exception serialization
- sensitive-field redaction
- reserved field protection
- idempotent process bootstrap
- named logger level overrides

Callers must not instantiate formatters, handlers, adapters, or context
variables directly.

### AD-4: Do not add a general sink port

The application emits records to standard process logging. File discovery,
tailing, retention, and export are deployment concerns. There are not yet two
application-level sink adapters that justify a public sink seam.

### AD-5: Framework variation uses internal adapters

Uvicorn, Ray Serve, and vLLM have materially different initialization and
process-lifecycle behavior, so they form real internal adapter seams. Keep
their imports and version-specific behavior private to the logging module or
runtime bootstrap layer.

## Target Architecture

```text
LoggingSettings
      |
      v
Logging bootstrap -----------------------------+
      |                                        |
      |                           framework adapters
      |                         /        |         \
      v                    Uvicorn   Ray Serve    vLLM
Request context middleware
      |
      +--> Gateway semantic event
      +--> Ray handle request_context
      +--> Model replica context
      +--> Proxy X-Request-ID / future traceparent
      |
      v
application JSONL + Ray node-local logs
      |
      v
Fluent Bit / Vector
      |
      v
central store such as Loki or ELK
```

Application semantic logs and third-party framework logs may have different
schemas. The collector must preserve a `source` field such as `infer_nexus`,
`ray_serve`, `ray_core`, or `vllm`, then add deployment metadata without
rewriting application event fields.

## Configuration Contract

Move logging configuration out of `service` and into a top-level
`observability` section:

```yaml
observability:
  logging:
    format: json
    level: INFO
    success_sample_rate: 1.0
    slow_request_ms: 10000
    include_traceback: true
    access_log: true
    named_levels:
      infer_nexus: INFO
      ray: INFO
      ray.serve: INFO
      uvicorn.access: WARNING
      vllm: INFO
```

Rules:

- Validate levels and formats; invalid values fail startup.
- Keep a temporary compatibility read of `service.log_level` for one migration
  window, with the new setting taking precedence.
- Environment overrides may change level and format, but must use explicit
  `INFER_NEXUS_LOG_*` names.
- Do not place log directories, retention periods, or collector credentials in
  application settings.
- `success_sample_rate` applies only to ordinary successful request terminal
  events. Errors, rejections, slow requests, startup, shutdown, and state
  changes are never sampled.

## JSONL Event Contract

### Common fields

Every infer-nexus record must contain:

| Field | Type | Meaning |
|---|---|---|
| `timestamp` | string | UTC RFC 3339 timestamp |
| `level` | string | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |
| `event` | string | Stable dotted event name |
| `message` | string, optional | Human-readable detail, not the query key |
| `logger` | string | Python module logger name |
| `service` | string | `infer-nexus` |
| `service_version` | string | Running build/version identifier |
| `environment` | string | Deployment environment |
| `process_role` | string | `gateway`, `model_replica`, `serve_deployer`, or CLI role |
| `pid` | integer | Process ID |
| `source` | string | `infer_nexus` for application records |

Request-scoped fields are added only when available:

| Field | Type | Meaning |
|---|---|---|
| `request_id` | string | Canonical request correlation ID |
| `trace_id` | string | Future tracing identifier; never reuse request ID as trace ID |
| `method` | string | HTTP method |
| `route` | string | Route template, not raw URL |
| `model` | string | Resolved public model name |
| `task` | string | `chat`, `embedding`, or `rerank` |
| `stream` | boolean | Whether the response is streaming |
| `backend` | string | Backend type |
| `compat_mode` | string | Backend compatibility mode |
| `serve_app` | string | Ray Serve application name |
| `deployment` | string | Ray Serve deployment name |
| `replica_id` | string | Ray Serve replica identity |
| `outcome` | string | `success`, `error`, `rejected`, `cancelled`, or `timeout` |
| `status_code` | integer | HTTP status when applicable |
| `error_code` | string | Stable domain error code |
| `duration_ms` | number | Completed lifecycle duration |

Example:

```json
{"timestamp":"2026-09-15T02:32:18.421Z","level":"INFO","event":"request.completed","logger":"infer_nexus.api.request_lifecycle","service":"infer-nexus","service_version":"0.1.0","environment":"production","process_role":"gateway","pid":421,"source":"infer_nexus","request_id":"req-8f42c1","method":"POST","route":"/v1/chat/completions","model":"qwen3.5-9b","task":"chat","stream":false,"outcome":"success","status_code":200,"duration_ms":843.6}
```

### Sensitive data policy

Never log these values by default:

- `Authorization` and API key headers
- prompt or message content
- embedding input
- rerank query or documents
- token IDs
- cookies
- full upstream URLs containing credentials or query parameters
- arbitrary request or response bodies

Allowlisted metadata such as input/output token counts, payload byte size, model
name, and upstream host may be logged. The formatter must also redact reserved
sensitive keys if a caller accidentally provides them.

## Event Catalogue and Ownership

Initial stable events:

| Event | Owner | Default level | Required event-specific fields |
|---|---|---:|---|
| `app.starting` | process bootstrap | INFO | `process_role`, settings identity |
| `app.ready` | process bootstrap | INFO | `process_role`, readiness summary |
| `app.shutdown` | process bootstrap | INFO | `process_role` |
| `model.replica.initializing` | model replica | INFO | model/deployment/replica |
| `model.replica.ready` | model replica | INFO | model/deployment/replica, duration |
| `model.replica.failed` | model replica | ERROR | model/deployment/replica, error code |
| `request.completed` | Gateway lifecycle | INFO | request fields, outcome, status, duration |
| `request.failed` | Gateway lifecycle | ERROR | request fields, error code, duration |
| `admission.rejected` | admission owner | WARNING | model, reason, retry hint |
| `runtime.call.failed` | runtime owner | ERROR | app, deployment, method, error code |
| `stream.completed` | stream lifecycle | INFO | model, outcome, duration |
| `stream.failed` | stream lifecycle | ERROR | model, error code, duration |
| `stream.cancelled` | stream lifecycle | INFO | model, duration |
| `proxy.request.failed` | proxy executor | WARNING/ERROR | upstream host/status, error code |
| `backend.adapter.fallback` | backend | WARNING | model, engine kind, reason |

Ownership rules:

1. The Gateway emits exactly one terminal request event.
2. The stream lifecycle emits exactly one terminal stream event after the
   iterator completes, fails, or is cancelled.
3. A layer that only translates and rethrows an exception does not log it.
4. Expected validation, admission, overload, and timeout failures do not carry
   stack traces by default.
5. Unexpected terminal failures carry one stack trace at the layer that owns
   the terminal outcome.
6. Background lifecycle failures with no request owner are logged where their
   state transition is owned.
7. Do not log per token or per SSE chunk at INFO.

## Request Context Design

Add an outer ASGI request lifecycle middleware before worker admission.

Request ID resolution order:

1. Accept a client `X-Request-ID` after length and character validation.
2. Under Ray Serve, ask an isolated Ray request-identity adapter for the Serve
   request ID.
3. Generate a UUID when neither source provides an ID.

Behavior:

- Return canonical `X-Request-ID` on every response.
- During one compatibility window, streaming responses may also return the old
  `X-Infer-Nexus-Request-ID` header with the same value.
- Bind request metadata before admission so overload rejections are correlated.
- Resolve and bind model/task fields when catalog lookup succeeds.
- Keep context active until a streaming body actually terminates.
- Always reset context tokens in `finally` blocks to prevent cross-request
  leakage.

Cross-process propagation:

- Add a small serializable `RequestContext` value containing allowlisted fields.
- Pass it beside `request_payload` in Ray deployment-handle calls; never insert
  internal metadata into the OpenAI-compatible payload.
- Bind it at the beginning of every `ModelRuntimeReplica` method and reset it
  after completion.
- Forward `X-Request-ID` to proxy upstreams.
- Reserve `trace_id` and `traceparent` for a later tracing change.

## Framework Adapters

### Uvicorn

- Bootstrap logging in each worker process before serving requests.
- In deployed mode, disable or raise the level of `uvicorn.access` after the
  semantic request event is verified.
- In local console mode, access logs may remain enabled.

### Ray Core and Ray Serve

- Configure the deployment driver before importing/initializing Ray where the
  selected Ray option requires that ordering.
- Pass a validated Serve `logging_config` to `serve.start()` for controller and
  proxy logs.
- Pass a deployment-level `logging_config` through Gateway and model deployment
  kwargs.
- Prefer JSON encoding where supported by the pinned Ray version.
- Keep proxy access logs as the transport-level edge record.
- Disable duplicate Gateway-replica and model-replica access logs by default;
  make them opt-in for diagnosis.
- Obtain application/deployment/replica identity through an isolated adapter so
  version-specific Ray calls do not leak into application callers.
- Add a compatibility test against the Ray version resolved by `uv.lock`.

### vLLM

- Configure vLLM logging before importing or constructing the engine where the
  pinned version requires it.
- Map the infer-nexus named level to the supported vLLM level setting.
- Use an explicit vLLM logging config when necessary because the vLLM logger
  may set `propagate: false`.
- Keep vLLM records under `source=vllm`; do not promise that third-party fields
  exactly match the infer-nexus event schema.
- Do not enable request-body or prompt logging.
- Add version-sensitive tests for the CUDA and Ascend dependency sets.

## Collection Design by Environment

### Local bootstrap scripts

- Set Ray's temp root explicitly below `.infer-nexus/ray/` so the real session
  logs are discoverable after startup.
- Rename `.infer-nexus/logs/ray.log` to `ray_bootstrap.log` because it contains
  CLI startup output rather than the full Ray log set.
- Keep `serve_runtime.log` for the one-shot deployment driver.
- Print the actual Ray `session_latest/logs` path on successful startup.
- Do not combine output from multiple nodes into one writable directory.

### Docker Compose

- Replace the unused `/app/.infer-nexus/logs` mount with separate `/tmp/ray`
  mounts for `ray-head` and `ray-worker`.
- Use distinct host paths or named volumes per service to avoid session and
  filename collisions.
- Add an optional collector profile only after local file collection is
  verified. The initial collector may stream parsed records to stdout; Loki or
  another store remains an operational choice.
- Retain Docker log rotation settings for bootstrap/container logs separately
  from Ray file rotation.

### KubeRay

- Add an `emptyDir` volume mounted at `/tmp/ray` in each Ray Pod.
- Share that volume with a Fluent Bit or Vector sidecar, or expose it to a
  node-level daemonset.
- Add Kubernetes and Ray metadata at collection time.
- Export to the selected central store with retention outside infer-nexus.
- Do not make `RAY_LOG_TO_STDERR=1` the default because it disables Ray log
  files and can affect features that depend on them.

## Implementation Sequence

Each phase should be independently reviewable. Prefer one small commit per
phase unless a phase has a natural test-first split.

### Phase 1: Configuration and event contract

Likely files:

- `src/infer_nexus/core/config.py`
- `config/settings.yaml`
- `config/settings.compose.yaml`
- `config/settings.ascend-compose.yaml`
- `tests/test_config.py`

Tasks:

- Add typed `ObservabilitySettings` and `LoggingSettings`.
- Validate format, sample rate, slow threshold, and logger levels.
- Add temporary `service.log_level` compatibility behavior.
- Add the stable field and event-name constants only if constants materially
  prevent caller drift; do not create one class per event.

Acceptance:

- Invalid log levels and formats fail configuration loading.
- All three settings files select the intended profile.
- Existing settings remain loadable during the migration window.

Suggested commit: `feat(logging): define observability configuration contract`

### Phase 2: Deep logging module

Likely files:

- `src/infer_nexus/observability/logging.py`, or a private package behind its
  existing import path
- `src/infer_nexus/observability/__init__.py`
- new `tests/test_logging.py`
- `pyproject.toml`
- `pyproject.ascend.toml` when the dependency is required there

Tasks:

- Implement `configure_logging`, `get_logger`, and `bind_log_context`.
- Add JSONL and console formatters.
- Add process identity and context injection.
- Add redaction and JSON serialization fallbacks.
- Make repeated bootstrap deterministic and safe in tests.

Acceptance:

- Every emitted JSON line parses independently.
- Context is attached and reset correctly under concurrent async tasks.
- Sensitive allowlist/denylist tests pass.
- Exception records remain one JSON line.
- Tests use the module interface and do not assert private formatter internals.

Suggested commit: `feat(logging): add structured logging deep module`

### Phase 3: Request lifecycle and canonical identity

Likely files:

- new `src/infer_nexus/api/request_logging_middleware.py`
- `src/infer_nexus/main.py`
- `src/infer_nexus/api/openai_routes.py`
- `src/infer_nexus/api/worker_admission_middleware.py`
- `tests/test_api.py`
- `tests/test_main.py`

Tasks:

- Add outer request lifecycle middleware.
- Preserve or generate the canonical request ID.
- Return the request ID header on success and error responses.
- Record terminal outcome after streaming completion.
- Move route-observation context so logs and metrics use one lifecycle state.
- Suppress successful health/readiness/metrics events at INFO.

Acceptance:

- Unary success/error, validation failure, overload, streaming success,
  streaming error, and cancellation each produce exactly one request terminal
  event.
- Streaming failure cannot be recorded as metric success.
- Concurrent requests do not exchange context.

Suggested commit: `feat(logging): add request lifecycle correlation`

### Phase 4: Cross-process context propagation

Likely files:

- `src/infer_nexus/runtime/executor.py`
- `src/infer_nexus/runtime/deployments.py`
- `src/infer_nexus/runtime/worker_client.py`
- `tests/test_dispatcher.py`
- `tests/test_runtime.py`
- `tests/test_proxy_streaming.py`

Tasks:

- Define the serializable request context.
- Pass it beside request payloads through unary and streaming handles.
- Bind/reset it in model replica methods.
- Forward request ID to proxy upstreams.
- Ensure future runtime-worker isolation uses the same context shape.

Acceptance:

- A fake handle test observes the exact context passed by the Gateway.
- A model replica log contains the same request ID as the Gateway log.
- Proxy tests verify request-ID forwarding without forwarding secrets.

Suggested commit: `feat(logging): propagate request context across runtime seams`

### Phase 5: Migrate application events

Likely files:

- `src/infer_nexus/api/openai_routes.py`
- `src/infer_nexus/runtime/executor.py`
- `src/infer_nexus/backends/vllm.py`
- `src/infer_nexus/backends/vllm_strict.py`
- control/admission modules where state owners need events

Tasks:

- Replace message-only logger calls with stable events and fields.
- Remove duplicate exception logging from translating layers.
- Treat adapter fallback as an explicit backend state transition.
- Add missing timeout, circuit-open, overload, and stream-cancel events.
- Keep payload and token/chunk content out of logs.

Acceptance:

- One injected backend failure yields one authoritative stack trace.
- Expected domain errors contain stable `error_code` without redundant traces.
- No first-party application logger bypasses `get_logger`.

Suggested commit: `refactor(logging): establish event ownership`

### Phase 6: Ray Serve and vLLM adapters

Likely files:

- `scripts/run_gateway.py`
- `scripts/run_serve_runtime.py`
- `src/infer_nexus/runtime/deployments.py`
- `src/infer_nexus/runtime/gateway_ingress.py`
- logging adapter implementation
- `tests/test_scripts.py`
- `tests/test_gateway_ingress.py`
- `tests/test_serve_gateway_poc.py`

Tasks:

- Bootstrap the driver and every worker/replica role.
- Add Serve global and deployment logging config.
- Add Ray and Serve identity fields where supported.
- Configure vLLM before engine initialization.
- Establish access-log ownership.

Acceptance:

- Driver, Gateway replica, and model replica each emit an identifiable startup
  event.
- The pinned Ray integration test validates JSON mode and request correlation.
- The selected vLLM versions do not install a conflicting duplicate handler.

Suggested commit: `feat(logging): integrate Ray Serve and vLLM logging`

### Phase 7: Collection paths

Likely files:

- `scripts/start_minimal.sh`
- `scripts/start_minimal_ascend.sh`
- `docker-compose.yml`
- `ascend_deploy/docker-compose.yml`
- `deploy/kuberay/raycluster.yaml`
- optional collector configuration under `monitoring/`

Tasks:

- Make the Ray session log path explicit locally.
- Separate bootstrap logs from Ray runtime logs.
- Give each Compose Ray node its own `/tmp/ray` storage.
- Add the KubeRay shared volume and collector pattern.
- Configure bounded rotation and external retention.

Acceptance:

- The operator can locate controller, proxy, Gateway, model replica, and vLLM
  records from documented paths.
- Restarting a Pod does not remove already-exported centralized records.
- No two Ray nodes write into the same session directory.

Suggested commit: `ops(logging): connect Ray logs to deployment collectors`

### Phase 8: Documentation and operational verification

Likely files:

- `docs/ARCHITECTURE.md`
- `docs/RAY_SERVE_VLLM_MONITORING.md`
- `.harness/context/current-architecture.md`
- `.harness/context/request-flows.md`
- `.harness/context/monitoring-benchmark.md`
- `.harness/rules/interface-rules.md` if the approved contract changes
- `.harness/checklists/verification.md`

Tasks:

- Record the implemented architecture, not the planned one.
- Add request-ID and event-name query examples.
- Document local, Compose, and KubeRay discovery paths.
- Add incident-oriented examples for overload, timeout, model startup failure,
  and stream cancellation.

Acceptance:

- A new operator can trace one request without reading source code.
- Context files clearly distinguish implemented behavior from remaining work.

Suggested commit: `docs(logging): document structured logging operations`

## Verification Matrix

| Scenario | Expected terminal event | Expected outcome | Stack trace |
|---|---|---|---|
| Unary success | `request.completed` | `success` | no |
| Unknown model | `request.completed` | `error` | no |
| Admission rejection | `request.completed` + owned `admission.rejected` | `rejected` | no |
| Serve timeout | `request.completed` | `timeout` | no by default |
| Unexpected backend crash | `request.failed` | `error` | exactly one |
| Stream success | `stream.completed`, then request terminal | `success` | no |
| Stream backend error | `stream.failed`, then request terminal | `error` | exactly one |
| Client disconnect | `stream.cancelled`, then request terminal | `cancelled` | no |
| Proxy upstream 4xx/5xx | request terminal with upstream status | `error` | no |
| Replica startup failure | `model.replica.failed` | `error` | exactly one |

Automated checks:

```bash
uv run --frozen pytest -q tests/test_logging.py tests/test_config.py
uv run --frozen pytest -q tests/test_api.py tests/test_main.py
uv run --frozen pytest -q tests/test_dispatcher.py tests/test_runtime.py tests/test_proxy_streaming.py
python3 -m compileall src tests
git diff --check
```

Run `tests/test_serve_gateway_poc.py` with
`INFER_NEXUS_RUN_RAY_INTEGRATION=1` against the pinned Ray environment before
claiming Serve correlation support. Use the repository remote validation
workflow before claiming CUDA or Ascend runtime completion.

Performance verification:

- Capture a baseline before enabling request JSONL.
- Compare Gateway CPU, p50/p95 latency, and log volume at the same workload.
- Agree on an explicit overhead budget before production rollout; an initial
  candidate is no more than 1 ms added Gateway p95 latency for non-streaming
  request logging.
- Verify success-event sampling does not affect errors, rejections, or slow
  requests.

## Definition of Done

- [ ] Deployed infer-nexus application logs are valid JSONL.
- [ ] Local development has an explicit human-readable console profile.
- [ ] The logging module exposes only the approved small interface.
- [ ] Configuration validation and compatibility migration are tested.
- [ ] A canonical request ID is returned and propagated across active runtime
      paths.
- [ ] Unary and streaming requests each have exactly one terminal request event.
- [ ] Error ownership prevents duplicate stack traces.
- [ ] Logs and metrics agree on request/stream outcomes.
- [ ] Sensitive data tests cover all prohibited payload categories.
- [ ] Ray Serve and vLLM adapters are validated against pinned versions.
- [ ] Local, Compose, and KubeRay log collection paths are documented and tested.
- [ ] Architecture and harness context are updated only after behavior exists.

## Rollback Strategy

- Keep JSON versus console formatting configuration-driven.
- Preserve the old `service.log_level` read during one migration window.
- Land request context propagation before removing old stream response headers.
- Keep framework adapter changes separable from application event migration.
- If JSON collection fails operationally, switch the formatter profile to
  console without reverting request identity, event ownership, or redaction.

## Primary References

- `docs/ARCHITECTURE.md`, especially the logging recommendations
- `.harness/rules/interface-rules.md`
- `.harness/context/current-architecture.md`
- `.harness/context/request-flows.md`
- `.harness/context/monitoring-benchmark.md`
- `src/infer_nexus/observability/logging.py`
- `src/infer_nexus/runtime/deployments.py`
- `src/infer_nexus/runtime/executor.py`
- Ray logging and Ray Serve logging configuration documentation for the pinned
  Ray version
- vLLM logger implementation/documentation for the pinned CUDA and Ascend
  dependency sets
