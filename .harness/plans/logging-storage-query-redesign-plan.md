# Logging Storage And Query Redesign Execution Plan

Status: ready for implementation, not implemented

Prepared: 2026-09-19; diagnostic coverage refined 2026-09-20
Scope: single-host CUDA and Ascend Docker Compose deployments

## Purpose

Complete the second stage of the infer-nexus logging work without rebuilding
the structured-event and request-correlation features that already exist.

The target is a service-first host layout with three deliberately separate log
sources:

```text
${LOGS_HOST_PATH}/
  ray-head/
    events-<role>-<process-instance>.jsonl
    ray/session_*/logs/...
  ray-worker/
    events-<role>-<process-instance>.jsonl
    ray/session_*/logs/...
  serve-deployer/
    events-<role>-<process-instance>.jsonl
    ray/session_*/logs/...
```

Container command stdout/stderr must return to Docker's logging driver and be
viewed with `docker compose logs`. `container.log` and
`scripts/container/run_with_log.sh` are removed. Docker output is the quick
container/bootstrap view, `events-*.jsonl` is the authoritative infer-nexus
application-event source, and the Ray session tree is the authoritative raw
Ray/Serve/vLLM diagnostic source.

A new local query command combines these sources into a useful troubleshooting
view without changing or deleting the originals. KubeRay, ELK, Loki, Fluent
Bit, Filebeat, and OpenTelemetry Collector are future integrations, not current
runtime dependencies.

## User-Approved Decisions

The conversation leading to this plan approved these design changes:

1. Keep one host-visible directory per physical Compose service.
2. Put `events-*.jsonl` directly in that service directory; do not add an
   `app/` subdirectory.
3. Keep each service's complete `/tmp/ray` tree mounted under `ray/`.
4. Separate stable infer-nexus events from raw Ray/Serve/vLLM diagnostics.
5. Restore service command stdout/stderr to Docker and make
   `docker compose logs` useful for startup and recent container output.
6. Use a local CLI as the main single-host troubleshooting entry point.
7. Keep the schema and directory layout compatible with later ELK/Loki file
   collection, but do not deploy a collector or backend now.

These decisions intentionally replace the current repository requirement that
every service command append to `${LOGS_HOST_PATH}/<service>/container.log`.
The implementation must update `docs/ARCHITECTURE.md`, `AGENTS.md`, relevant
harness files, Compose files, tests, README, deployment documentation, and
operator workflows so the repository has one consistent contract.

## Read Before Editing

Read these sources in order and use current code as the implementation
baseline:

1. `docs/ARCHITECTURE.md`, logging and request-correlation section
2. `AGENTS.md`, especially the Compose logging guardrail
3. `.harness/context/current-architecture.md`
4. `.harness/context/monitoring-benchmark.md`
5. `.harness/rules/interface-rules.md`
6. `.harness/checklists/verification.md`
7. `docs/LOGGING_REDESIGN_PROPOSAL.md`
8. `docs/LOGGING_REAL_WORLD_PATTERNS_2026-09-19.md`
9. `src/infer_nexus/observability/logging.py`
10. both Compose files, `scripts/prepare_compose_logs.py`, and
    `tests/test_compose_logging.py`

The older `.harness/plans/logging-management-execution-plan.md` is historical.
Its structured logger, request ID, event ownership, redaction, framework
adapters, and existing Compose Ray mounts have already been substantially
implemented. Verify current source and do not repeat those phases.

## Non-Goals And Guardrails

- Do not introduce Kubernetes or KubeRay in the single-host implementation.
- Do not deploy ELK, Loki, Fluent Bit, Filebeat, Vector, or an OTel Collector.
- Do not send logs directly from request handlers to a remote service.
- Do not log prompts, messages, tokens, embeddings, rerank documents,
  credentials, request/response bodies, or upstream secrets.
- Do not put request ID, trace ID, PID, process instance, or error text in a
  future backend's low-cardinality index labels.
- Do not change request lifecycle ownership, stable event semantics, sampling,
  admission behavior, inference behavior, or model placement. Adding safe
  fields to an existing owner's event is allowed; do not create a second
  terminal event for the same request.
- Do not force Ray into stdout-only mode or set `RAY_LOG_TO_STDERR=1`.
- Do not recursively delete files from an active Ray session.
- Do not silently ignore malformed or unreadable log records.
- Do not make the first version of the query CLI depend on Docker daemon API
  access. Docker logs remain a separate quick-view command.
- Runtime containers remain non-root. Host directory preparation remains the
  host script's responsibility; services must not chown host mounts.

## Source Ownership Contract

| Category | Authoritative source | Normal access |
| --- | --- | --- |
| infer-nexus structured events | `events-*.jsonl` and its rotation files | `scripts/logs.py` |
| Ray Core, Serve, worker, actor, and vLLM diagnostics | actual `ray/session_*/logs/` files | `scripts/logs.py --infra` or raw files |
| container command/bootstrap output | Docker logging driver | `docker compose logs` |
| current container state and exit status | Docker metadata | `docker compose ps` / `docker inspect` |

Application events may also appear on stdout so `docker compose logs` remains
useful during an incident. This is an intentional display copy, not another
authoritative persistence source. A future collector must ingest application
events from `events-*.jsonl` only, or explicitly discard the infer-nexus copy
from Docker output.

Do not merge records merely because their text looks alike. Different requests
or replicas can produce valid similar failures. Remove duplicates by selecting
the authoritative source, not by fuzzy text matching.

## Target Event File Contract

### Path and identity

- Compose sets the same in-container event directory for all services:
  `/var/log/infer-nexus`.
- Use explicit environment names unless current configuration conventions
  provide an equivalent typed path: `INFER_NEXUS_EVENT_LOG_DIR` for the event
  directory and `INFER_NEXUS_PHYSICAL_SERVICE` for `ray-head`, `ray-worker`, or
  `serve-deployer`. Do not derive the physical service from `process_role`.
- Existing service-specific bind mounts map that path to the correct physical
  host directory.
- Compose provides a physical service identifier independently for each
  container, such as `ray-head`, `ray-worker`, or `serve-deployer`.
- Do not inject the deployer's physical service identifier through Ray
  `runtime_env.env_vars`; a remote replica must use the environment inherited
  from the Ray node that actually runs it.
- Each logging process generates one startup-unique `process_instance` and
  records it on every event.
- Use a sanitized filename shaped like
  `events-<process-role>-<process-instance>.jsonl`. PID may be a readable field
  or filename component but must not be the sole instance identity.
- Repeated `configure_logging()` calls in one process must not add duplicate
  handlers or create a new process instance accidentally.

### Records

Reuse the existing JSON formatter, redaction, context, exception serialization,
and structured logger. Add or verify these stable fields:

- `schema_version`
- UTC `timestamp`
- `level`
- `event`
- logical `service`
- physical Compose service/node identity
- `process_role`
- `process_instance`
- `pid`
- `source`
- applicable request/model/deployment/replica/outcome/error/duration fields

Do not rename the existing canonical `request_id` to `trace_id`. Future tracing
may add `trace_id` and `span_id` as separate fields.

Use `schema_version: 1` for the first event-file schema. In Compose,
`physical_service` must equal the service that actually hosts the writing
process; `process_instance` must remain stable for that process lifetime.
`timestamp` is UTC ISO-8601, `duration_ms` and other durations are
nonnegative numbers, and `error_code` remains a stable code rather than a
free-form exception message. Do not duplicate the same concept under two
new field names. Optional fields are omitted when unknown; use an explicit
`*_available` or `*_source` field only when consumers need to distinguish
"not measured" from "not applicable".

The allowed `failure_stage` vocabulary for this plan is:
`request_validation`, `catalog_lookup`, `gateway_admission`,
`model_admission`, `serve_handle`, `model_backend`, `proxy_upstream`,
`response_stream`, `unknown`. Add another value only with a documented
owner and test. `timeout_kind` uses a similarly bounded vocabulary drawn
from actual code branches (`admission_wait`, `serve_handle`, `stream_idle`,
`stream_lifetime`, `upstream_connect`, `upstream_read`, `unknown`). Omit both
fields for successful events. A `request.failed` record may report
`status_code: 200` if a stream failed after response headers were sent;
`outcome` and the stream terminal event carry the failure truth.

### Diagnostic coverage: logs versus metrics

The current application already emits request terminal, stream terminal,
admission rejection, circuit transition, replica startup, and process startup
events. Prometheus already has request counts/latency, inflight,
gateway-local admission wait/depth, Serve-handle calls/timeouts, stream TTFT
and chunk intervals, and token totals when response usage exists. The
`infer_nexus_request_queue_seconds` instrument is defined but currently has
no call site; use Ray Serve queue metrics for proxy/router waiting. The
`infer_nexus_stream_tpot_seconds` value is an SSE chunk-interval approximation,
not model token timing. Audit the `request_latency_seconds` help text against
the actual request-finalization boundary before documenting it. Do not
reimplement existing counters in this storage project. A log record explains
one incident; a metric measures a rate, distribution, or current pressure.
Never use `request_id` as a metric label. Reconcile both views instead of
logging every queue depth change or every generated token.

The event schema is **diagnostically sufficient** only if an operator can
answer who handled a request, what outcome it had, which stage failed or
waited, and where to inspect the next raw source. It is not a promise that
every framework/internal event has an infer-nexus JSONL equivalent. Review
each row below against current code before adding fields; `when available`
means omit the field or mark its measurement status explicitly, never invent
zero or estimate from unrelated timestamps.

| Boundary/question | Current evidence | Required improvement and owner | Priority |
| --- | --- | --- | --- |
| Which request, model, route, and process? | `request.completed` / `request.failed` contain `request_id`, method, route, model/task/backend when resolved, status, outcome, error code, total duration; replica events identify deployment/replica. | Preserve these in the event file; add physical service, process instance and schema version. Use route templates, never raw URL/query. | P0 |
| Why was it rejected or timed out? | `admission.rejected` has some layer/reason details; request terminal has stable error code; Serve-handle timeout metrics exist. | On the owning failure event, add bounded `failure_stage` and `timeout_kind` from the vocabulary above when known. Keep an explicit `admission_layer` and final status; do not duplicate traceback. | P0 |
| Which stage caused a slow request? | Total `duration_ms`; admission wait and Serve-handle latency are measured in metrics; stream TTFT is a metric but not in stream terminal fields. | Carry already measured `admission_wait_ms`, `serve_handle_ms`, and `ttft_ms` into the owning terminal event when reliable, especially for slow/failing requests. Document the timing boundary of each field; do not sum overlapping intervals or call first SSE chunk a true model token. | P1 |
| Did a stream complete, stall, or get cancelled? | `stream.completed` / `failed` / `cancelled` plus request terminal; metrics observe TTFT and chunk intervals. | Keep one stream terminal and one request terminal. Add `emitted_chunk_count` and stream duration when measured without per-chunk logging; record idle/lifetime timeout distinction and whether response headers were already sent. | P1 |
| Did a proxy retry or fail upstream? | `proxy.request.failed` records safe host, error code and optional upstream status; request terminal can include upstream status. | Add bounded attempt count, final upstream status and failure class (`connect`, `read_timeout`, `status`, `protocol`) at the proxy owner when available. Never log full URL, auth headers, upstream body, or exception text that may contain them. | P1 |
| Did the model become ready, and how long did startup take? | `model.replica.initializing/ready/failed` exist; ready has `duration_ms`, failed has stable error code. | Record safe `startup_stage` on failure (config/artifact/backend/engine/unknown) when the code can identify it. Keep model/deployment/replica/node identity and correlate initializing to ready/failed through process instance. | P0 |
| Which model/config build was running? | Model/deployment identity exists; the hard-coded service version alone does not identify a deployed catalog revision. | Add a stable, non-secret build/config fingerprint only if a deployment source can provide it consistently to Gateway and replicas. Never log model paths, raw catalog contents, tokens or credentials. | P1 |
| How many tokens did one request use? | Gateway counters increment only when response usage exists; terminal event does not carry usage counts or availability. | On a request terminal, optionally add `input_tokens`/`output_tokens` only from trustworthy response usage and `usage_source` or `usage_available`. Missing usage means unknown, especially for streaming; never log token contents. | P1 |
| Did readiness or capacity change after startup? | `app.ready` and replica startup events exist; `/readyz` checks current state, and Ray/Serve metrics expose deployment pressure. | Emit a state-transition event only when readiness changes, if a reliable owner can observe that change; use Ray/Serve metrics for continuously varying queue, replica and resource pressure. Do not log every health poll. | P1 |
| What happened before Gateway middleware? | Ray Serve proxy can reject a queued request before infer-nexus middleware, so no application terminal event or Gateway metric is produced. | Do not fabricate an application event. Document Ray proxy access logs and Serve queue/rejection metrics as the source; add a `--infra` runbook for this blind spot and verify it in a saturated-proxy test. | P0 |
| Did a replica or container die without flushing an event? | Ray actor/core files and Docker state can survive a process crash even when no application terminal event is written. | The `--infra` runbook must connect Ray actor/worker crash records with Docker restart/OOM/exit state using time, node and replica identity. Do not claim request-level certainty without a shared ID. | P0 |
| Is node/engine pressure causing failures? | Ray/Serve metrics expose some queue and replica state; embedded vLLM metrics, GPU memory and host disk pressure are not yet verified as one complete path. | Document actual scrape targets and gaps. Use Ray/engine/node/Docker metrics for changing pressure and resource exhaustion, with an incident link from `--infra`; do not emit per-request GPU/KV-cache snapshots in JSONL. | P1 |
| Is logging itself healthy? | No event-file writer exists yet; Docker/Ray raw sources remain separate. | Emit rate-limited stderr diagnostics on open/write/rotation failure and count failures per process. If async writing exists, count dropped records and queue saturation. `--stats` reports file coverage and bytes but must not claim that absent records prove health. | P0 |

P0 is required for this redesign. P1 fields are valuable for performance and
incident analysis but should be implemented only at an existing measurement
boundary and only when they do not alter inference behavior. If a P1 field
needs a new expensive timer, backend modification, or cross-process protocol,
record the gap in the final report and create follow-up work rather than
inventing data. Keep successful-request source sampling semantics visible:
if `success_sample_rate < 1`, event files are not a complete request audit.
Errors, rejections, slow requests, and lifecycle failures remain unsampled
under the existing contract.

For metrics, the first required addition is event-writer health (failure
counter, plus dropped counter only if a bounded async queue is used). Do not
expose a Gateway-only counter as if it covered model replicas on other Ray
nodes. Use a Ray-compatible per-process metric path if verified; otherwise
report per-process stderr diagnostics and treat cross-node metric aggregation
as follow-up. Additional engine internals such as vLLM queue depth, KV-cache
pressure and GPU memory belong to engine/Ray metrics when actually exported,
not fabricated application event fields. This repository has not verified
embedded vLLM metric exposure; document the observability gap explicitly.

### Handler behavior

- The event-file handler accepts only infer-nexus structured event records.
  Test the marker/namespace rule explicitly.
- It must not copy arbitrary Ray, Serve, vLLM, Uvicorn, or third-party text into
  `events-*.jsonl`.
- The existing stdout handler remains available so Docker output includes
  immediate process information.
- One process is the sole writer of each event file. Use a process-owned
  rotating handler; never let multiple processes share one ordinary rotating
  file.
- If a child process could inherit an open handler, detect the PID change,
  close the inherited handler, and initialize a new file before writing.
- A file open/write failure must be visible on stderr with rate limiting. It
  must not recursively log through the broken handler or block inference
  indefinitely.
- If asynchronous writing is introduced, the queue must be bounded and expose
  dropped/write-failure counts. Synchronous writing is acceptable for the first
  version if the runtime benchmark shows acceptable overhead.

### Rotation and retention

- Rotation settings belong in `LoggingSettings`, with validated positive
  values and explicit deployed defaults.
- Select initial byte and backup values only after recording a representative
  startup, successful request, request failure, and idle-period sample. Record
  the measured data and selected values in the implementation PR/commit notes.
- Event writer rotation must use rename/create semantics controlled by the sole
  writer. Do not apply a second host rotator to the same files.
- Per-file rotation does not bound files left by exited processes. The first
  release must at least report total bytes, file counts, and oldest/newest
  timestamps. Automatic deletion of exited-process files and historical Ray
  sessions is a separate operation and must default to dry-run until an active
  session safety rule is verified.

## Query Command Contract

Create a thin wrapper at `scripts/logs.py`. Put discover/parse/filter/render
logic in a testable private module under `src/infer_nexus/observability/`; do
not put the whole implementation in the script.

Minimum supported interface:

```bash
python scripts/logs.py --since 30m
python scripts/logs.py --follow
python scripts/logs.py --request-id REQUEST_ID
python scripts/logs.py --model MODEL --since 1h
python scripts/logs.py --service ray-head --since 30m
python scripts/logs.py --infra --service ray-worker --since 30m
python scripts/logs.py --raw --service ray-worker
python scripts/logs.py --stats
```

Requirements:

- Read the default log root from `.env` using the same semantics as the Compose
  preparation flow; allow an explicit `--logs-dir` override.
- Validate the root and service names. Never accept a service/path that escapes
  the log root.
- Default mode reads `events-*.jsonl` and rotation files.
- Default presentation shows important lifecycle events, WARN/ERROR, failure,
  rejection, timeout, cancellation, and slow-request outcomes. Expected
  failures can be INFO `request.completed`, so filtering cannot use severity
  alone.
- An explicit request ID query shows all retained matching application events,
  including ordinary successful events hidden by the default summary.
- `--infra` adds known Ray/Serve/vLLM log files from actual session paths.
- `--raw` preserves original text and source path when parsing is unsupported.
- Never ingest both `session_latest` and its real `session_*` target. Resolve it
  once for current-session mode. A future `--all-sessions` may enumerate real
  session directories while excluding the symlink.
- Include source path and line/record location in detailed output.
- JSON parse failures, permission failures, partial lines, and unsupported
  formats must be reported. They must not turn into an empty-success result.
- Historical output may sort by parsed timestamp but must state that the order
  is not a distributed causal order. Records without timestamps retain file
  order and cannot be assigned invented timestamps.
- `--follow` must notice newly created process files and rotations. Do not use
  a tight polling loop; make the interval bounded/configurable.
- `--stats` reports bytes and file counts separately for events, Ray logs,
  other Ray session contents, and historical sessions. It must not count the
  `session_latest` alias twice.
- Keep the first output renderer text-oriented. A JSON output flag is useful
  for automation but does not justify a renderer plugin framework.

## Implementation Phases

Each phase must leave the repository reviewable. Do not combine all work into
one large commit. Update the architecture and operator documentation affected
by a phase in that phase's commit, especially when Phase 1 changes the
`container.log` guardrail. Phase 6 is a final consistency audit, not permission
to leave stale instructions in earlier commits.

### Phase 0: Reconfirm baseline and capture samples

Actions:

- Run the narrow existing logging/config/Compose tests and record results.
- Confirm both Compose files still use `run_with_log.sh` and that Docker output
  is empty or nearly empty after redirection.
- On the available real server, capture file counts and byte growth for one
  startup, idle period, successful request, request failure, and model startup
  failure. Do not copy sensitive payloads into the repository.
- Audit the observed metric series, not just metric declarations. Record that
  `request_queue_seconds` has no current call site, that `stream_tpot_seconds`
  measures SSE chunk intervals, and whether Ray/embedded-vLLM/node metrics are
  actually exposed in this deployment.
- Identify which records are duplicated by Ray worker-to-driver forwarding.
- Inventory representative event records and fill a coverage worksheet with:
  event owner, existing fields, missing fields, metric/raw fallback, source
  sampling setting, and whether a reliable measurement boundary exists.
  Classify each diagnostic coverage row as implemented, P0 work, P1 feasible,
  or P1 deferred. Keep this worksheet in the implementation report or PR body,
  not in a new runtime configuration file.

No production behavior changes in this phase.

### Phase 1: Restore Docker stdout/stderr

Likely files:

- `docker-compose.yml`
- `ascend_deploy/docker-compose.yml`
- `tests/test_compose_logging.py`
- `scripts/container/run_with_log.sh` (remove after all references are gone)
- `scripts/prepare_compose_logs.py`

Actions:

- Remove the wrapper invocation from all three services in both Compose files.
- Preserve direct `exec` semantics for `ray start` and nested service scripts so
  PID 1, signals, and exit codes remain correct.
- Retain the existing Docker logging driver and `10m` × 5 rotation initially;
  do not mix a driver migration into this phase.
- Remove tests for `container.log` and the wrapper. Replace them with tests that
  prove commands no longer redirect stdout/stderr, still have valid shell
  syntax, keep per-service `/var/log/infer-nexus` and `/tmp/ray` mounts, and
  retain bounded Docker rotation.
- Keep service directory preparation for event and Ray files. Replace any
  preservation test that uses `container.log` with an unrelated sentinel file.
- Delete the wrapper only after `rg run_with_log` finds no runtime reference.
- In the same change, update `docs/ARCHITECTURE.md`, `AGENTS.md`, README, both
  Compose deployment instructions, and the affected harness context/checklist
  or workflow to describe Docker stdout/stderr. Keep the host-visible `ray/`
  mount and non-root preparation rule. Do not leave the old `container.log`
  guardrail active until Phase 6.

Acceptance:

- `docker compose config -q` succeeds for CUDA and Ascend profiles.
- A deliberately printed stdout and stderr line appears in
  `docker compose logs` during a controlled smoke test.
- SIGTERM reaches the service process and the observed exit code remains
  correct.
- No new `container.log` is created.
- Each service's Ray session still lands below its own host `ray/` directory.

Suggested commit: `refactor(logging): restore compose stdout and stderr`

### Phase 2A: Add process-isolated application event files

Likely files:

- `src/infer_nexus/core/config.py`
- `src/infer_nexus/observability/logging.py`
- possibly one private handler module under `observability/`
- `scripts/run_serve_runtime.py`
- both Compose files
- all three settings YAML files
- `tests/test_config.py`
- `tests/test_logging.py`
- `tests/test_compose_logging.py`

Actions:

- Add the minimal file-output configuration under
  `observability.logging`; local development defaults to no event file,
  deployed Compose profiles enable `/var/log/infer-nexus`.
- Add physical-service environment identity independently to every Compose
  service. Verify Ray replicas use their actual node environment.
- Add startup-unique process identity and the event-only rotating handler.
- Keep the existing public logging interface unless an unavoidable process
  lifecycle requirement proves a small addition is necessary.
- Keep stdout output. Establish event files as the application source used by
  the query CLI and future collectors.
- Make configuration idempotent and safe in tests and process reuse.
- Update the architecture, harness context, README and deployment docs with
  the new event path and the source-ownership rule in this change.

Acceptance:

- Two processes never write the same event filename.
- A Gateway or model replica event appears as one valid JSON object per line in
  the physical service directory where that process ran.
- The same `request_id` correlates Gateway and model events.
- Third-party free-form log records do not appear in event files.
- Repeated configuration does not duplicate event or stdout records.
- A simulated write/open failure is visible on stderr without recursion.
- Existing redaction, terminal-event ownership, streaming, and sampling tests
  remain green.

Suggested commit: `feat(logging): persist process-isolated application events`

### Phase 2B: Close mandatory diagnostic coverage gaps

Likely files:

- `src/infer_nexus/core/request_context.py`
- `src/infer_nexus/api/request_logging_middleware.py`
- `src/infer_nexus/api/worker_admission_middleware.py`
- `src/infer_nexus/runtime/executor.py`
- `src/infer_nexus/runtime/deployments.py`
- focused tests for request, stream, admission, proxy and replica events

Actions:

- Implement the P0 rows in the diagnostic coverage table at the existing event
  owners. On the request path, carry an explicit bounded `failure_stage` and
  `timeout_kind` through `RequestLifecycleState` only when an owner knows them;
  do not guess them from an HTTP status or exception message. Keep the
  existing `request.completed`/`request.failed` ownership and one traceback.
- At model initialization, set a safe `startup_stage` before each identifiable
  config/artifact/backend/engine operation. If the stage cannot be separated
  reliably, use `unknown` and explain the gap.
- Preserve the existing `admission.rejected`, circuit, stream and replica
  lifecycle event names. Reuse stable error codes; do not create an event for
  every internal call or per-token output.
- Add P1 fields only if Phase 0 identified a trustworthy existing measurement
  boundary. Otherwise list each deferred field and the missing source in the
  implementation report. Missing usage and timing must be absent or explicitly
  unknown, never zero-filled.
- Extend the operator runbook for requests rejected before Gateway and for
  process death without an application event. These conditions are owned by
  Ray/Docker raw sources, not by invented infer-nexus terminal events.

Acceptance:

- A gateway-local rejection identifies its admission layer and failure stage.
- A Serve-handle timeout and a stream timeout retain distinct timeout kinds
  where code already distinguishes them.
- A model startup failure identifies a safe stage or explicit `unknown`.
- One backend exception still produces only one authoritative traceback and
  one request terminal event.
- Raw URL/query, authorization, request/response body and exception text that
  may contain them do not enter the event file.
- A missing usage value is absent or explicitly unknown, never `0` tokens.
- Metrics and events agree on outcome; expected pre-Gateway failures remain
  outside the Gateway event/metric counts and have a documented Ray source.

Suggested commit: `feat(logging): classify diagnostic failure stages`

### Phase 3: Add the local query engine and CLI

Likely files:

- `src/infer_nexus/observability/log_query.py` or a small private package
- `scripts/logs.py`
- new focused query tests

Actions:

- Implement safe discovery, event parsing, filtering, timestamp handling, and
  concise rendering first.
- Add request/model/service/time filters.
- Make `failure_stage`, `error_code`, and `outcome` visible in the default
  summary; permit filtering by stable error code and failure stage without
  requiring users to inspect raw JSON.
- Add infrastructure/raw discovery without assuming every Ray file is JSON.
- Add follow mode only after finite-file queries and rotation fixtures work.
- Add stats mode without deletion.
- Update README, `docs/RAY_SERVE_VLLM_MONITORING.md`, the deployment checklist,
  harness context/checklist, and the remote deployment workflow with working
  CLI commands in the same change.

Acceptance:

- Fixtures covering three physical services produce one merged query result
  with correct source attribution.
- A request-ID query finds correlated events across head and worker files.
- `session_latest` plus its target is not double counted.
- New files and renamed rotations are discovered in follow mode.
- Partial JSON lines are retried in follow mode and reported in finite mode.
- Malformed data and permission errors yield visible diagnostics and a useful
  nonzero exit status when the requested query cannot be completed.
- Path traversal through `--service` or `--logs-dir` is rejected.

Suggested commit: `feat(logging): add unified local log queries`

### Phase 4: Complete capacity and failure controls

Actions:

- Use Phase 0 measurements to set and document event rotation defaults.
- Add a metric or another non-recursive observable counter for event writer
  failures per writing process; add a dropped counter if an async queue exists.
  Verify the chosen metric actually covers Ray replica processes before
  advertising it as a cluster-wide measure.
- Confirm disk-full/permission failures do not hang a synthetic request.
- Use `--stats` output to document a capacity calculation covering active event
  files, exited process files, Ray log files, non-log Ray session contents, and
  historical sessions.
- If implementing cleanup, require `--dry-run` by default, exclude the active
  session, operate only under the validated log root, and test symlink/path
  escape protection. It is acceptable to defer deletion and ship reporting
  first.
- Update the monitoring/deployment docs with the actual writer failure signal,
  rotation limits, `--stats` interpretation, and any deferred cleanup policy.

Acceptance:

- Rotation continues to be queryable without duplicate counting.
- Writer failure and any dropped records are visible.
- No retention command can delete active session files by default.
- The operator documentation distinguishes per-file rotation from total
  directory retention.

Suggested commit: `feat(logging): bound and report local log storage`

### Phase 5: Real runtime verification

Follow `.harness/workflows/remote-deploy-and-validate.md` for the available
accelerator server. Validate CUDA on A100 and retain the explicit Ascend live
validation gap unless an NPU environment is actually tested.

Required scenarios:

1. clean startup and `docker compose logs` visibility
2. one successful unary request
3. one expected validation or admission failure
4. successful stream and stream failure/cancellation
5. model initialization failure
6. actor or Core failure with no application event
7. container/process restart creating a new event file
8. event-file rotation while query/follow remains active
9. unwritable event directory or simulated writer failure
10. Serve proxy queue rejection before Gateway middleware (Ray access log and
    Serve metric present; no invented infer-nexus request terminal event)
11. proxy upstream timeout/retry and missing response usage, if those backends
    are available in the validation environment
12. container exit/OOM or Ray actor death where no application event can be
    emitted; confirm the raw/Docker fallback and state its correlation limit

For each scenario record the command, expected source, actual source, request
ID when applicable, and whether a traceback is owned exactly once. Compare
Gateway CPU, p50/p95 latency, request throughput, and log bytes with the Phase 0
baseline. Do not claim runtime completion from `docker compose config` or unit
tests alone.

Suggested commit: none unless runtime findings require fixes.

### Phase 6: Audit architecture, harness, Compose, and deployment docs

This phase is required as a cross-file audit. Each prior phase must already
have updated the documents affected by its implementation. Do not declare
completion while current instructions still tell operators to read
`container.log` or treat a planned P1 field as already available.

Update at least:

- `docs/ARCHITECTURE.md`
  - replace the `container.log` contract with Docker stdout/stderr
  - document event-file and Ray-source ownership
- `AGENTS.md`
  - replace the always-on Compose logging guardrail
  - retain non-root directory preparation and host visibility requirements
- `README.md`
  - update the service directory tree and day-one commands
- `docs/RAY_SERVE_VLLM_MONITORING.md`
  - add `docker compose logs`, `scripts/logs.py`, request, infra, and raw
    troubleshooting examples
  - document which log/metric owns each request, rejection, stream, startup,
    and pre-Gateway failure signal; explain optional timings and usage fields
    and embedded-vLLM metric visibility limits
  - correct metric descriptions where a declared series has no live producer
    or its timing boundary differs from the help text
- `docs/DEPLOYMENT_CHECKLIST.md`
  - add preflight directory preparation and post-start log verification
- relevant CUDA and Ascend deployment documentation
- `.harness/context/current-architecture.md`
  - describe only verified implemented behavior
- `.harness/context/monitoring-benchmark.md`
  - update paths, retention, query, actual metric producers and performance
    baseline; keep unimplemented P1 measurements visibly unimplemented
- `.harness/context/project-map.md`
  - add the query module and script
- `.harness/context/docs-map.md`
  - classify the research/design reports appropriately
- `.harness/checklists/verification.md`
  - remove `container.log` checks and add Docker/event/query/diagnostic
    coverage checks
- `.harness/workflows/remote-deploy-and-validate.md`
  - replace tail commands with Docker and unified-query commands
- `.harness/README.md`
  - mark this plan implemented only after all acceptance criteria pass
- the old `.harness/plans/logging-management-execution-plan.md`
  - mark its implemented portions and point remaining storage/query work here;
    do not leave its stale baseline presented as current

The research documents remain design evidence, not current-behavior manuals:

- `docs/LOGGING_MANAGEMENT_RESEARCH_2026-09-18.md`
- `docs/LOGGING_REAL_WORLD_PATTERNS_2026-09-19.md`
- `docs/LOGGING_REDESIGN_PROPOSAL.md`

Suggested commit: `docs(logging): document compose log source ownership`

## Verification Commands

Start narrow and expand only after relevant failures are resolved:

```bash
uv run --frozen pytest -q tests/test_logging.py tests/test_config.py
uv run --frozen pytest -q tests/test_compose_logging.py
uv run --frozen pytest -q tests/test_api.py tests/test_main.py
uv run --frozen pytest -q tests/test_runtime.py tests/test_proxy_streaming.py
python3 -m compileall src scripts tests
docker compose --env-file .env config -q
docker compose --env-file .env -f ascend_deploy/docker-compose.yml config -q
git diff --check
git status --short
```

Run the query CLI against deterministic temporary fixtures in automated tests.
Do not make the ordinary unit suite depend on Docker, Ray, a GPU, external
storage, or a network service.

## Future ELK Compatibility Checklist

No ELK code is required now. Preserve these contracts:

- Application events remain newline-delimited JSON.
- Add `schema_version`; map fields later through an ingest pipeline rather than
  prematurely renaming the application schema.
- A future Filebeat/Elastic Agent `filestream` input can glob
  `events-*.jsonl*` and use content fingerprint identity.
- A separate input/pipeline handles Ray raw files.
- Never collect both `session_latest` and its actual target.
- Use bounded source fields such as environment, service, and source as index
  dimensions. Keep request ID, trace ID, PID, replica ID, process instance, and
  error text as document fields.
- If Docker output is also collected, exclude the duplicate infer-nexus event
  copy or do not ingest that stream as the application dataset.
- Keep future datasets distinct, for example `infer_nexus.events`,
  `infer_nexus.container`, `ray.serve`, and `ray.core`.
- Collector offset state, retry queues, backend retention, and collector health
  alerts must be designed together when centralized collection is introduced.

## Definition Of Done

- [ ] Both Compose profiles send service command stdout/stderr to Docker.
- [ ] `docker compose logs` shows useful startup and failure output.
- [ ] `container.log` and `run_with_log.sh` have no active runtime, operator
      manual, or current-harness references. Historical research may retain
      them when clearly labeled as the previous design.
- [ ] Every infer-nexus process writes a unique, rotated `events-*.jsonl` file
      under its physical service directory in Compose.
- [ ] Event files contain only structured infer-nexus events and preserve all
      existing redaction and correlation guarantees.
- [ ] The P0 diagnostic coverage rows are validated, including bounded failure
      stages, startup failure stages when known, and the pre-Gateway Ray
      proxy blind spot. P1 additions or explicit deferrals are reported.
- [ ] Phase 0 coverage worksheet names the owner and trustworthy source for
      every field; final examples show a success, rejection, stream failure,
      model startup failure, and a pre-Gateway failure with no app event.
- [ ] Metrics and event fields retain their distinct meanings; no per-request
      identifiers enter Prometheus labels, and missing usage/timing data is
      never represented as zero.
- [ ] Ray's complete per-service `/tmp/ray` tree remains host-visible.
- [ ] The local query CLI supports default, request, model, service, infra, raw,
      follow, and stats workflows.
- [ ] Rotation, new files, malformed records, symlink aliases, restarts, and
      write failures have meaningful coverage.
- [ ] Source ownership prevents duplicate counting in the CLI and documents how
      a future collector must do the same.
- [ ] Capacity and runtime overhead are measured on a real deployment.
- [ ] `docs/ARCHITECTURE.md`, `AGENTS.md`, README, monitoring/deployment docs,
      Compose tests, harness contexts, harness checklist, and remote deployment
      workflow all describe the implemented design.
- [ ] CUDA runtime validation is reported accurately; Ascend is not claimed
      without a real NPU run.
- [ ] No collector, centralized backend, or KubeRay dependency was added.

## Stop Conditions

Stop and report the evidence instead of improvising if:

- the pinned Ray runtime does not preserve the physical node environment in a
  replica process, because event placement identity then needs a different
  injection seam;
- direct Compose `exec` changes shutdown or exit-code behavior;
- an event handler cannot be isolated from third-party records without changing
  the existing public logging interface materially;
- live measurements show unacceptable inference latency or disk growth;
- safe active-session detection cannot be proven for automatic cleanup.

These conditions block only the affected phase. Preserve completed,
independently correct phases and document the remaining decision.
