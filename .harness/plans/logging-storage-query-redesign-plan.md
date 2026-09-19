# Logging Storage And Query Redesign Execution Plan

Status: ready for implementation, not implemented

Prepared: 2026-09-19
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
  admission behavior, inference behavior, or model placement.
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
one large commit.

### Phase 0: Reconfirm baseline and capture samples

Actions:

- Run the narrow existing logging/config/Compose tests and record results.
- Confirm both Compose files still use `run_with_log.sh` and that Docker output
  is empty or nearly empty after redirection.
- On the available real server, capture file counts and byte growth for one
  startup, idle period, successful request, request failure, and model startup
  failure. Do not copy sensitive payloads into the repository.
- Identify which records are duplicated by Ray worker-to-driver forwarding.

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

Acceptance:

- `docker compose config -q` succeeds for CUDA and Ascend profiles.
- A deliberately printed stdout and stderr line appears in
  `docker compose logs` during a controlled smoke test.
- SIGTERM reaches the service process and the observed exit code remains
  correct.
- No new `container.log` is created.
- Each service's Ray session still lands below its own host `ray/` directory.

Suggested commit: `refactor(logging): restore compose stdout and stderr`

### Phase 2: Add process-isolated application event files

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

### Phase 3: Add the local query engine and CLI

Likely files:

- `src/infer_nexus/observability/log_query.py` or a small private package
- `scripts/logs.py`
- new focused query tests

Actions:

- Implement safe discovery, event parsing, filtering, timestamp handling, and
  concise rendering first.
- Add request/model/service/time filters.
- Add infrastructure/raw discovery without assuming every Ray file is JSON.
- Add follow mode only after finite-file queries and rotation fixtures work.
- Add stats mode without deletion.

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
  failures; add a dropped counter if an async queue exists.
- Confirm disk-full/permission failures do not hang a synthetic request.
- Use `--stats` output to document a capacity calculation covering active event
  files, exited process files, Ray log files, non-log Ray session contents, and
  historical sessions.
- If implementing cleanup, require `--dry-run` by default, exclude the active
  session, operate only under the validated log root, and test symlink/path
  escape protection. It is acceptable to defer deletion and ship reporting
  first.

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

For each scenario record the command, expected source, actual source, request
ID when applicable, and whether a traceback is owned exactly once. Compare
Gateway CPU, p50/p95 latency, request throughput, and log bytes with the Phase 0
baseline. Do not claim runtime completion from `docker compose config` or unit
tests alone.

Suggested commit: none unless runtime findings require fixes.

### Phase 6: Update architecture, harness, Compose, and deployment docs

This phase is required. Do not declare implementation complete while the
repository still tells operators to read `container.log`.

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
- `docs/DEPLOYMENT_CHECKLIST.md`
  - add preflight directory preparation and post-start log verification
- relevant CUDA and Ascend deployment documentation
- `.harness/context/current-architecture.md`
  - describe only verified implemented behavior
- `.harness/context/monitoring-benchmark.md`
  - update paths, retention, query and performance baseline
- `.harness/context/project-map.md`
  - add the query module and script
- `.harness/context/docs-map.md`
  - classify the research/design reports appropriately
- `.harness/checklists/verification.md`
  - remove `container.log` checks and add Docker/event/query checks
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
