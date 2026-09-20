# Logging Storage and Query Implementation Report

Status: implementation delivered; CUDA/A100 runtime validation completed on
2026-09-20. The explicit unsafe or unavailable failure scenarios below remain
validation gaps rather than unreported assumptions.

This report records the evidence and boundaries for
`.harness/plans/logging-storage-query-redesign-plan.md`. It is deliberately
separate from the operator runbook: the runbook gives commands, while this file
records which source owns each field and what was actually verified.

## Phase 0 coverage worksheet

| Diagnostic question | Authoritative owner/source | Event or raw field | Current status |
| --- | --- | --- | --- |
| Which request/model/route/process? | Gateway request lifecycle; replica context | `request_id`, route, model/task/backend, physical service, process instance | Implemented; A100 request IDs were queried from Gateway events and worker identity was observed in its separate event file |
| Why rejected? | Gateway admission owner or Serve-handle guard | `admission_layer`, `failure_stage`, stable `error_code` | Implemented and unit-covered; A100 catalog rejection observed; overload live test deferred |
| Why timed out? | Serve-handle/stream/proxy branch | bounded `timeout_kind` and `failure_stage` | Implemented at known branches and unit-covered; live timeout not induced |
| Which stage made startup fail? | `ModelRuntimeReplica.__init__` | `startup_stage` | Implemented (`config`, `artifact`, `backend`, `engine`, `unknown`); A100 startup success observed, failure injection deferred |
| How slow was the request? | Gateway lifecycle plus existing executor observations | `duration_ms`, optional admission/handle/TTFT fields | Implemented with documented boundaries; A100 non-stream and stream timings observed |
| How many tokens? | Response usage owner | optional input/output counts and `usage_source` | Implemented; A100 non-stream usage observed and missing stream usage remained absent |
| What happened before Gateway? | Ray Serve proxy and Ray raw files | Docker/Ray access and queue signals | No fabricated application event; A100 Ray infra query works, saturation evidence not induced |
| Did a process die without flushing? | Docker state plus Ray actor/core files | exit/OOM/restart and raw session records | No fabricated application event; process-death scenario not induced |
| Is the event writer healthy? | Process-owned handler and Prometheus counter | stderr diagnostic and `infer_nexus_event_writer_failures_total` | Implemented, failure-injection tested locally and in the A100 Python 3.12 test run |

The current defaults are 10 MiB per application event file with five rotated
files per process. Docker uses `json-file` with the existing 10 MiB × 5 policy;
Ray component files retain their 50 MiB × 3 policy. These are per-file limits,
not a promise about total service-directory size. `--stats` reports the actual
directory coverage without deleting files or treating missing records as proof
of logging health.

## Source ownership

- `events-*.jsonl*` is the authoritative structured infer-nexus application
  dataset. One process owns one filename and the handler accepts only marked
  infer-nexus structured events.
- Docker stdout/stderr is the container/bootstrap and signal/exit dataset. It
  may contain a display copy of an application event and must not be ingested
  as a second application-event dataset.
- `ray/` is the raw Ray/Serve/vLLM/actor dataset. `session_latest` is treated
  as an alias and is not counted in addition to its target.
- `scripts/logs.py` keeps these sources separate in default, `--infra`, `--raw`,
  `--follow`, and `--stats` modes.

## Implemented checks

Deterministic tests cover schema/identity, redaction, third-party filtering,
idempotent configuration, writer failure fallback, request correlation across
three physical services, malformed and partial JSONL, rotation/follow,
`session_latest` deduplication (including container absolute symlinks), path
safety, and storage statistics. The final A100 run passed all 43 tests in the
focused logging/query/config/Compose set.

## Remote A100 evidence

Validation target: `/home/mineru_dev/github/infer-nexus` on `a100`, branch
`dev`, final commit `a7eca12`. The implementation was delivered in the main
feature commit `23323a7` followed by diagnostic-boundary, container-session,
and test-harness follow-ups `6c23db4`, `7761143`, and `a7eca12`.

- Remote toolchain: `/home/mineru_dev/.local/bin/uv` `0.9.28`; `uv run
  --frozen --python 3.12 python --version` reported Python `3.12.12`.
- Focused validation command: `CUDA_VISIBLE_DEVICES=2,3 uv run --frozen
  --python 3.12 --extra dev pytest -q tests/test_logging.py
  tests/test_log_query.py tests/test_config.py tests/test_compose_logging.py`;
  result: `43 passed in 0.39s`.
- `scripts/prepare_compose_logs.py --env-file .env` prepared
  `/home/mineru_dev/github/infer-nexus-logs/{ray-head,ray-worker,serve-deployer}`
  for uid/gid `1002:1003`. Root CUDA Compose `config -q` passed. Ascend
  Compose `config -q` also passed with the non-starting placeholder
  `ASCEND_RT_VISIBLE_DEVICES=0,1`.
- Root CUDA Compose reached `ray-head` healthy, `ray-worker` healthy, and
  `serve-deployer` exited `0`. The model replica emitted `model.replica.ready`
  after its real vLLM startup; the Gateway application then became ready.
  `docker compose logs` showed direct Ray/vLLM/bootstrap output and structured
  `app.ready` records on Docker stdout.
- Host API checks returned `{"status":"ok","service":"infer-nexus"}` for
  `/healthz` and `/readyz`, and `/v1/models` exposed the active
  `qwen3.5-9b` catalog entry.
- Non-stream request ID `41a3d1b409e24b79af8cdce036f35208` returned HTTP 200
  with usage and produced a structured `request.completed` event. Stream
  request ID `f6b5408b984c4848a71911bfe06d1723` returned HTTP 200 SSE with
  both request and legacy stream request headers; the query showed
  `stream.completed`, `emitted_chunk_count=10`, `response_headers_sent=true`,
  and `ttft_ms=69.849`. Safe catalog rejection request ID
  `b27b311b57e7465eb061685a4e700343` returned HTTP 404 and was recorded with
  `error_code=model_not_found` and `failure_stage=catalog_lookup`.
- Current event files were:
  `ray-head/events-gateway-2208512809d7462185db3c9dc232017d.jsonl`,
  `ray-worker/events-model_replica-7287e98aca8b403986d164961d048060.jsonl`,
  and `serve-deployer/events-serve_deployer-f3a884ceeccd4ffa824cfbbb606e6532.jsonl`.
  They contained schema version 1 records with physical-service and stable
  process-instance identity.
- After the container-absolute `session_latest` fix, `scripts/logs.py
  --stats` reported `events: files=3 bytes=6007`, `ray_logs: files=113
  bytes=4519794`, `ray_other_session_contents: files=363 bytes=3979555`, and
  `historical_sessions: files=461 bytes=5523757`, with oldest/newest UTC
  timestamps. `--infra --service ray-worker --since 30m --raw` returned 500
  source-attributed lines from the current session.
- The current run did not create a `container.log`; an older file at
  `/home/mineru_dev/github/infer-nexus-logs/serve-deployer/container.log` has
  timestamp 2026-09-17 and was preserved by the non-destructive preparation
  script. The active repository has no `scripts/container/run_with_log.sh`.

The broader remote suite was also sampled with `pytest -q
--ignore=tests/test_compose_logging.py`: it reported `161 passed, 2 skipped,
52 failed`. Those failures are the documented pre-existing catalog/alias
fixture mismatch, dependency-version-sensitive runtime mocks, and
cross-test global-state baseline, not failures in the 43 focused tests above.

## Explicit gaps and deferred P1 work

- No cleanup command is implemented; retention is limited to per-process event
  rotation plus Ray/Docker runtime policies and reported by `--stats`.
- Queue pressure before FastAPI middleware remains owned by Ray Serve proxy
  access logs and Serve queue metrics.
- First emitted SSE chunk is recorded as gateway-observed TTFT, not a claim of
  true model-token timing.
- Live overload, Serve-handle timeout, stream failure, replica startup failure,
  and process-death scenarios were not induced on the shared A100. Their
  ownership remains explicit in the application/Ray/Docker sources and their
  safe deterministic branches are covered by focused tests where applicable.
- Capacity growth and inference overhead were not benchmarked as part of this
  change; the A100 run verifies functionality and storage observability, not a
  performance budget.
- Ascend end-to-end validation requires an Ascend host and is not implied by
  CUDA/A100 validation.
