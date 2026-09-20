# Logging Storage and Query Implementation Report

Status: implementation in progress; remote A100 validation is required before
the redesign is considered complete.

This report records the evidence and boundaries for
`.harness/plans/logging-storage-query-redesign-plan.md`. It is deliberately
separate from the operator runbook: the runbook gives commands, while this file
records which source owns each field and what was actually verified.

## Phase 0 coverage worksheet

| Diagnostic question | Authoritative owner/source | Event or raw field | Current status |
| --- | --- | --- | --- |
| Which request/model/route/process? | Gateway request lifecycle; replica context | `request_id`, route, model/task/backend, physical service, process instance | Implemented in structured event schema; remote correlation pending |
| Why rejected? | Gateway admission owner or Serve-handle guard | `admission_layer`, `failure_stage`, stable `error_code` | Implemented and unit-covered; live rejection pending |
| Why timed out? | Serve-handle/stream/proxy branch | bounded `timeout_kind` and `failure_stage` | Implemented at known branches; live timeout pending |
| Which stage made startup fail? | `ModelRuntimeReplica.__init__` | `startup_stage` | Implemented (`config`, `artifact`, `backend`, `engine`, `unknown`); live failure pending |
| How slow was the request? | Gateway lifecycle plus existing executor observations | `duration_ms`, optional admission/handle/TTFT fields | Implemented with documented boundaries; measurement pending |
| How many tokens? | Response usage owner | optional input/output counts and `usage_source` | Implemented; missing usage stays absent |
| What happened before Gateway? | Ray Serve proxy and Ray raw files | Docker/Ray access and queue signals | No fabricated application event; live saturation evidence pending |
| Did a process die without flushing? | Docker state plus Ray actor/core files | exit/OOM/restart and raw session records | No fabricated application event; live failure evidence pending |
| Is the event writer healthy? | Process-owned handler and Prometheus counter | stderr diagnostic and `infer_nexus_event_writer_failures_total` | Implemented and failure-injection tested |

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
`session_latest` deduplication, path safety, and storage statistics. Local
dependency availability is environment-dependent; the required remote
commands and observed outputs must be appended below after the A100 run.

## Remote A100 evidence

To be filled from `/home/mineru_dev/github/infer-nexus` after the pushed commit
is pulled. Record the commit, test commands, Compose status, API smoke request
ID, event file paths, and query output categories here. If a planned failure
scenario is unsafe or unavailable on the host, record it as an explicit gap
rather than manufacturing an event.

## Explicit gaps and deferred P1 work

- No cleanup command is implemented; retention is limited to per-process event
  rotation plus Ray/Docker runtime policies and reported by `--stats`.
- Queue pressure before FastAPI middleware remains owned by Ray Serve proxy
  access logs and Serve queue metrics.
- First emitted SSE chunk is recorded as gateway-observed TTFT, not a claim of
  true model-token timing.
- Ascend end-to-end validation requires an Ascend host and is not implied by
  CUDA/A100 validation.
