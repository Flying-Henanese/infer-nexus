# Harness Workspace

`.harness/` is a Codex collaboration workspace for this repository. It does not replace `docs/`, `tests/`, or `scripts/`.

Use it as a short, current guide to:

- orient Codex before complex architecture, refactor, debugging, or review tasks
- find the relevant long-form document under `docs/`
- choose the right verification checklist before claiming work is complete

Do not store secrets, large logs, model artifacts, or generated benchmark outputs here.

Files under `context/` describe the checked-in implementation and configuration as they exist now. Files under `rules/` are normative guardrails and may describe required target behavior that is still only partial or stubbed. Do not infer that a rule is already implemented without checking the matching context file and source.

Files under `plans/` record execution sequences and their completion status.
They are not evidence that a step has been implemented; confirm completion
against source, tests, and the matching `context/` files.

## Read By Task

| Task | Start with |
| --- | --- |
| Architecture, Ray Serve runtime, backend, scaling, or admission | `rules/architecture-rules.md` and `context/current-architecture.md` |
| API, configuration, metrics, logging, or error behavior | `rules/interface-rules.md`, then the matching context file below |
| Source and test navigation | `context/project-map.md` |
| Startup, inference, platform, metrics, or health request paths | `context/request-flows.md` |
| Model catalog or `config/models.yaml` | `context/model-config.md` |
| Monitoring or benchmarking | `context/monitoring-benchmark.md` |
| Choosing a long-form design or operations document | `context/docs-map.md` |
| Completion checks | `checklists/verification.md` |
| Shared accelerator deployment and live validation | `workflows/remote-deploy-and-validate.md` |

Read the smallest set of files needed for the task. The current implementation
is summarized in `context/`; confirm details against source before editing.

## Active Plans

- `plans/logging-storage-query-redesign-plan.md`: Compose stdout restoration,
  process-isolated event files, local queries, storage reporting, and related
  documentation are implemented. Its unchecked acceptance items track
  diagnostic coverage and capacity/runtime-overhead validation gaps. Check
  `docs/LOGGING_STORAGE_QUERY_IMPLEMENTATION.md` for remote evidence.

## Historical Plans

- `plans/logging-management-execution-plan.md`: structured JSONL logging,
  request correlation, and framework adapters. Its application-layer work is
  substantially implemented; use the active plan's unchecked items for
  remaining validation.
