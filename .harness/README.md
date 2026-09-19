# Harness Workspace

`.harness/` is a Codex collaboration workspace for this repository. It does not replace `docs/`, `tests/`, or `scripts/`.

Use it as a short, current guide to:

- orient Codex before complex architecture, refactor, debugging, or review tasks
- find the relevant long-form document under `docs/`
- choose the right verification checklist before claiming work is complete

Do not store secrets, large logs, model artifacts, or generated benchmark outputs here.

Files under `context/` describe the checked-in implementation and configuration as they exist now. Files under `rules/` are normative guardrails and may describe required target behavior that is still only partial or stubbed. Do not infer that a rule is already implemented without checking the matching context file and source.

Files under `plans/` describe future execution sequences. They are not evidence
that the planned behavior has been implemented; confirm completion against
source, tests, and the matching `context/` files.

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

- `plans/logging-storage-query-redesign-plan.md`: ready-to-execute Compose
  stdout restoration, process-isolated event files, unified local queries,
  retention controls, and documentation updates.

## Historical Plans

- `plans/logging-management-execution-plan.md`: structured JSONL logging,
  request correlation, and framework adapters. Its application-layer work is
  substantially implemented; use the active plan for remaining storage and
  query work.
