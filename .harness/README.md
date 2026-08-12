# Harness Workspace

`.harness/` is a Codex collaboration workspace for this repository. It does not replace `docs/`, `tests/`, or `scripts/`.

Use it as a short, current guide to:

- orient Codex before complex architecture, refactor, debugging, or review tasks
- find the relevant long-form document under `docs/`
- choose the right verification checklist before claiming work is complete

Do not store secrets, large logs, model artifacts, or generated benchmark outputs here.

Files under `context/` describe the checked-in implementation and configuration as they exist now. Files under `rules/` are normative guardrails and may describe required target behavior that is still only partial or stubbed. Do not infer that a rule is already implemented without checking the matching context file and source.

## Current Entry Points

- `context/current-architecture.md`: current implemented architecture baseline.
- `context/docs-map.md`: status map for architecture-related documents.
- `context/project-map.md`: source tree and test navigation map.
- `context/request-flows.md`: common startup, inference, platform, metrics, and health request flows.
- `context/model-config.md`: current model catalog and config rules.
- `context/monitoring-benchmark.md`: current monitoring and benchmark baseline.
- `rules/implementation-rules.md`: routing index for detailed implementation guardrails.
- `rules/architecture-rules.md`: architecture and runtime guardrails.
- `rules/interface-rules.md`: config, API, metrics, and error rules.
- `rules/delivery-rules.md`: implementation sequence and acceptance criteria.
- `checklists/verification.md`: practical verification checklist for Codex work.
