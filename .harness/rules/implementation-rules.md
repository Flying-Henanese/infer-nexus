# Implementation Rules

This is the routing index for detailed implementation guardrails. Keep this file short; open only the narrow rule file that matches the task.

## Rule Files

- `.harness/rules/architecture-rules.md`: model lifecycle, Ray Serve boundaries, accelerator pool rules, admission, scaling, and backend substitution.
- `.harness/rules/interface-rules.md`: configuration, model catalog expectations, OpenAI-compatible APIs, native APIs, metrics, logs, and errors.
- `.harness/rules/delivery-rules.md`: implementation sequence and Phase 1 acceptance criteria.

## When To Read

- For architecture, runtime, scheduler, backend, or admission changes, read `architecture-rules.md`.
- For API, config, metrics, logging, or error-shape changes, read `interface-rules.md`.
- For planning, sequencing, or final acceptance checks, read `delivery-rules.md`.

If a task spans multiple areas, read the smallest set of rule files needed for that task.
