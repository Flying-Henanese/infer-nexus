# AGENTS.md

## Purpose

This file guides Codex and other implementation agents working in the `infer-nexus` repository. Keep it short: it is the always-loaded entry point, while task-specific details live under `.harness/`.

The source of truth for architectural decisions is [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md). This file translates that design into durable routing rules, guardrails, and repository expectations so that code changes stay aligned with the intended system.

## Project Intent

`infer-nexus` is a shared inference service factory for internal development and testing environments.

It exists to replace ad hoc per-user model deployments on shared servers with a centrally managed service that:
- exposes a single API entrypoint
- shares a bounded accelerator pool
- serves pre-registered models
- uses `Ray Serve` for deployment lifecycle and scaling
- uses `vLLM` as the default inference backend
- stays compatible with OpenAI-style client integrations where practical

## Source of Truth

Use the source that matches the question:
- For intended architecture and durable design constraints, use
  `docs/ARCHITECTURE.md`. This file and `.harness/rules/` route and apply those
  constraints to development tasks.
- For current checked-in behavior, inspect source code, configuration, and
  tests. `.harness/context/` summarizes the implementation, while
  `.harness/context/docs-map.md` identifies which long-form documents describe
  current behavior or future designs. Verify status claims against source.
- `.harness/plans/` describes future work. A plan does not change current
  behavior or guardrails before its implementation is complete.

Do not invent architecture that conflicts with `docs/ARCHITECTURE.md`.

## Harness Routing

This repository uses `.harness/` as a Codex collaboration workspace. Use
`.harness/README.md` to find the context, rules, plan, checklist, or workflow
that matches the task. Read only the relevant files, then verify current
behavior against source before changing code.

## Always-On Guardrails

- Keep models pre-registered in configuration unless the user explicitly changes Phase 1 scope.
- Preserve one Ray Serve deployment per model and one shared accelerator pool.
- Treat accelerator visibility variables as platform pool boundaries, not per-model pinning.
- Keep OpenAI-compatible inference APIs separate from native platform/control APIs.
- Keep admission control explicit; do not hide readiness or overload behind generic timeouts.
- Isolate vLLM-specific behavior behind backend/runtime adapters.
- Keep Compose runtime logs host-visible under `${LOGS_HOST_PATH}/<service>/`:
  `container.log` captures the service command and `ray/` is that service's
  `/tmp/ray` tree. Prepare host directories with
  `scripts/prepare_compose_logs.py` before Compose startup; runtime services
  remain non-root and must not change host ownership.
- Do not add Web UI, tenant-specific isolation, dynamic registration, cross-cluster scheduling, or multi-backend coexistence unless explicitly requested.

## Change Discipline

When making changes:
- prefer small, architecture-aligned commits
- avoid speculative abstractions unless they clearly preserve known extension paths
- document any intentional deviation from `docs/ARCHITECTURE.md`
- update `docs/ARCHITECTURE.md`, this file, and relevant `.harness/` files if a user-approved design change alters repository expectations
- when a task explicitly implements a user-approved design change, update those
  documents alongside the implementation; the plan alone does not replace
  current guardrails

## When in Doubt

If an implementation choice would trade away architecture clarity for short-term convenience, prefer architecture clarity.

If a requested change conflicts with `docs/ARCHITECTURE.md` and is not an
explicitly requested, user-approved design change, stop and surface the
conflict before proceeding.
