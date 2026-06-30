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

When there is ambiguity, use these priorities:
1. `docs/ARCHITECTURE.md`
2. this `AGENTS.md`
3. relevant files under `.harness/`
4. local code and configuration conventions introduced in this repository

Do not invent architecture that conflicts with `docs/ARCHITECTURE.md`.

## Harness Routing

This repository uses `.harness/` as a Codex collaboration workspace. Read only the files that match the task:

- `.harness/rules/implementation-rules.md`: routing index for detailed implementation guardrails.
- `.harness/context/current-architecture.md`: current implemented gateway/catalog/runtime/backend shape.
- `.harness/context/project-map.md`: source tree and test navigation map before editing code.
- `.harness/context/docs-map.md`: which long-form docs are current, future-only, or superseded.
- `.harness/context/request-flows.md`: startup, inference, platform, metrics, health, and readiness flows.
- `.harness/context/model-config.md`: model catalog and `config/models.yaml` rules.
- `.harness/context/monitoring-benchmark.md`: monitoring and benchmark baseline.
- `.harness/checklists/verification.md`: verification checklist before claiming completion.

For complex design, refactor, debugging, or review tasks, start with the relevant `.harness/` context or rules file instead of loading every document.

## Always-On Guardrails

- Keep models pre-registered in configuration unless the user explicitly changes Phase 1 scope.
- Preserve one Ray Serve deployment per model and one shared accelerator pool.
- Treat accelerator visibility variables as platform pool boundaries, not per-model pinning.
- Keep OpenAI-compatible inference APIs separate from native platform/control APIs.
- Keep admission control explicit; do not hide readiness or overload behind generic timeouts.
- Isolate vLLM-specific behavior behind backend/runtime adapters.
- Do not add Web UI, tenant-specific isolation, dynamic registration, cross-cluster scheduling, or multi-backend coexistence unless explicitly requested.

## Change Discipline

When making changes:
- prefer small, architecture-aligned commits
- avoid speculative abstractions unless they clearly preserve known extension paths
- document any intentional deviation from `docs/ARCHITECTURE.md`
- update `docs/ARCHITECTURE.md`, this file, and relevant `.harness/` files if a user-approved design change alters repository expectations

## When in Doubt

If an implementation choice would trade away architecture clarity for short-term convenience, prefer architecture clarity.

If a requested change conflicts with `docs/ARCHITECTURE.md`, stop and surface the conflict explicitly before proceeding.
