# infer-nexus

Shared inference service factory for internal development and testing.

## Current Runtime Shape

- Runtime mode: `Ray Serve + vLLM`
- Gateway: `FastAPI` (`/v1/*` OpenAI-compatible + `/api/*` platform APIs)
- Default gateway bind: `0.0.0.0:8000`
- Current catalog focus: chat models (see `config/models.yaml`)

## Quick Start

For the current smallest runnable deployment shape:

- [MINIMAL_STARTUP.md](./MINIMAL_STARTUP.md)

For first-time Ubuntu + CUDA bring-up:

- [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md)

## Documentation Map

- [ARCHITECTURE.md](./ARCHITECTURE.md): system design, boundaries, and phase goals
- [AGENT.md](./AGENT.md): implementation rules and constraints
- [NEXT_SESSION.md](./NEXT_SESSION.md): latest handoff notes and known runtime issues
