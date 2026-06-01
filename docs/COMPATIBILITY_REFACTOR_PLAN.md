# Infer-Nexus Compatibility Refactor Plan

## Goal

Reduce protocol drift between `infer-nexus` and upstream `vLLM` / OpenAI-compatible behavior.

The core direction is:

- keep the gateway focused on routing, policy, and observability
- minimize request/response reinterpretation in local gateway code
- use native `vLLM` OpenAI serving or proxy passthrough as the primary chat path
- keep local `LLM.chat(...)` behavior as a limited fallback, not the compatibility baseline


## Problem Summary

The current chat path still contains gateway-local protocol interpretation in
[src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py).

This creates repeated compatibility failures when upstream `vLLM` behavior changes or when callers use modern OpenAI-style payloads:

- text content blocks vs plain string content
- multimodal message content
- tool-calling request fields
- reasoning-related request fields
- streaming chunk shape
- native OpenAI serving adapter availability and fallback behavior

The specific class of bugs we just hit is a symptom of this: the gateway interpreted message content locally and rejected payloads that should have been normalized or passed through.


## Target Architecture

### Desired State

For chat requests:

1. **Strict compatibility models**
   - must use native `vLLM` OpenAI serving or `vllm_openai_proxy`
   - must not silently fall back to local `LLM.chat(...)`

2. **Best-effort local models**
   - may use local `LLM.chat(...)`
   - do not claim full OpenAI compatibility
   - support only the subset explicitly documented by this project

### Gateway Responsibilities

The gateway should primarily own:

- auth
- routing
- model selection
- admission / rate limiting
- request ID propagation
- logging / metrics
- minimal model name remapping

The gateway should avoid owning:

- detailed chat message semantics
- multimodal content interpretation beyond basic validation
- tool-calling protocol behavior replication
- reasoning field semantics
- stream chunk protocol recreation unless unavoidable


## Implementation Plan

## Phase 1: Stop Silent Drift

### Objective

Make compatibility failures explicit instead of silently degrading into a different execution path.

### Changes

1. **Fail fast when native serving is required but unavailable**
   - File:
     - [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
   - Change:
     - when `openai_serving.enabled=true` for a strict-compatibility model, adapter init failure must surface as an error
     - remove silent fallback for that mode
   - Reason:
     - silent fallback hides protocol drift and makes debugging much harder

2. **Introduce explicit compatibility mode in model config**
   - Files:
     - [src/infer_nexus/catalog/models.py](/F:/GitHub/infer-nexus/src/infer_nexus/catalog/models.py)
     - [src/infer_nexus/runtime/dispatcher.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/dispatcher.py)
   - Proposed config:
     - `compat_mode: strict_openai | local_best_effort`
   - Meaning:
     - `strict_openai`: native serving or proxy only
     - `local_best_effort`: local backend path is allowed

3. **Improve adapter initialization visibility**
   - Files:
     - [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
     - optionally [src/infer_nexus/api/openai_routes.py](/F:/GitHub/infer-nexus/src/infer_nexus/api/openai_routes.py)
   - Change:
     - log adapter init failure with concrete reason
     - make it visible in request failure logs

### Acceptance Criteria

- strict models do not silently fall back to local `LLM.chat(...)`
- adapter init failures are visible in logs
- model config clearly states compatibility expectations


## Phase 2: Move Chat to Native Paths

### Objective

Make native `vLLM` OpenAI serving or proxy passthrough the default compatibility path for chat.

### Changes

1. **Use native OpenAI serving as the default local chat path**
   - Files:
     - [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
     - [src/infer_nexus/runtime/deployments.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/deployments.py)
     - [src/infer_nexus/runtime/serve_app.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/serve_app.py)
   - Change:
     - for strict chat models, require native serving adapter readiness during startup
     - fail deployment startup if the adapter cannot be constructed

2. **Downgrade local `LLM.chat(...)` to a documented fallback path**
   - File:
     - [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
   - Change:
     - keep it for simple local/dev cases
     - do not continue expanding it as the main OpenAI compatibility layer

3. **Prefer `vllm_openai_proxy` for compatibility-sensitive models**
   - Files:
     - [src/infer_nexus/runtime/executor.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/executor.py)
     - [src/infer_nexus/catalog/models.py](/F:/GitHub/infer-nexus/src/infer_nexus/catalog/models.py)
   - Candidate models:
     - multimodal models
     - tool-calling heavy models
     - reasoning-sensitive models
     - models used by external SDK clients expecting close OpenAI behavior

### Acceptance Criteria

- strict chat models use native serving or proxy only
- local `LLM.chat(...)` is no longer the implicit default compatibility path
- compatibility-sensitive models can be isolated onto proxy/native execution paths


## Phase 3: Add Contract Tests

### Objective

Detect protocol drift in CI before it reaches test or production environments.

### Changes

1. **Add chat compatibility contract tests**
   - Files:
     - [tests/test_runtime.py](/F:/GitHub/infer-nexus/tests/test_runtime.py)
     - [tests/test_dispatcher.py](/F:/GitHub/infer-nexus/tests/test_dispatcher.py)
     - [tests/test_api.py](/F:/GitHub/infer-nexus/tests/test_api.py)
     - new file recommended:
       - `tests/test_openai_compat_contract.py`

2. **Cover request-shape compatibility cases**
   - string content
   - text-only content blocks
   - image content blocks
   - tool-calling fields
   - `tool_choice`
   - `parallel_tool_calls`
   - reasoning fields
   - stream and non-stream behavior
   - error code and error body shape

3. **Add comparison tests against upstream behavior**
   - compare gateway behavior to:
     - native `vLLM` OpenAI serving
     - or a controlled upstream proxy target
   - focus on:
     - request normalization
     - response shape
     - SSE chunk shape
     - error classification

### Acceptance Criteria

- compatibility-sensitive request forms are covered by tests
- `vLLM` upgrades cannot pass CI if request/response behavior drifts
- stream and non-stream behavior are validated separately


## Files Most Likely to Change

### Core runtime path

- [src/infer_nexus/backends/vllm.py](/F:/GitHub/infer-nexus/src/infer_nexus/backends/vllm.py)
- [src/infer_nexus/runtime/executor.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/executor.py)
- [src/infer_nexus/runtime/dispatcher.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/dispatcher.py)
- [src/infer_nexus/runtime/deployments.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/deployments.py)
- [src/infer_nexus/runtime/serve_app.py](/F:/GitHub/infer-nexus/src/infer_nexus/runtime/serve_app.py)

### Configuration and schema

- [src/infer_nexus/catalog/models.py](/F:/GitHub/infer-nexus/src/infer_nexus/catalog/models.py)
- [src/infer_nexus/core/enums.py](/F:/GitHub/infer-nexus/src/infer_nexus/core/enums.py)

### API surface and error mapping

- [src/infer_nexus/api/openai_routes.py](/F:/GitHub/infer-nexus/src/infer_nexus/api/openai_routes.py)

### Tests

- [tests/test_runtime.py](/F:/GitHub/infer-nexus/tests/test_runtime.py)
- [tests/test_dispatcher.py](/F:/GitHub/infer-nexus/tests/test_dispatcher.py)
- [tests/test_api.py](/F:/GitHub/infer-nexus/tests/test_api.py)
- [tests/test_multimodal_contract.py](/F:/GitHub/infer-nexus/tests/test_multimodal_contract.py)


## Recommended Short-Term Execution Order

1. add `compat_mode` to model config
2. disable silent fallback for strict models
3. improve adapter init logging and startup validation
4. route strict chat models to native serving or proxy
5. add high-signal contract tests


## First Deliverable Scope

The smallest useful first delivery is:

1. `compat_mode` in model config
2. fail-fast behavior for strict models when native serving adapter init fails
3. no silent fallback for strict chat models
4. 6-8 contract tests covering:
   - string content
   - text-only block content
   - image block content
   - tool-calling
   - reasoning
   - streaming shape

This does not fully finish the architecture shift, but it removes the most dangerous class of hidden compatibility drift.


## Non-Goals

These items should not be mixed into the first refactor pass:

- broad unrelated cleanup in runtime modules
- changing embedding or rerank paths unless compatibility issues require it
- redesigning all configuration structures at once
- rewriting every local fallback path before strict-mode guardrails are in place


## Final Success Criteria

This refactor is successful when:

- chat compatibility behavior no longer depends primarily on gateway-local protocol recreation
- strict models either behave like upstream native serving or fail clearly
- SDK clients can use modern OpenAI request shapes without unexpected gateway-specific breakage
- `vLLM` version upgrades become test-driven instead of incident-driven
