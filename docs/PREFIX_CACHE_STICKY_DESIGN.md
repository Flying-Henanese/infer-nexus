# infer-nexus Prefix Cache Affinity Design Note

## 1. Goal

Re-evaluate how `infer-nexus` should improve KV/prefix cache locality for chat models.

Updated conclusion:

- do not continue with a gateway-owned sticky-session design
- do not treat `session_id` stickiness as the primary solution
- prefer Ray Serve's replica-level prefix-cache-aware routing as the first-class direction
- keep the optimization scope limited to `task: chat`

This document records the updated design position after investigating Ray Serve and vLLM capabilities.

---

## 2. Why The Previous Sticky-Session Draft Is No Longer Preferred

The earlier design direction assumed `infer-nexus` should implement its own affinity layer using conversation or prefix-derived hashes at the gateway/proxy layer.

That design is no longer preferred for this project.

Reasons:

- the real optimization target is vLLM prefix/KV cache reuse, not business-session stickiness
- `session_id` is only an indirect proxy for cache locality
- Ray Serve already provides a more direct routing abstraction for this exact problem
- vLLM cache locality is replica-local, so replica-level routing is more fundamental than gateway-level sticky bookkeeping
- a gateway-owned sticky table or gateway-owned prefix-hash routing would duplicate logic that is better placed closer to replica scheduling

The old session-oriented approach is therefore demoted from "planned implementation direction" to "fallback idea only if Ray-native routing proves unusable".

---

## 3. Updated Architectural Position

For local `backend: vllm` chat models served through Ray Serve, the most promising route is:

```text
Client
  -> infer-nexus FastAPI gateway
  -> RuntimeDispatcher / RuntimeExecutor
  -> Ray Serve deployment handle
  -> Ray Serve replica-level request router
  -> chosen vLLM replica
  -> vLLM APC / prefix cache reuse
```

This is more direct than:

```text
Client
  -> infer-nexus
  -> gateway-owned sticky routing
  -> manually selected upstream instance
```

because the actual cache island lives inside the vLLM replica, not at the HTTP gateway boundary.

---

## 4. Scope

This design note applies only to:

- `task: chat`
- local `backend: vllm`
- Ray Serve managed replicas
- vLLM prefix/KV cache locality

This design note does not apply to:

- `embedding`
- `rerank`
- `mineru`-style one-shot multimodal parsing workloads
- `vllm_openai_proxy` as the primary target

Current position:

- only chat models are considered worth cache-affinity work
- non-chat tasks should continue using standard routing unless a separate workload-specific reason appears later

---

## 5. What Ray Serve And vLLM Already Provide

### 5.1 Ray Serve

Ray Serve already has a replica-level routing layer and a built-in `PrefixCacheAffinityRouter` in the Ray Serve LLM stack.

Why this matters:

- cache locality is created or lost at replica selection time
- the router works at the same layer where replica choice is actually made
- this is a better fit than trying to emulate replica affinity from the gateway

The key design implication is:

- if Ray Serve's prefix-cache-aware routing can be used in our deployment topology, it should be preferred over custom gateway sticky routing

### 5.2 vLLM

vLLM APC/prefix caching is based on prompt-prefix reuse, not session identity.

Implications:

- `session_id` is not a native cache key inside vLLM
- routing requests by prefix similarity is more faithful to the real optimization target than routing by conversation ID alone
- `cache_salt` is a cache-isolation control, not a sticky-session mechanism

---

## 6. Core Design Decision

The primary direction for future investigation is:

- use Ray Serve's prefix-cache-aware routing rather than implementing session-based sticky routing in `infer-nexus`

Restated more concretely:

1. `infer-nexus` should continue owning northbound OpenAI-compatible APIs and model selection.
2. Ray Serve should be treated as the preferred layer for replica-level cache-affinity decisions.
3. vLLM should remain responsible for the actual APC/prefix-cache behavior inside each replica.

This is a cleaner responsibility split than trying to make the gateway itself simulate cache-affinity ownership.

---

## 7. What This Replaces

The following ideas from the earlier draft are no longer preferred:

- gateway-owned session stickiness
- conversation-ID-driven affinity as the main mechanism
- gateway-computed prefix fingerprints as the primary routing primitive
- a custom upstream-instance rendezvous hash design as the default future path
- treating `vllm_openai_proxy` as the first implementation target for prefix-cache affinity

These are not forbidden forever, but they are no longer the recommended baseline.

---

## 8. Current Fit With infer-nexus

### 8.1 What already aligns well

The current `infer-nexus` architecture already routes local `backend: vllm` requests through Ray Serve deployment handles:

```text
FastAPI gateway
  -> RuntimeDispatcher
  -> RuntimeExecutor
  -> handle.chat_completion.remote(payload)
```

This is good news because:

- replica selection already happens on the Ray Serve side
- the current stack does not need to invent a fake "replica address" abstraction
- the right routing layer is already present in the execution path

### 8.2 What does not line up yet

The current codebase does not yet configure any Serve request router for model deployments.

Today, the local `vllm` path:

- creates one deployment per model
- binds `ModelRuntimeReplica`
- gets a deployment handle by app/deployment name
- calls replica methods with a dict payload

What is missing:

- deployment-level `RequestRouterConfig`
- a verified integration of `PrefixCacheAffinityRouter`
- proof that the built-in router correctly understands our current request shape

So the situation is:

- the architecture is compatible in principle
- the integration is not yet proven in this codebase

---

## 9. Known Integration Risks

These are the main reasons this should still be treated as an investigation, not as a closed implementation plan.

### 9.1 Request-shape compatibility is not yet proven

Ray Serve's built-in prefix-aware routing is documented in the Ray Serve LLM stack.

Our current path does not use Ray Serve's standard OpenAI ingress objects.
Instead, `infer-nexus` serializes `ChatCompletionsRequest` into a plain payload dict and calls deployment methods directly.

Open question:

- can `PrefixCacheAffinityRouter` derive useful prefix-affinity signals from our current handle-call payload shape without additional adaptation?

This is the most important unresolved risk.

### 9.2 API maturity risk

Ray Serve request-router APIs are still relatively new and documented with evolving/alpha-style caveats.

Implication:

- even if the design direction is correct, we should expect integration details to be somewhat version-sensitive

### 9.3 Current code path is custom, not Ray Serve LLM stock ingress

We use:

- custom FastAPI gateway
- custom dispatcher/executor
- custom replica methods

We do not use:

- Ray Serve LLM's stock ingress path end-to-end

Implication:

- "Ray has this feature" does not automatically mean "we can enable it without adaptation"

---

## 10. Updated Non-Goals

This design note does not propose:

- implementing a new session-based sticky router in the gateway
- introducing gateway-managed sticky tables
- requiring `session_id` as a new first-class API field
- treating `session_id` as the canonical cache-affinity key
- extending affinity logic to embedding or rerank
- changing the strict OpenAI request/response compatibility strategy

---

## 11. Interim Recommendation

Until the Ray-native path is validated, the correct position is:

- do not implement the original session-sticky design
- do not add speculative `session_id` plumbing just to prepare for sticky routing
- keep the design centered on investigating Ray Serve prefix-aware routing first

This avoids building a second-best routing system before confirming whether the first-class one is viable.

---

## 12. Investigation Checklist

The next stage should answer these questions in order.

### 12.1 Can our deployments attach a Serve request router?

Need to verify:

- deployment-level configuration shape in the exact Ray version we run
- whether `RequestRouterConfig` can be applied to our one-model-per-deployment topology

### 12.2 Can `PrefixCacheAffinityRouter` work with our current request path?

Need to verify:

- whether it can inspect the payload produced by `handle.chat_completion.remote(payload)`
- whether the current payload contains enough stable prompt structure for useful routing decisions

### 12.3 If not, what is the smallest adaptation layer?

Possible outcomes:

- no adaptation needed
- minor payload-shape adaptation before handle invocation
- custom thin router needed for our payload format

The preferred fallback, if needed, is:

- a thin Ray-side prefix-aware router adapted to our payload format

The non-preferred fallback is:

- reviving the old gateway-owned sticky-session design

### 12.4 Does it materially help real chat workloads?

Need to measure:

- TTFT
- prefill latency
- p95/p99 latency
- timeout/error rate
- cache-locality-related improvements under repeated chat turns

---

## 13. If Ray-Native Routing Proves Unusable

Only if the Ray-native route fails for structural reasons should we reopen a custom design.

If that happens, the fallback order should be:

1. custom Ray-side router adapted to our payload shape
2. only then reconsider gateway-owned affinity logic

Even in fallback mode, session stickiness should still be viewed as an approximation, not the ideal target.

---

## 14. Required Project Touch Points For Future Evaluation

If we continue this work later, the most likely files to examine or update are:

- `src/infer_nexus/runtime/serve_app.py`
- `src/infer_nexus/runtime/deployments.py`
- `src/infer_nexus/runtime/handles.py`
- `src/infer_nexus/runtime/executor.py`
- `src/infer_nexus/api/openai_routes.py`
- `tests/test_runtime.py`
- future integration or contract tests for Serve routing behavior

This list is intentionally smaller and more focused than the previous draft because the preferred path has moved from gateway routing logic to Ray Serve deployment routing.

---

## 15. Final Recommendation

The previous session-based sticky design should be considered superseded.

The current recommended direction is:

- for local `backend: vllm` chat models, investigate Ray Serve `PrefixCacheAffinityRouter` first
- treat prefix-aware replica routing as the primary solution candidate
- keep `infer-nexus` focused on gateway, compatibility, and model dispatch responsibilities
- avoid building a custom gateway sticky-affinity system unless Ray-native integration proves insufficient

In short:

- session stickiness is not the right primary abstraction
- prefix-aware replica routing is the more direct and more correct optimization target
- the remaining work is integration validation, not a fresh routing design from scratch
