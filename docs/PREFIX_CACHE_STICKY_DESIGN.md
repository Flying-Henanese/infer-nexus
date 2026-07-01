# Prefix Cache Affinity

## Purpose

This note records the current `infer-nexus` approach for improving vLLM
prefix/KV cache locality for chat models.

Current position:

- use Ray Serve replica-level prefix-cache-aware routing
- keep vLLM responsible for the actual prefix/KV cache inside each replica
- do not implement gateway-owned session stickiness as the default design
- limit this optimization to local `backend: vllm` chat deployments

## Current Scheme

For local vLLM chat models, requests follow this path:

```text
Client
  -> infer-nexus FastAPI gateway
  -> RuntimeDispatcher / RuntimeExecutor
  -> Ray Serve deployment handle
  -> Ray Serve request router
  -> selected vLLM replica
```

`infer-nexus` can pass Ray Serve request-router configuration from
`config/models.yaml` into the generated Serve deployment. A chat model may opt
into Ray Serve's prefix-aware router with:

```yaml
deployment_config:
  request_router_config:
    request_router_class: ray.serve.llm.request_router.PrefixCacheAffinityRouter
    request_router_kwargs:
      imbalanced_threshold: 16
```

This configuration is deployment-level Ray Serve configuration. It is not a
vLLM engine option. The division of responsibility is:

- Ray Serve chooses the replica with prefix-cache affinity in mind.
- vLLM performs the actual prefix/KV cache reuse inside the selected replica.
- `infer-nexus` keeps owning the northbound API, model lookup, policy, admission,
  and dispatch boundary.

The implementation touch points are:

- `config/models.yaml`
- `src/infer_nexus/catalog/models.py`
- `src/infer_nexus/runtime/deployments.py`
- `tests/test_runtime.py`

## Scope

This scheme applies to:

- `task: chat`
- local `backend: vllm`
- Ray Serve managed deployments
- vLLM prefix/KV cache locality

It does not currently apply to:

- `embedding`
- `rerank`
- one-shot multimodal parsing workloads
- `backend: vllm_openai_proxy`

## Why Not Gateway-Owned Stickiness

The old gateway-owned sticky-session idea is no longer the preferred design.

The main reason is that the real cache island is the vLLM replica. Prefix/KV
cache locality is gained or lost when Ray Serve chooses which replica receives a
request, so the routing decision belongs closer to Ray Serve's replica
scheduler than to the HTTP gateway.

`session_id` stickiness is also only an approximation. vLLM prefix caching is
based on repeated prompt prefixes, not business-session identity. Two requests
from the same session may not share a useful prefix, while requests from
different sessions may share a long system prompt or few-shot prefix.

Putting sticky tables or prefix-hash routing in the gateway would therefore:

- duplicate logic at a layer that does not own replica scheduling
- couple `infer-nexus` to backend-local cache behavior
- blur the boundary between OpenAI-compatible gateway duties and runtime
  placement decisions

Ray Serve's `PrefixCacheAffinityRouter` is a more direct fit because it operates
at the replica-selection layer.

## Validation Notes

The configuration path is implemented and covered by unit tests, but runtime
validation is still environment-dependent. Keep checking:

- whether the deployed Ray version supports the configured request router
- whether the router can extract useful prefix signals from the current
  `handle.chat_completion.remote(payload)` call shape
- whether repeated-prefix chat workloads improve TTFT, prefill latency, p95/p99,
  or timeout/error rate

If Ray-native routing becomes unusable for structural reasons, the preferred
fallback is a thin Ray-side router adapted to the `infer-nexus` payload shape.
Gateway-owned affinity should remain a last resort.
