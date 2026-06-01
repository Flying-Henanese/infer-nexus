# infer-nexus Prefix-Cache Sticky Routing Design Draft

## 1. Goal

Design a prefix-cache-friendly sticky routing strategy for `infer-nexus` when requests are proxied to self-hosted upstream Ray/vLLM replicas.

The primary objective is:

- increase prefix cache locality
- improve cache hit probability for repeated multi-turn requests
- preserve OpenAI-compatible request semantics
- avoid hard pinning that can overload one upstream replica

This design is intentionally scoped to routing affinity for cache locality.
It does not introduce server-side conversation state management.

---

## 2. Background

In the current architecture, `infer-nexus` acts as a gateway:

```text
Client
  -> infer-nexus gateway (/v1/*)
      -> model routing by request.model
          -> upstream OpenAI-compatible vLLM endpoints
              -> Ray Serve / vLLM replicas
```

Each upstream vLLM replica maintains its own KV cache and its own prefix cache locality.
There is no cache sharing across replicas.

This means:

- routing related requests to the same replica can improve prefix cache reuse
- routing related requests to different replicas reduces cache locality
- strong stickiness can create hotspots and hurt tail latency

So the design target is not "always send to the same replica".
The design target is "prefer the same cache island unless health or load says otherwise".

---

## 3. Non-Goals

This draft does not attempt to solve:

- cross-session shared cache reuse by default
- server-side chat memory persistence
- sticky routing for local `vllm` backend inside one Ray Serve deployment
- exact replica-level addressing inside Ray Serve internal scheduling
- stream resume or mid-stream failover

The first implementation target is the `vllm_openai_proxy` backend only.

---

## 4. Design Principles

1. Affinity should be soft, not absolute.
2. Routing should remain stateless where possible.
3. Isolation boundaries must be stronger than cache-sharing boundaries.
4. Sticky decisions should happen before proxying upstream.
5. Streaming requests may select an upstream once, but must not switch mid-stream.
6. Overload protection is more important than preserving affinity.

---

## 5. Why Prefix-Hash Routing

For this project, prefix-hash routing is a better fit than plain session stickiness.

Reasons:

- the optimization target is prefix cache locality, not server-side session state
- the current system is already request-oriented and stateless at the gateway layer
- multiple gateway replicas can independently compute the same routing result
- it avoids a centralized sticky table for the common path

Compared with conversation-only stickiness:

- conversation stickiness is simpler
- prefix-hash routing is more aligned with cache locality
- conversation stickiness can still be used as an input signal when available

This draft therefore uses prefix-hash routing as the primary mechanism.

---

## 6. Scope of Cache Sharing

Default position:

- do not assume cross-session cache sharing is desirable
- do not intentionally merge unrelated users or tenants into one cache domain
- only allow affinity within a defined isolation boundary

Recommended cache-sharing boundary:

- `tenant/workspace + model`

This means a prefix fingerprint may be reused for routing decisions only within the same isolation domain.

Suggested rule:

- same tenant + same model + similar prefix -> eligible for the same upstream preference
- different tenant -> never intentionally share prefix affinity

This reduces both accidental cache interference and side-channel exposure risk.

---

## 7. Routing Layer Placement

The affinity logic should be implemented in the gateway proxy routing layer, not in Ray Serve internal replica routing.

Recommended routing flow:

```text
request
  -> resolve model
  -> load upstream pool for this model
  -> derive affinity context
  -> compute prefix fingerprint
  -> rank upstream instances by affinity score
  -> filter by health / enabled / overload
  -> select best candidate
  -> proxy request
```

Why not place it inside Ray Serve deployment routing:

- current local `vllm` path uses deployment handles, not replica-addressable routing
- Ray Serve internal scheduling is not exposed here as stable replica affinity control
- upstream proxy mode already gives a natural instance-selection layer

In practice, the gateway should target distinct upstream vLLM instances or stable upstream instance endpoints.

If `upstream_base_url` points to an external load balancer instead of a concrete instance, affinity value will be reduced.

---

## 8. Affinity Model

The affinity key should be derived from:

- isolation boundary
- model identity
- normalized prefix signal

Recommended logical affinity key:

```text
affinity_key = hash(
  tenant_id
  + model_name
  + prefix_fingerprint
)
```

Optional augmentation:

- if a client provides a stable conversation id, it may be added
- if no tenant id exists, use the strongest available authenticated principal id

Important:

- do not use raw prompt text as a routing key
- do not log raw prefix material
- only log hashes or short opaque identifiers

---

## 9. Prefix Fingerprint Strategy

### 9.1 Objective

The prefix fingerprint should approximate "same cache-relevant prefix" without performing expensive deep parsing or tokenization in the gateway.

### 9.2 First-Version Strategy

For `chat/completions`, compute the fingerprint from stable early prompt structure:

- `model`
- normalized `system` messages
- first `K` messages, where `K` is small
- optional tool schema summary
- optional response-format shape

Normalization rules:

- strip known volatile fields
- collapse repeated whitespace
- ignore timestamps, trace ids, random request ids
- ignore temporary file URLs if they are not part of semantic prompt identity

Recommended first version:

- use only `system` content + tool schema summary + first 1-2 stable messages
- avoid full-payload hashing
- avoid tokenizer-dependent fingerprinting at the gateway

### 9.3 Fingerprint Safety Notes

The fingerprint should be:

- one-way hashed
- non-reversible in practice
- never emitted with raw source material in logs

This does not eliminate side-channel risk entirely, but it significantly reduces accidental leakage from observability.

---

## 10. Upstream Pool Model

This design assumes each proxy model can define multiple upstream instances.

Example:

```yaml
proxy_config:
  upstreams:
    - name: mineru-a
      base_url: http://10.0.0.21:8001/v1
      enabled: true
      weight: 1
    - name: mineru-b
      base_url: http://10.0.0.22:8001/v1
      enabled: true
      weight: 1
```

Each upstream should represent a stable cache island.

Best case:

- one upstream entry maps to one vLLM engine process or one tightly controlled Ray Serve app endpoint

Less ideal:

- one upstream entry points to a separate load balancer that may still redistribute requests internally

The more stable the instance identity, the higher the cache-locality benefit.

---

## 11. Routing Algorithm

Recommended algorithm:

- use Rendezvous Hashing over the enabled and healthy upstream set

Why:

- deterministic across multiple gateway replicas
- low remapping churn when upstream membership changes
- no shared in-memory sticky map required for the common path

Selection flow:

1. Build candidate upstream list for the target model.
2. Remove disabled upstreams.
3. Remove unhealthy upstreams unless no healthy upstream remains.
4. Compute affinity score for each candidate:

```text
score = hash(affinity_key + upstream.name)
```

5. Rank by score.
6. If load-shedding is enabled, reject overloaded candidates.
7. Select the highest-ranked acceptable candidate.
8. If no affinity key is available, fall back to standard routing such as round-robin.

---

## 12. Soft Affinity and Overload Protection

Hard stickiness is not recommended.

Instead, affinity should be honored only when the preferred upstream is:

- enabled
- healthy
- below overload threshold

Suggested break conditions:

- upstream health check failed
- recent timeout/error rate exceeded threshold
- in-flight request count exceeded threshold
- moving average latency exceeded threshold

Suggested behavior:

- `affinity honored`: preferred upstream is selected
- `affinity broken`: a different healthy upstream is selected
- `affinity unavailable`: no valid affinity key, use standard routing

This is the main protection against "all similar tool requests collapse onto one replica".

---

## 13. Streaming Behavior

For `stream=true` requests:

- choose upstream once before connection establishment
- do not switch upstream mid-stream
- if stream setup fails before any bytes are returned, retry policy may choose another upstream
- if stream breaks after bytes are emitted, return failure to the client; no transparent migration

Reason:

- mid-stream failover breaks response semantics
- prefix-cache affinity is only meaningful at stream start

---

## 14. Isolation and Security Analysis

### 14.1 Why Cross-Session Sharing Is Not the Default

Cross-session sharing can be low-value and higher-risk when:

- prompt similarity across sessions is weak
- tenant isolation matters
- request timing or cache-hit behavior may leak weak signals

### 14.2 Security Risks if Isolation Is Too Broad

Potential risks:

- cache-sharing boundary does not align with tenant boundary
- routing/latency side channels reveal that a similar prefix was recently used
- explicit cache identifiers, if introduced later, are not access-controlled strongly enough

### 14.3 Recommended Boundary

Affinity computation should never intentionally cross:

- tenant
- workspace
- project

depending on the platform's true security boundary.

### 14.4 Recommended Safe Inputs

Good candidates for stable prefix reuse:

- common system prompts
- fixed tool schemas
- standardized response-format directives
- common app-level instructions within one tenant

Avoid treating these as shareable across sessions without extra care:

- user-uploaded files
- PII-heavy prompts
- secrets in system instructions
- tenant-specific private context mixed into prompt prefixes

---

## 15. Request Metadata Contract

This design should not require body changes.

Recommended inputs:

- authenticated principal identity
- tenant/workspace id
- model
- optional stable conversation id from request header

Suggested optional header:

```text
X-Conversation-ID
```

This header is not required for prefix-hash routing, but it can improve stability for true multi-turn conversations.

The gateway should not require cookies for affinity.

---

## 16. Observability

The system must expose enough data to validate whether affinity helps or harms.

Recommended log fields:

- `request_id`
- `public_model`
- `tenant_scope_hash`
- `prefix_fingerprint_hash`
- `affinity_enabled`
- `affinity_candidate_count`
- `selected_upstream`
- `affinity_honored`
- `affinity_break_reason`
- `status_code`
- `latency_ms`
- `stream`

Recommended metrics:

- `proxy_requests_total`
- `proxy_affinity_requests_total`
- `proxy_affinity_honored_total`
- `proxy_affinity_broken_total`
- `proxy_affinity_break_reason_total`
- `proxy_upstream_selected_total{model,instance}`
- `proxy_upstream_inflight{model,instance}`
- `proxy_prefix_cache_affinity_enabled_total{model}`

The key success metrics are not just routing statistics.
They should also include:

- TTFT
- prefill latency
- p95/p99 latency
- upstream timeout rate
- load skew by instance

---

## 17. Failure and Recovery Semantics

Recommended rules:

1. If preferred upstream is unhealthy before send, select the next-ranked healthy upstream.
2. If non-streaming request fails with connect/read timeout before response, retry may pick the next-ranked candidate.
3. If upstream returns a valid application-level response, do not second-guess it for affinity reasons.
4. If all upstreams are unavailable, return gateway error.
5. Do not attempt automatic fallback from proxy backend to local `vllm` backend.

This keeps semantics predictable and avoids "surprising success paths" that weaken correctness.

---

## 18. Hotspot Risk Analysis

### Concern

If many similar tool requests share the same prefix structure, prefix-hash routing may concentrate them on one upstream.

### Reality

Yes, this can happen.

This is not a reason to avoid affinity entirely.
It is a reason to keep affinity soft and to add overload escape hatches.

### Mitigations

1. Apply overload thresholds per upstream.
2. Break affinity when inflight or latency thresholds are exceeded.
3. Keep upstream pool size > 1 for sticky-enabled models.
4. Consider weighted rendezvous hashing if instances have different capacities.
5. Consider splitting extremely hot traffic classes at the product level rather than forcing one cache domain.

The design should prefer:

- better cache locality under normal load
- graceful degradation under hotspot conditions

not:

- perfect cache locality at any cost

---

## 19. Suggested Config Draft

```yaml
models:
  - name: mineru
    alias: mineru
    task: chat
    backend: vllm_openai_proxy
    served_model_name: mineru
    proxy_config:
      upstreams:
        - name: mineru-a
          base_url: http://10.0.0.21:8001/v1
          enabled: true
          weight: 1
        - name: mineru-b
          base_url: http://10.0.0.22:8001/v1
          enabled: true
          weight: 1
      timeout:
        connect_seconds: 3
        read_seconds: 180
        write_seconds: 30
        pool_seconds: 5
      retry:
        max_attempts: 2
        backoff_ms: 100
        retry_on_status: [502, 503, 504]
      streaming:
        enabled: true
        passthrough_sse: true
      headers_policy:
        pass_request_id: true
        forward_authorization: false
      affinity:
        enabled: true
        strategy: rendezvous_hash
        isolation_scope: tenant_model
        include_conversation_id: true
        break_on_unhealthy: true
        break_on_overload: true
        overload:
          max_inflight: 64
          max_latency_ms: 5000
          max_error_rate: 0.2
```

---

## 20. Proposed Implementation Phases

### Phase 1: Instance Pool Foundation

- add multi-upstream config
- add health state tracking
- add explicit upstream selection result to runtime context/logs

### Phase 2: Basic Prefix-Hash Affinity

- implement prefix fingerprinting
- implement rendezvous hash selection
- honor affinity only when upstream is healthy

### Phase 3: Overload-Aware Soft Affinity

- track per-upstream inflight and latency
- break affinity on overload
- expose observability fields and metrics

### Phase 4: Validation

- compare cache-hit-related latency before and after affinity
- verify no unacceptable hotspot amplification
- verify tenant isolation boundaries

---

## 21. Required Project Touch Points

Likely files to change later if this is implemented:

- `src/infer_nexus/catalog/models.py`
- `src/infer_nexus/runtime/types.py`
- `src/infer_nexus/runtime/dispatcher.py`
- `src/infer_nexus/runtime/executor.py`
- `src/infer_nexus/api/openai_routes.py`
- `tests/test_runtime.py`
- `tests/test_proxy_health_routing.py`
- `tests/test_proxy_streaming.py`

Additional docs likely needed:

- rollout guide
- observability guide
- security boundary note for affinity-enabled models

---

## 22. Final Recommendation

For this project, the best first implementation is:

- proxy-backend only
- multiple explicit upstream vLLM instances
- prefix-hash routing with rendezvous hashing
- tenant-scoped cache affinity
- soft stickiness with overload break

This balances:

- cache locality
- safety
- operational simplicity
- future extensibility

It also fits the current `infer-nexus` architecture much better than trying to force replica-level affinity inside local Ray Serve deployment routing.
