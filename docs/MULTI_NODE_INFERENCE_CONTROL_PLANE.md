# Multi-node Inference Control Plane

This document describes the target direction for evolving `infer-nexus` from a
single-machine shared inference service into a multi-node inference control
plane.

The long-term target is one Ray Cluster that contains heterogeneous accelerator
nodes, such as T4 and A100 servers. `infer-nexus` should expose one unified
OpenAI-compatible gateway, while Ray schedules model replicas onto the right
hardware according to declarative model placement requirements.

## 1. Target Outcome

The target runtime shape is:

```text
Client
  -> infer-nexus FastAPI gateway
  -> RuntimeDispatcher / RuntimeExecutor
  -> Ray Serve deployment handle
  -> Ray Cluster
       -> T4 nodes for small models, embeddings, rerankers, light chat
       -> A100 nodes for large models, long context, high-throughput chat
       -> future Ascend or CPU nodes through the same resource abstraction
```

The important architectural rule is:

```text
infer-nexus declares what a model needs.
Ray decides where the model replica can run.
The gateway never hard-codes a physical machine, GPU id, port, or process id.
```

This keeps the public API stable while the execution plane grows from one
machine to many machines.

## 2. Current Baseline

The current project already has several pieces that should be preserved:

- The FastAPI gateway owns public ingress and OpenAI-compatible API behavior.
- Local `vllm` models are reached through Ray Serve deployment handles.
- `DeploymentFactory` converts catalog model declarations into Ray Serve
  deployment kwargs.
- `gpu_per_replica` is already mapped to Ray CUDA `num_gpus` or custom Ascend
  `NPU` resources, depending on `cluster.inference_device_type`.
- `deployment_config.ray_actor_options` can override or extend generated Ray
  actor options.
- `deployment_config.request_router_config` is already passed through to Ray
  Serve for supported local chat deployments.

The current gap is that model resource declarations do not yet express
heterogeneous CUDA hardware constraints, such as "this model must run on T4" or
"this model must run on A100".

## 3. Long-term Scheduling Model

The long-term direction is a single Ray Cluster with explicitly labeled
accelerator resources.

Conceptually, Ray workers should expose resources like:

```text
T4 worker:
  GPU: 4
  accelerator_type:nvidia: 4
  accelerator_model:t4: 4

A100 worker:
  GPU: 8
  accelerator_type:nvidia: 8
  accelerator_model:a100: 8
```

`infer-nexus` should then generate Ray Serve actor options that combine normal
GPU demand with hard placement resources:

```python
{
    "num_gpus": 1,
    "resources": {
        "accelerator_model:t4": 1,
    },
}
```

For larger models:

```python
{
    "num_gpus": 4,
    "resources": {
        "accelerator_model:a100": 4,
    },
}
```

The exact Ray resource names should be centralized in `infer-nexus` config, not
scattered through model entries or runtime code.

## 4. Component Adoption Roadmap

The multi-node roadmap should introduce infrastructure components only when the
previous layer has proved useful. The immediate target is not "use every
component", but "add the minimum component that proves the next control-plane
capability".

### 4.1 Already In The Architecture: FastAPI Gateway And Ray Serve

These components stay in the first-class runtime path:

- FastAPI remains the public API gateway.
- Ray Serve remains the local model deployment runtime.
- Serve handles remain the internal invocation path.
- `vllm` remains the first local inference backend.

This layer is already present in the current design and should be preserved
while multi-node scheduling is added.

### 4.2 Phase 1: Bare-metal Ray Cluster

Introduce a real multi-node Ray Cluster before introducing Kubernetes.

Purpose:

- Validate Ray head and worker startup across multiple servers.
- Validate that T4 and A100 workers can join one cluster.
- Validate Ray custom resource labels for accelerator class selection.
- Validate that Ray Serve deployments land on the intended hardware.
- Validate OpenAI-compatible chat, streaming, embedding, and rerank behavior
  through the existing gateway.

Components introduced:

- Ray head node.
- Ray worker nodes.
- Ray Serve on the cluster.
- Custom Ray resources such as `accelerator_model:t4` and
  `accelerator_model:a100`.

Components intentionally not introduced yet:

- Kubernetes.
- KubeRay.
- AIBrix-style control plane.
- Automatic placement.

This is the most important near-term stage because it proves that
`infer-nexus` can become a unified manager for heterogeneous hardware without
changing the public API contract.

### 4.3 Phase 2: Standardized Runtime Images

After bare-metal Ray Cluster validation works, standardize the runtime
environment.

Purpose:

- Make T4 and A100 workers reproducible.
- Reduce drift across CUDA, driver, vLLM, PyTorch, Ray, and Python versions.
- Make worker bootstrap repeatable.
- Prepare the system for Kubernetes later.

Components introduced:

- CUDA image for NVIDIA GPU workers.
- vLLM runtime image.
- `infer-nexus` gateway/runtime image.
- Shared model storage or an internal model artifact mirror.

KubeRay should still wait until the bare-metal cluster and images are stable.
If the runtime image is not stable, KubeRay will make failures harder to
understand rather than easier to operate.

### 4.4 Phase 3: K8s And KubeRay

Introduce K8s and KubeRay only after the bare-metal Ray Cluster path and runtime
images have been validated.

Purpose:

- Let Kubernetes own pod scheduling, service discovery, restart behavior, and
  rollout mechanics.
- Let KubeRay own RayCluster lifecycle on Kubernetes.
- Keep Ray responsible for inference actor scheduling inside the RayCluster.
- Keep `infer-nexus` responsible for model catalog, deployment intent, routing,
  and control-plane APIs.

Components introduced:

- Kubernetes cluster.
- NVIDIA GPU device plugin, and later NPU device plugins if Ascend is added.
- KubeRay operator.
- KubeRay `RayCluster` custom resources.
- Kubernetes Services and networking for the gateway and Ray control endpoints.

Important boundary:

```text
KubeRay manages Ray clusters on Kubernetes.
Ray Serve manages model deployments inside Ray.
infer-nexus manages model intent, placement policy, API, and routing.
```

KubeRay should not be treated as an inference framework. It is the lifecycle
manager for Ray on K8s.

### 4.5 Phase 4: Platform Control-plane Capabilities

After the system can deploy models to the intended accelerator class, add the
higher-level platform features incrementally.

Purpose:

- Make model deployment auditable.
- Make resource usage visible.
- Make routing and scaling policy explicit.
- Prepare for enterprise deployment patterns.

Components and capabilities introduced:

- Model Registry.
- Deployment Registry.
- Resource inventory API.
- Deployment planning and dry-run API.
- Placement validation.
- Health and status reconciliation.
- Autoscaling policy.
- Canary or staged rollout policy.
- Optional AIBrix-inspired routing, scaling, and control-plane ideas.

AIBrix should be treated as a reference for platform capabilities, not as
something to copy wholesale in the first pass.

### 4.6 Phase 5: Automatic Placement

Automatic placement should be built only after manual placement, resource
inventory, and deployment planning are reliable.

Purpose:

- Infer T4/A100 placement from model metadata.
- Estimate memory from parameter count, quantization, and `max_model_len`.
- Recommend tensor parallel size.
- Reject impossible deployments before they reach Ray Serve.

Components introduced:

- Model profile metadata.
- Memory estimator.
- Placement recommendation engine.
- Optional benchmark profile store.

The initial automatic placement output should still be visible and auditable.
The control plane should explain why it selected T4, A100, or rejected the
deployment.

## 5. Reference Projects And Lessons

The following projects are useful references, but they should influence
different layers of the roadmap.

### 5.1 KubeRay

Reference:

- <https://github.com/ray-project/kuberay>
- <https://docs.ray.io/en/latest/cluster/kubernetes/>

KubeRay is useful for the Kubernetes phase, not for the first bare-metal
multi-node validation phase.

What to borrow:

- Treat `RayCluster` as the Kubernetes lifecycle object for Ray.
- Use KubeRay to manage RayCluster creation, deletion, autoscaling, and fault
  recovery once `infer-nexus` runs on Kubernetes.
- Study `RayService` for zero-downtime Ray Serve application upgrades after
  the Serve deployment model is stable.
- Keep KubeRay observability, Service, Ingress, and ecosystem integration in
  the K8s phase.

What not to copy early:

- Do not introduce KubeRay before the bare-metal Ray Cluster path proves that
  T4/A100 resource labels and Ray Serve placement work.
- Do not treat KubeRay as the inference control plane. It manages Ray on K8s;
  it does not replace `infer-nexus` model catalog, placement policy, routing,
  or OpenAI-compatible gateway ownership.

How it changes this roadmap:

- It confirms that Phase 3 should be "K8s + KubeRay", not "K8s only".
- It suggests that later `infer-nexus` deployment specs may need a clean export
  path into KubeRay `RayCluster` / `RayService` resources.
- It reinforces the boundary:

```text
KubeRay owns RayCluster lifecycle.
Ray Serve owns model replicas inside Ray.
infer-nexus owns model intent, placement, API, routing, and platform policy.
```

### 5.2 BentoML OpenLLM

Reference:

- <https://github.com/bentoml/OpenLLM>

OpenLLM is useful as a model-serving product reference, but it should not be
treated as the scheduling substrate for `infer-nexus`.

What to borrow:

- Simple local developer workflow: one command should make a model runnable.
- OpenAI-compatible API as the default user-facing contract.
- A model repository concept that describes runnable models, their names,
  required GPU class, and startup command.
- Model metadata that includes parameter size and required GPU memory. This is
  directly relevant to the future automatic placement phase.
- BentoML/Bento-style packaging ideas for reproducible model artifacts and
  deployment bundles.

What not to copy directly:

- Do not make `infer-nexus` a CLI-only single-model launcher.
- Do not route around the existing FastAPI gateway and Ray Serve handle path.
- Do not rely on OpenLLM's cloud or BentoCloud deployment model for the internal
  multi-node control plane.

How it changes this roadmap:

- It strengthens Phase 2's focus on standardized runtime images and model
  packaging.
- It suggests adding a later Model Registry / model profile layer with fields
  such as parameter count, recommended GPU memory, supported tasks,
  quantization, and context length.
- It provides a useful reference for future automatic placement because its
  model list exposes a practical mapping from model family/size to required GPU
  memory.

## 6. Phase 1: Manual Placement

The first implementation phase should use explicit placement in model config.
This is intentionally manual. It is easier to verify and avoids guessing model
memory requirements too early.

Example target model declarations:

```yaml
models:
  - name: qwen2.5-7b-instruct
    alias: qwen-7b
    task: chat
    backend: vllm
    model_path: Qwen/Qwen2.5-7B-Instruct
    dtype: bfloat16
    tensor_parallel_size: 1
    max_model_len: 8192
    cpu_per_replica: 4
    gpu_per_replica: 1
    min_replicas: 1
    max_replicas: 2
    placement:
      accelerator_type: nvidia
      accelerator_model: t4
      accelerator_count: 1

  - name: qwen2.5-72b-instruct
    alias: qwen-72b
    task: chat
    backend: vllm
    model_path: Qwen/Qwen2.5-72B-Instruct
    dtype: bfloat16
    tensor_parallel_size: 4
    max_model_len: 32768
    cpu_per_replica: 8
    gpu_per_replica: 4
    min_replicas: 1
    max_replicas: 1
    placement:
      accelerator_type: nvidia
      accelerator_model: a100
      accelerator_count: 4
```

Phase 1 should treat `placement.accelerator_model` as a hard requirement. If no
matching Ray resources exist, deployment should fail clearly during planning or
Serve deployment, instead of silently falling back to the wrong hardware.

## 7. Proposed Config Objects

### 7.1 Cluster Resource Labels

`config/settings.yaml` should eventually define how high-level placement values
map to Ray resource names:

```yaml
cluster:
  inference_device_type: cuda
  accelerator_resource_labels:
    type_prefix: accelerator_type
    model_prefix: accelerator_model
```

This lets the model catalog stay hardware-oriented while the Ray implementation
details stay configurable.

### 7.2 Model Placement

Add a catalog model field:

```yaml
placement:
  accelerator_type: nvidia
  accelerator_model: t4
  accelerator_count: 1
  policy: required
```

Initial semantics:

- `accelerator_type`: broad hardware family, such as `nvidia`, `ascend`, or
  `cpu`.
- `accelerator_model`: concrete class, such as `t4`, `a100`, or `910b`.
- `accelerator_count`: number of accelerator units required per replica.
- `policy`: initially only `required`; future values may include `preferred`
  and `auto`.

During Phase 1, `accelerator_count` should usually equal `gpu_per_replica` for
CUDA models. Keeping both fields explicit makes the migration less disruptive:

- `gpu_per_replica` remains the Ray/vLLM capacity demand.
- `placement.accelerator_count` expresses the hardware-label demand.

If they diverge, validation should require a clear reason or reject the config.

## 8. Runtime Mapping

The first code path to extend is:

```text
ModelConfig
  -> DeploymentFactory.build_spec(...)
  -> DeploymentSpec.ray_actor_options
  -> serve.deployment(..., ray_actor_options=...)
```

For CUDA models with placement:

```text
gpu_per_replica: 1
placement.accelerator_model: t4

maps to:

ray_actor_options:
  num_gpus: 1
  resources:
    accelerator_model:t4: 1
```

For A100 tensor-parallel models:

```text
gpu_per_replica: 4
tensor_parallel_size: 4
placement.accelerator_model: a100

maps to:

ray_actor_options:
  num_gpus: 4
  resources:
    accelerator_model:a100: 4
```

Implementation rule:

- Keep public request routing unchanged.
- Keep Serve handle invocation unchanged.
- Extend only deployment planning and validation first.
- Do not introduce gateway-side HTTP calls between internal components.

## 9. Node Bootstrap Requirement

The Ray Cluster must expose matching resources when nodes join.

Illustrative examples:

```bash
# T4 worker example
ray start \
  --address="$RAY_HEAD_ADDRESS" \
  --num-gpus=4 \
  --resources='{"accelerator_type:nvidia": 4, "accelerator_model:t4": 4}'
```

```bash
# A100 worker example
ray start \
  --address="$RAY_HEAD_ADDRESS" \
  --num-gpus=8 \
  --resources='{"accelerator_type:nvidia": 8, "accelerator_model:a100": 8}'
```

Operational requirements:

- T4 and A100 nodes must use compatible Ray, Python, vLLM, PyTorch, CUDA, and
  driver versions when they join the same Ray Cluster.
- Model storage paths must be visible from the nodes that may host the model.
- Tensor parallel models should be scheduled only where enough same-class GPU
  resources exist.
- First bring-up should prefer one replica per physical device or one model per
  tensor-parallel group before reintroducing fractional sharing.

## 10. Future: Automatic Placement

Manual placement is the first milestone. Automatic placement should be a later
layer built on the same fields.

Future model metadata:

```yaml
model_profile:
  parameter_count_b: 7
  max_model_len: 8192
  quantization: none
  task: chat
```

Future placement:

```yaml
placement:
  policy: auto
  constraints:
    latency: normal
    cost: low
```

The control plane can then estimate:

- model weight memory
- KV cache memory from `max_model_len`
- tensor parallel requirement
- task-specific memory pressure
- expected throughput class
- candidate accelerator models

Example future decisions:

```text
7B chat, 8k context        -> T4 if capacity exists
8B embedding              -> T4 or shared A100 fallback
32B chat, 30k context     -> A100
72B chat, long context    -> A100 tensor parallel group
vision/OCR model          -> hardware class chosen from benchmark profile
```

Automatic placement should not be introduced until manual placement, resource
visibility, health reporting, and deployment reconciliation are reliable.

## 11. Control-plane Capabilities To Add Later

Once manual placement works, the control plane can grow in this order:

1. Resource inventory API
   - Show Ray cluster resources grouped by accelerator type and model.
   - Show allocated, available, and pending placement demand.

2. Deployment planning API
   - Dry-run a model deployment and show the generated Ray actor options.
   - Explain why a model can or cannot fit the current cluster.

3. Placement validation
   - Reject impossible configs before Serve deployment.
   - Detect mismatches between `tensor_parallel_size`, `gpu_per_replica`, and
     `placement.accelerator_count`.

4. Automatic placement recommendation
   - Suggest T4 or A100 from model metadata.
   - Keep the final deployment explicit and auditable at first.

5. Runtime routing policy
   - Add fallback only when the model has been deployed in multiple compatible
     placements.
   - Do not silently move a model to a different accelerator class while a
     request is in flight.

## 12. Suggested Codex Iteration Tasks

Use this section to create future implementation tasks.

### Task 1: Add Placement Schema

Goal:

- Add a `PlacementConfig` model under the catalog schema.
- Add `placement: PlacementConfig | None` to `ModelConfig`.
- Validate `policy: required` and the supported accelerator fields.

Likely files:

- `src/infer_nexus/catalog/models.py`
- `src/infer_nexus/core/schemas.py` if API-facing schema output needs placement
  visibility
- `tests/test_model_store.py`
- `tests/test_runtime.py`

Acceptance:

- Existing `config/models.yaml` still loads without placement.
- A model entry with `placement.accelerator_model: t4` loads successfully.
- Invalid placement policy or negative accelerator count fails clearly.

### Task 2: Map Placement To Ray Actor Options

Goal:

- Extend `DeploymentFactory` to merge generated placement resources into
  `ray_actor_options.resources`.
- Preserve existing `num_gpus` behavior for CUDA.
- Preserve existing `NPU` behavior for Ascend.

Likely files:

- `src/infer_nexus/runtime/deployments.py`
- `tests/test_runtime.py`

Acceptance:

- T4 placement generates `resources: {"accelerator_model:t4": 1}`.
- A100 placement with `gpu_per_replica: 4` generates
  `resources: {"accelerator_model:a100": 4}`.
- Existing explicit `deployment_config.ray_actor_options.resources` are merged
  predictably or rejected on conflict.

### Task 3: Add Deployment Plan Visibility

Goal:

- Expose or print deployment plans that include placement resources.
- Make it easy to inspect whether a model will target T4 or A100 before
  deployment.

Likely files:

- `src/infer_nexus/runtime/serve_app.py`
- `src/infer_nexus/api/platform_routes.py`
- `tests/test_api.py`
- `tests/test_runtime.py`

Acceptance:

- A dry-run or platform status endpoint shows generated `ray_actor_options`.
- The output includes model name, deployment name, GPU demand, and placement
  resources.

### Task 4: Multi-node Smoke Test Guide

Goal:

- Document a minimal single Ray Cluster test with one T4-labeled worker and one
  A100-labeled worker.
- Include the expected model config and verification commands.

Likely files:

- `docs/DEPLOYMENT_CHECKLIST.md`
- `docs/MODELS_YAML_CONFIGURATION_MANUAL.md`
- optional script under `scripts/`

Acceptance:

- An operator can start the Ray head, join T4/A100 workers, deploy models, and
  verify that each deployment lands on the intended hardware class.

### Task 5: Automatic Placement Design

Goal:

- Add a design-only follow-up for deriving placement from model metadata.
- Do not implement automatic placement until manual placement is validated.

Likely files:

- This document
- `docs/ARCHITECTURE.md`
- future model registry or model profile docs

Acceptance:

- The design defines required metadata: parameter count, context length,
  quantization, task type, and benchmark profile.
- The design explains when to choose T4, A100, or reject deployment.

## 13. Non-goals For The First Phase

Do not include these in the first implementation pass:

- Kubernetes or KubeRay.
- Multiple Ray Clusters as the primary architecture.
- Automatic model placement.
- Live migration of already-running models between accelerator classes.
- Cost-aware routing.
- Multi-tenant quota and billing.
- Gateway-side per-request hardware selection for a model that is only deployed
  once.

These are valid future capabilities, but they depend on the manual placement
foundation first.

## 14. Success Criteria

The first useful milestone is complete when:

```text
One Ray Cluster contains T4 and A100 workers.
config/models.yaml declares one small model for T4 and one large model for A100.
infer-nexus generates Ray Serve deployments with the expected resource labels.
Ray schedules each model onto the intended hardware class.
The gateway serves both models through the same OpenAI-compatible API.
Streaming chat still works through the existing Serve handle path.
```

This milestone proves the core platform direction without requiring automatic
placement, KubeRay, or AIBrix-style platform features.
