# KubeRay Serve-Native Gateway Ingress

The KubeRay runtime exposes one northbound endpoint without a standalone
Uvicorn Deployment:

```text
client -> infer-nexus-gateway NodePort :30800 -> Ray head Serve proxy :8000
       -> InferNexusGatewayIngress -> model Serve applications
```

`deploy/kuberay/serve-gateway.service.yaml` preserves the client-facing
`infer-nexus-gateway:30800` address and selects only the Ray head pod. It
exposes port 8000 exclusively; it must never publish the Ray Client, GCS,
dashboard, or metrics ports. Before applying it, verify the real head labels
with `kubectl -n ray get pods -l ray.io/node-type=head --show-labels` and
confirm both selector labels in the manifest.

`config/settings.k8s.yaml` explicitly configures the CPU-only, bounded
`InferNexusGatewayIngress` deployment. `service.workers` now affects only the
local Uvicorn debug entrypoint. `serve-deployer.job.yaml` starts Serve with
`HeadOnly`, submits every catalog-derived model application first, then the
Gateway application, and exits only after both are healthy.

The deployer is a one-shot RayJob, not an autoscaling companion for the
RayCluster. Re-run it only after an intentional catalog or runtime deployment
change. Updating the catalog does not reload existing Serve applications.

## Controlled rollout

The former controller used `proxy_location=Disabled`. A running controller
cannot be converted to `HeadOnly` merely by re-running the RayJob. In an
approved maintenance window, record the known-good revision and replace the
RayCluster (or fully shut down Serve after draining traffic), then submit the
model applications and Gateway again. Do not perform that reset as part of a
normal manifest validation.

After the cluster is available, run client-side dry-runs and diffs for only
`raycluster.recovery.yaml`, `serve-gateway.service.yaml`, and
`serve-deployer.job.yaml`; then verify `/healthz`, `/readyz`, `/v1/models`,
the platform catalog endpoint, a request for each enabled task, and a streaming
chat request through NodePort 30800.
