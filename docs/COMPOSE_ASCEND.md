# Compose Ascend Startup

This document describes the compose-based Ascend startup shape. It replaces the
all-in-one `scripts/start_minimal_ascend.sh` workflow for production-like runs,
while keeping the gateway listener model unchanged.

## Process Shape

```text
client
  -> gateway container :8000
       -> Uvicorn master
            -> gateway worker process 1
            -> gateway worker process N
       -> Ray Serve DeploymentHandle
            -> Ray head / Serve controller
                 -> Ray worker container
                      -> Ray Serve model replicas
```

The public `8000` port belongs to the `gateway` container. Uvicorn still owns
that listener and manages its worker child processes inside the container. This
avoids adding an nginx/haproxy network hop for the first compose architecture.

## Services

### `ray-head`

Starts the Ray head node and Ray dashboard. By default, it advertises no NPU
resources so it stays focused on Ray control-plane work.

Override `RAY_NUM_NPUS` only when model replicas should also run on the head
container.

### `ray-worker`

Joins the Ray cluster at `ray-head:6379` and advertises Ascend NPU resources.
The default compose file derives the NPU resource count from
`ASCEND_RT_VISIBLE_DEVICES`.

Use multiple worker services, or scale this service if the target compose
environment supports it and the device assignment is adjusted per container.

### `serve-deployer`

Runs `scripts/run_serve_runtime.py` once. It connects to `ray-head:6379`, starts
Ray Serve, submits the configured model deployments, waits until they are ready,
and exits.

This is a deployment job, not a request-serving process. After it exits, Ray
Serve controller and replica processes continue to live inside the Ray cluster.

### `gateway`

Runs `scripts/run_gateway.py` and binds host port `8000`. It connects to Ray via
`INFER_NEXUS_RAY_ADDRESS=ray-head:6379` when it first resolves Serve deployment
handles. The `/healthz` liveness endpoint only checks that the gateway process
is responsive; deeper Ray/Serve readiness can be added to `/readyz` later.

Set `GATEWAY_WORKERS` to control Uvicorn worker count inside this one gateway
container.

## Startup

```bash
docker compose up ray-head ray-worker serve-deployer gateway
```

For a long-running deployment, start all services:

```bash
docker compose up -d
```

Check the gateway:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models
```

## Current Supervisor Boundary

There is no independent `supervisor` process in the current codebase.

The closest old equivalent was `scripts/start_minimal_ascend.sh`, which started
Ray, deployed Serve applications, launched the gateway, wrote pid files, and
performed readiness checks. In compose mode those responsibilities are split:

- Docker Compose handles container process lifecycle and restart policy.
- Uvicorn handles gateway child worker processes inside the `gateway` container.
- Ray handles Serve controller and model replica processes inside the Ray
  cluster.
- `serve-deployer` submits deployments once.

A future supervisor/reconciler container can be added later if the system needs
continuous reconciliation, such as redeploying missing Serve applications after
Ray state loss.
