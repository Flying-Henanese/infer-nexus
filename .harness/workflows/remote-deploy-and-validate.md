# Remote Deploy And Validate

Use this workflow after a code change that must be validated in the shared
accelerator-backed test environment. It is the required delivery path unless
the user explicitly asks for a different deployment method.

Connection and target-directory details are intentionally not versioned. Use
the approved internal test-environment profile.

CUDA is the default deployment target. Use the Ascend alternative for an Ascend
host; both follow the same publish, pull, recreate, and validation sequence.

## 1. Validate And Publish Locally

Run the smallest relevant local checks, then review, commit, and push:

```bash
git status --short
git diff --check
# Run the relevant tests or checks for the change.
git add <intended-files>
git commit -m "<concise change description>"
git push origin HEAD
git branch --show-current
git rev-parse HEAD
```

Do not deploy uncommitted local changes. If the commit or push fails, stop and
resolve that issue before using the test server.

## 2. Update The Remote Working Tree

Connect to the approved test server and update the same branch with a
fast-forward-only pull. Check for a clean working tree first so no remote-only
changes are overwritten. Confirm that the remote branch matches the published
local branch before pulling, then compare its resulting SHA with the published
commit before recreating services.

```bash
ssh <approved-test-server>
cd <deployment-directory>
git status --short
git branch --show-current
git pull --ff-only
git rev-parse HEAD
```

If the remote working tree is dirty or the pull cannot fast-forward, stop and
report the condition. Do not discard, reset, or overwrite remote changes
without explicit approval.

## 3. Recreate The Compose Deployment

Run the commands below from the remote repository root. Select one Compose file
for the target and keep that selection for every subsequent Compose command.

| Target | Compose file | Settings used by deployer/Gateway | Required pool variable |
| --- | --- | --- | --- |
| CUDA (default) | `docker-compose.yml` | `config/settings.compose.yaml` | `CUDA_VISIBLE_DEVICES` |
| Ascend | `ascend_deploy/docker-compose.yml` | `config/settings.ascend-compose.yaml` | `ASCEND_RT_VISIBLE_DEVICES` |

Default CUDA selection:

```bash
export INFER_NEXUS_COMPOSE_FILE=docker-compose.yml
```

For an Ascend host, use this selection instead:

```bash
export INFER_NEXUS_COMPOSE_FILE=ascend_deploy/docker-compose.yml
```

Load the selected target's approved pool and image/mount settings before
rendering Compose. The pool variable defines the platform's shared accelerator
budget. Set `MODEL_STORE_HOST_PATH` to the prepared host model directory;
an absolute path avoids ambiguity between the two Compose file locations.
Both variants mount that directory at `/models`. For Ascend driver/device
mounts, image dependencies, and device-group settings, use
`docs/ASCEND_DEPLOYMENT_CONFIGURATION.md` together with the checked-in Compose
file. Preserve any existing Compose project-name and environment-file options
throughout the sequence.

Render the selected configuration, then recreate its services so mounted source
and configuration use the pulled revision:

```bash
docker compose -f "$INFER_NEXUS_COMPOSE_FILE" config --quiet
docker compose -f "$INFER_NEXUS_COMPOSE_FILE" up -d --no-build --force-recreate
docker compose -f "$INFER_NEXUS_COMPOSE_FILE" ps --all
```

The checked-in Compose files mount source, scripts, and configuration. Reuse
the existing image with `--no-build` when only those files changed. Use
`docker compose -f "$INFER_NEXUS_COMPOSE_FILE" up -d --build --force-recreate`
when the selected image is missing or its Dockerfile/base-image inputs or
packaged runtime dependencies changed.

Both variants define four services. `log-init` is a root-only one-shot job that
prepares host log directory ownership before the non-root runtime services
start. `serve-deployer` is also one-shot; both jobs must exit with status `0`.
`ray-head` and `ray-worker` are the only long-running Compose containers;
the public Gateway is a CPU-only Ray Serve deployment reached through the
HeadOnly Serve HTTP proxy on `ray-head`.

## 4. Verify Runtime Health

Wait for `log-init` to exit `0`, `ray-head` and `ray-worker` to become healthy,
and `serve-deployer` to exit `0`. Then run read-only API checks from the server;
these are external requests from the host into the Compose deployment:

```bash
docker compose -f "$INFER_NEXUS_COMPOSE_FILE" ps --all
curl --fail --silent --show-error http://127.0.0.1:8000/healthz
curl --fail --silent --show-error http://127.0.0.1:8000/readyz
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models
```

For inference-path changes, run a minimal request against an enabled model
alias. The active validation catalog exposes `qwen3.5-9b`; do not copy the
disabled `qwen3.5-27b` example from `docs/DEPLOYMENT_CHECKLIST.md`.

CUDA/A100 has recorded end-to-end HTTP/SSE validation. The checked-in Ascend
configuration still requires equivalent validation on an NPU host; a successful
configuration render alone is not evidence of real inference support.

## 5. Report Completion

Only report the change as deployed after recording:

- local commit SHA and successful push
- matching remote branch/SHA and successful fast-forward pull result
- target platform and selected Compose/settings files
- Compose status, including both `log-init` and `serve-deployer` exit codes
- health, readiness, and model-discovery results
- the result of any change-specific smoke test

If validation fails, collect only relevant read-only logs such as
`tail -n 200 logs/<service>/container.log` and files below
`logs/<service>/ray/session_latest/logs/`. Docker Compose logs remain a fallback
for failures before the host log wrapper starts; use
`docker compose -f "$INFER_NEXUS_COMPOSE_FILE" logs --tail 200 <service>`.
Do not make destructive recovery changes without user approval.
