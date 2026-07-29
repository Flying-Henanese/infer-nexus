# Remote Deploy And Validate

Use this workflow after a code change that must be validated in the shared
accelerator-backed test environment. It is the required delivery path unless
the user explicitly asks for a different deployment method.

Connection and target-directory details are intentionally not versioned. Use
the approved internal test-environment profile.

## 1. Validate And Publish Locally

Run the smallest relevant local checks, then review, commit, and push:

```bash
git status --short
git diff --check
# Run the relevant tests or checks for the change.
git add <intended-files>
git commit -m "<concise change description>"
git push origin HEAD
```

Do not deploy uncommitted local changes. If the commit or push fails, stop and
resolve that issue before using the test server.

## 2. Update The Remote Working Tree

Connect to the approved test server and update the same branch with a
fast-forward-only pull. Check for a clean working tree first so no remote-only
changes are overwritten.

```bash
ssh <approved-test-server>
cd <deployment-directory>
git status --short
git pull --ff-only
```

If the remote working tree is dirty or the pull cannot fast-forward, stop and
report the condition. Do not discard, reset, or overwrite remote changes
without explicit approval.

## 3. Recreate The Compose Deployment

From the remote repository root, rebuild the runtime image and recreate the
Compose services so the deployed processes use the pulled revision:

```bash
docker compose up -d --build --force-recreate
docker compose ps --all
```

`serve-deployer` is a one-shot deployment job and normally exits with status
`0`; `ray-head`, `ray-worker`, and `gateway` are the long-running services.

## 4. Verify Runtime Health

Wait for `gateway` and `ray-head` to become healthy, then run read-only API
checks from the server:

```bash
docker compose ps --all
curl --fail --silent --show-error http://127.0.0.1:8000/healthz
curl --fail --silent --show-error http://127.0.0.1:8000/readyz
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models
```

For inference-path changes, also run a minimal request against an enabled model
alias, following `docs/DEPLOYMENT_CHECKLIST.md`.

## 5. Report Completion

Only report the change as deployed after recording:

- local commit SHA and successful push
- remote fast-forward pull result
- Compose status, including the `serve-deployer` exit code
- health, readiness, and model-discovery results
- the result of any change-specific smoke test

If validation fails, collect only relevant read-only logs such as
`docker compose logs --tail=200 <service>` and report the failing step. Do not
make destructive recovery changes without user approval.
