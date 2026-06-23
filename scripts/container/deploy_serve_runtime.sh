#!/usr/bin/env bash
set -euo pipefail

# Submit infer-nexus Ray Serve applications to an already-running Ray cluster.
#
# This container is intentionally job-like: it exits after all configured Serve
# applications report ready unless SERVE_DEPLOY_BLOCKING=1 is set.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

SETTINGS="${INFER_NEXUS_SETTINGS:-config/settings.yaml}"
RAY_ADDRESS="${RAY_ADDRESS:-ray-head:6379}"
PROXY_LOCATION="${SERVE_PROXY_LOCATION:-Disabled}"
READY_TIMEOUT_SECONDS="${SERVE_READY_TIMEOUT_SECONDS:-900}"
READY_POLL_INTERVAL_SECONDS="${SERVE_READY_POLL_INTERVAL_SECONDS:-2}"
SERVE_DEPLOY_BLOCKING="${SERVE_DEPLOY_BLOCKING:-0}"

args=(
  python
  scripts/run_serve_runtime.py
  --settings "${SETTINGS}"
  --ray-address "${RAY_ADDRESS}"
  --proxy-location "${PROXY_LOCATION}"
  --ready-timeout-seconds "${READY_TIMEOUT_SECONDS}"
  --ready-poll-interval-seconds "${READY_POLL_INTERVAL_SECONDS}"
)

if [[ "${SERVE_DEPLOY_BLOCKING}" == "1" ]]; then
  args+=(--blocking)
fi

exec "${args[@]}"
