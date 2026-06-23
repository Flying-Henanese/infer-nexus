#!/usr/bin/env bash
set -euo pipefail

# Start a Ray head node for the compose-based Ascend deployment.
#
# The head owns Ray control-plane services. By default it advertises no NPU
# resources; attach one or more ray-worker containers to host model replicas.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-0.0.0.0}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-}"
RAY_NUM_NPUS="${RAY_NUM_NPUS:-0}"
RAY_BLOCK="${RAY_BLOCK:-1}"

if ! command -v ray >/dev/null 2>&1; then
  echo "Missing 'ray' in PATH. The container image must include Ray." >&2
  exit 1
fi

args=(
  start
  --head
  --disable-usage-stats
  --port "${RAY_PORT}"
  --dashboard-host "${RAY_DASHBOARD_HOST}"
  --dashboard-port "${RAY_DASHBOARD_PORT}"
)

if [[ -n "${RAY_NUM_CPUS}" ]]; then
  args+=(--num-cpus "${RAY_NUM_CPUS}")
fi

if [[ "${RAY_NUM_NPUS}" != "0" ]]; then
  args+=(--resources "{\"NPU\": ${RAY_NUM_NPUS}}")
fi

if [[ "${RAY_BLOCK}" == "1" ]]; then
  args+=(--block)
fi

exec ray "${args[@]}"
