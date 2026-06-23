#!/usr/bin/env bash
set -euo pipefail

# Start the FastAPI gateway in a single container.
#
# Uvicorn still owns the public listener and manages its worker child processes.
# This avoids adding an ingress proxy hop for the first compose architecture.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

SETTINGS="${INFER_NEXUS_SETTINGS:-config/settings.yaml}"
GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"
GATEWAY_WORKERS="${GATEWAY_WORKERS:-}"
GATEWAY_RELOAD="${GATEWAY_RELOAD:-0}"

args=(
  python
  scripts/run_gateway.py
  --settings "${SETTINGS}"
  --host "${GATEWAY_HOST}"
  --port "${GATEWAY_PORT}"
)

if [[ -n "${GATEWAY_WORKERS}" ]]; then
  args+=(--workers "${GATEWAY_WORKERS}")
fi

if [[ "${GATEWAY_RELOAD}" == "1" ]]; then
  args+=(--reload)
fi

exec "${args[@]}"
