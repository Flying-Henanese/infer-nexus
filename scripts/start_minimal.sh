#!/usr/bin/env bash
set -euo pipefail

# One-click minimal bring-up:
# 1) (optional) uv sync extras
# 2) start/attach Ray head
# 3) deploy Serve runtime app
# 4) start FastAPI gateway
#
# Logs:   .infer-nexus/logs/{ray,serve_runtime,gateway}.log
# PIDs:   .infer-nexus/pids/{serve_runtime,gateway}.pid
# Status: .infer-nexus/STATUS

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${ROOT_DIR}/.infer-nexus"
LOG_DIR="${STATE_DIR}/logs"
PID_DIR="${STATE_DIR}/pids"

SETTINGS="config/settings.yaml"
RAY_ADDRESS="auto"
PROXY_LOCATION="Disabled"
INSTALL=1
RELOAD=0
CUDA_VISIBLE_DEVICES_VALUE=""
NUM_GPUS=""

usage() {
  cat <<'EOF'
Usage: scripts/start_minimal.sh [options]

Options:
  --settings PATH            Path to settings.yaml (default: config/settings.yaml)
  --no-install               Skip "uv sync --extra serve --extra vllm"
  --install-artifacts        Also install artifacts extra (adds --extra artifacts)
  --ray-address ADDR         Ray address passed to run_serve_runtime.py (default: auto)
  --num-gpus N               If starting local Ray head, set --num-gpus N (recommended with CUDA_VISIBLE_DEVICES)
  --cuda-visible-devices CSV If non-empty, export CUDA_VISIBLE_DEVICES before starting Ray and runtime
  --proxy-location VALUE     Serve proxy location (default: Disabled)
  --reload                   Enable uvicorn reload for gateway (dev only)
  -h, --help                 Show this help

Examples:
  scripts/start_minimal.sh --cuda-visible-devices 0,1,2,3 --num-gpus 4
  scripts/start_minimal.sh --no-install --ray-address auto
EOF
}

INSTALL_ARTIFACTS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --settings) SETTINGS="$2"; shift 2;;
    --no-install) INSTALL=0; shift;;
    --install-artifacts) INSTALL_ARTIFACTS=1; shift;;
    --ray-address) RAY_ADDRESS="$2"; shift 2;;
    --num-gpus) NUM_GPUS="$2"; shift 2;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2;;
    --proxy-location) PROXY_LOCATION="$2"; shift 2;;
    --reload) RELOAD=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

mkdir -p "${LOG_DIR}" "${PID_DIR}"

cd "${ROOT_DIR}"

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Missing 'uv' in PATH. Install uv first." >&2
  exit 1
fi

if [[ "${INSTALL}" -eq 1 ]]; then
  if [[ "${INSTALL_ARTIFACTS}" -eq 1 ]]; then
    uv sync --extra serve --extra vllm --extra artifacts
  else
    uv sync --extra serve --extra vllm
  fi
fi

if ! command -v ray >/dev/null 2>&1; then
  echo "Missing 'ray' in PATH. Did you run: uv sync --extra serve ?" >&2
  exit 1
fi

ray_is_running() {
  # "ray status" exits non-zero if it can't connect to a running head.
  ray status >/dev/null 2>&1
}

start_ray_head_if_needed() {
  if ray_is_running; then
    return 0
  fi

local args=(start --head --disable-usage-stats)
  if [[ -n "${NUM_GPUS}" ]]; then
    args+=(--num-gpus "${NUM_GPUS}")
  fi

  # Ray prints useful diagnostics; keep them in a file.
  # Note: Ray daemonizes; this command returns quickly.
  (ray "${args[@]}" >"${LOG_DIR}/ray.log" 2>&1) || {
    echo "Failed to start Ray head. See ${LOG_DIR}/ray.log" >&2
    exit 1
  }

  # Wait briefly for head to become reachable.
  local i
  for i in {1..40}; do
    if ray_is_running; then
      return 0
    fi
    sleep 0.25
  done

  echo "Ray did not become ready in time. See ${LOG_DIR}/ray.log" >&2
  exit 1
}

pid_is_running() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" >/dev/null 2>&1
}

write_status() {
  local msg="$1"
  printf '%s\n' "${msg}" >"${STATE_DIR}/STATUS"
}

start_ray_head_if_needed

SERVE_PID_FILE="${PID_DIR}/serve_runtime.pid"
GATEWAY_PID_FILE="${PID_DIR}/gateway.pid"

if pid_is_running "${SERVE_PID_FILE}"; then
  echo "Serve runtime already running (pid $(cat "${SERVE_PID_FILE}"))."
else
  (
    exec uv run python scripts/run_serve_runtime.py \
      --settings "${SETTINGS}" \
      --ray-address "${RAY_ADDRESS}" \
      --proxy-location "${PROXY_LOCATION}"
  ) >"${LOG_DIR}/serve_runtime.log" 2>&1 &
  echo $! >"${SERVE_PID_FILE}"
fi

if pid_is_running "${GATEWAY_PID_FILE}"; then
  echo "Gateway already running (pid $(cat "${GATEWAY_PID_FILE}"))."
else
  gateway_args=(python scripts/run_gateway.py --settings "${SETTINGS}")
  if [[ "${RELOAD}" -eq 1 ]]; then
    gateway_args+=(--reload)
  fi

  (exec uv run "${gateway_args[@]}") >"${LOG_DIR}/gateway.log" 2>&1 &
  echo $! >"${GATEWAY_PID_FILE}"
fi

write_status "started"

echo "Started."
echo "Logs: ${LOG_DIR}"
echo "PIDs: ${PID_DIR}"
echo "Try:"
echo "  curl http://127.0.0.1:8000/healthz"
echo "  curl http://127.0.0.1:8000/v1/models"

