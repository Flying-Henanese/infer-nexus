#!/usr/bin/env bash
set -euo pipefail

# Minimal Ascend NPU bootstrap for Infer Nexus.
#
# This script mirrors start_minimal.sh, but uses Ascend/NPU device visibility
# and Ray custom NPU resources instead of CUDA_VISIBLE_DEVICES/--num-gpus.
#
# Host/container prerequisites are still external to this script:
# - CANN/driver/toolkit/runtime must already be installed and sourced.
# - Containers must be started with Ascend devices mounted, for example:
#   /dev/davinci*, /dev/davinci_manager, /dev/devmm_svm, /dev/hisi_hdc.
# - The container image is expected to already include the required Python
#   dependencies for Infer Nexus on Ascend.
#
# Logs:   .infer-nexus/logs/{ray,serve_runtime,gateway}.log
# PIDs:   .infer-nexus/pids/{serve_runtime,gateway}.pid
# Status: .infer-nexus/STATUS

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STATE_DIR="${ROOT_DIR}/.infer-nexus"
LOG_DIR="${STATE_DIR}/logs"
PID_DIR="${STATE_DIR}/pids"
RAY_STATE_FILE="${STATE_DIR}/ray_state.env"

SETTINGS="config/settings.yaml"
RAY_ADDRESS="auto"
PROXY_LOCATION="Disabled"
RELOAD=0
ASCEND_VISIBLE_DEVICES_VALUE="0,1,2,3"
NUM_NPUS="4"
CHECK_DEVICES=1

usage() {
  cat <<'EOF'
Usage: scripts/start_minimal_ascend.sh [options]

Options:
  --settings PATH             Path to settings.yaml (default: config/settings.yaml)
  --ray-address ADDR          Ray address passed to the Serve runtime launcher
  --num-npus N                NPU count for a locally started Ray head
  --ascend-visible-devices CSV
                             Export ASCEND_RT_VISIBLE_DEVICES before startup
  --no-device-check           Skip /dev/davinci* preflight checks
  --proxy-location VALUE      Ray Serve proxy location setting
  --reload                    Enable uvicorn reload for the gateway
  -h, --help                  Show this help

Examples:
  scripts/start_minimal_ascend.sh --ascend-visible-devices 0,1,2,3 --num-npus 4
  scripts/start_minimal_ascend.sh --ray-address auto
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --settings) SETTINGS="$2"; shift 2;;
    --ray-address) RAY_ADDRESS="$2"; shift 2;;
    --num-npus) NUM_NPUS="$2"; shift 2;;
    --ascend-visible-devices) ASCEND_VISIBLE_DEVICES_VALUE="$2"; shift 2;;
    --no-device-check) CHECK_DEVICES=0; shift;;
    --proxy-location) PROXY_LOCATION="$2"; shift 2;;
    --reload) RELOAD=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

mkdir -p "${LOG_DIR}" "${PID_DIR}"
rm -f "${RAY_STATE_FILE}"

cd "${ROOT_DIR}"

source "${ROOT_DIR}/.venv/bin/activate"

# The repository uses a src/ layout. When running with the container's system
# Python instead of an installed wheel/venv, make src importable explicitly.
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${ASCEND_VISIBLE_DEVICES_VALUE}" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES_VALUE}"
fi

# Ray may otherwise rewrite ASCEND_RT_VISIBLE_DEVICES for workers. vLLM Ascend
# deployments usually want the device mask to remain under explicit control.
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}"

if [[ "${CHECK_DEVICES}" -eq 1 ]]; then
  missing_devices=()
  for device in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    if [[ ! -e "${device}" ]]; then
      missing_devices+=("${device}")
    fi
  done
  if ! compgen -G "/dev/davinci[0-9]*" >/dev/null; then
    missing_devices+=("/dev/davinci*")
  fi
  if [[ "${#missing_devices[@]}" -gt 0 ]]; then
    echo "Missing Ascend device nodes: ${missing_devices[*]}" >&2
    echo "If this is a container, mount the Ascend devices before running this script." >&2
    exit 1
  fi
fi

if ! command -v ray >/dev/null 2>&1; then
  echo "Missing 'ray' in PATH. Did you build/install the Ascend runtime environment?" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Missing 'python' in PATH. The container image must provide the runtime environment." >&2
  exit 1
fi

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "Missing 'uvicorn' in PATH. The container image must provide the runtime environment." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Missing 'curl' in PATH. The container image must provide curl for gateway health checks." >&2
  exit 1
fi

ray_is_running() {
  ray status >/dev/null 2>&1
}

start_ray_head_if_needed() {
  if ray_is_running; then
    cat >"${RAY_STATE_FILE}" <<EOF
RAY_STARTED_BY_SCRIPT=0
RAY_ADDRESS=${RAY_ADDRESS}
ACCELERATOR_TYPE=ascend
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-}
EOF
    return 0
  fi

  local resources
  resources="{\"NPU\": ${NUM_NPUS}}"
  local args=(start --head --disable-usage-stats --resources "${resources}")

  (ray "${args[@]}" >"${LOG_DIR}/ray.log" 2>&1) || {
    echo "Failed to start Ray head. See ${LOG_DIR}/ray.log" >&2
    exit 1
  }

  local i
  for i in {1..40}; do
    if ray_is_running; then
      cat >"${RAY_STATE_FILE}" <<EOF
RAY_STARTED_BY_SCRIPT=1
RAY_ADDRESS=${RAY_ADDRESS}
ACCELERATOR_TYPE=ascend
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-}
EOF
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

wait_for_pid_exit() {
  local pid_file="$1"
  local label="$2"
  local timeout_seconds="$3"
  local waited=0

  while pid_is_running "${pid_file}"; do
    if [[ "${waited}" -ge "${timeout_seconds}" ]]; then
      echo "Timed out waiting for ${label} to finish initialization." >&2
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done

  wait "$(cat "${pid_file}")"
}

write_status() {
  local msg="$1"
  printf '%s\n' "${msg}" >"${STATE_DIR}/STATUS"
}

wait_for_gateway_ready() {
  local pid_file="$1"
  local timeout_seconds="$2"
  local url="http://127.0.0.1:8000/healthz"
  local waited=0

  while [[ "${waited}" -lt "${timeout_seconds}" ]]; do
    if ! pid_is_running "${pid_file}"; then
      echo "Gateway exited before becoming ready. See ${LOG_DIR}/gateway.log" >&2
      return 1
    fi

    if curl -fsS --max-time 1 "${url}" >/dev/null 2>&1; then
      return 0
    fi

    sleep 1
    waited=$((waited + 1))
  done

  echo "Timed out waiting for gateway to listen on ${url}. See ${LOG_DIR}/gateway.log" >&2
  return 1
}

start_ray_head_if_needed

SERVE_PID_FILE="${PID_DIR}/serve_runtime.pid"
GATEWAY_PID_FILE="${PID_DIR}/gateway.pid"

STARTED_SERVE_RUNTIME=0

if pid_is_running "${SERVE_PID_FILE}"; then
  echo "Serve runtime already running (pid $(cat "${SERVE_PID_FILE}"))."
else
  (
    exec python scripts/run_serve_runtime.py \
      --settings "${SETTINGS}" \
      --ray-address "${RAY_ADDRESS}" \
      --proxy-location "${PROXY_LOCATION}"
  ) >"${LOG_DIR}/serve_runtime.log" 2>&1 &
  echo $! >"${SERVE_PID_FILE}"
  STARTED_SERVE_RUNTIME=1
fi

if [[ "${STARTED_SERVE_RUNTIME}" -eq 1 ]]; then
  if ! wait_for_pid_exit "${SERVE_PID_FILE}" "serve runtime deployment" 900; then
    echo "Serve runtime failed to deploy. See ${LOG_DIR}/serve_runtime.log" >&2
    exit 1
  fi
fi

if pid_is_running "${GATEWAY_PID_FILE}"; then
  echo "Gateway already running (pid $(cat "${GATEWAY_PID_FILE}"))."
else
  gateway_args=(python scripts/run_gateway.py --settings "${SETTINGS}")
  if [[ "${RELOAD}" -eq 1 ]]; then
    gateway_args+=(--reload)
  fi

  (PYTHONUNBUFFERED=1 exec "${gateway_args[@]}") >"${LOG_DIR}/gateway.log" 2>&1 &
  echo $! >"${GATEWAY_PID_FILE}"
fi

if ! wait_for_gateway_ready "${GATEWAY_PID_FILE}" 60; then
  exit 1
fi

write_status "started"

echo "Started on Ascend NPU."
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-}"
echo "Ray NPU resources: ${NUM_NPUS}"
echo "Logs: ${LOG_DIR}"
echo "PIDs: ${PID_DIR}"
echo "Try:"
echo "  curl http://127.0.0.1:8000/healthz"
echo "  curl http://127.0.0.1:8000/v1/models"
