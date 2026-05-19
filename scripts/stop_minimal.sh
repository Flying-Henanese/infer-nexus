#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${ROOT_DIR}/.infer-nexus"
PID_DIR="${STATE_DIR}/pids"
RAY_STATE_FILE="${STATE_DIR}/ray_state.env"

SERVE_PID_FILE="${PID_DIR}/serve_runtime.pid"
GATEWAY_PID_FILE="${PID_DIR}/gateway.pid"

kill_from_pid_file() {
  local pid_file="$1"
  local name="$2"
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    rm -f "${pid_file}"
    return 0
  fi

  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "Stopping ${name} (pid ${pid})..."
    # Try process-group first so children spawned by wrappers are also signaled.
    kill -- "-${pid}" >/dev/null 2>&1 || true
    kill "${pid}" >/dev/null 2>&1 || true

    local i
    for i in {1..40}; do
      if ! kill -0 "${pid}" >/dev/null 2>&1; then
        break
      fi
      sleep 0.25
    done

    if kill -0 "${pid}" >/dev/null 2>&1; then
      echo "${name} did not exit; sending SIGKILL..."
      kill -9 -- "-${pid}" >/dev/null 2>&1 || true
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  fi

  rm -f "${pid_file}"
}

kill_stray_vllm_local() {
  # Best-effort cleanup for local leftover engine workers.
  local patterns=(
    "vllm/v1/engine/core.py"
    "vllm.entrypoints"
  )
  local pattern
  for pattern in "${patterns[@]}"; do
    if pgrep -f "${pattern}" >/dev/null 2>&1; then
      pkill -TERM -f "${pattern}" >/dev/null 2>&1 || true
    fi
  done
  sleep 1
  for pattern in "${patterns[@]}"; do
    if pgrep -f "${pattern}" >/dev/null 2>&1; then
      pkill -KILL -f "${pattern}" >/dev/null 2>&1 || true
    fi
  done
}

load_ray_state() {
  RAY_STARTED_BY_SCRIPT=1
  RAY_ADDRESS="auto"
  if [[ -f "${RAY_STATE_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${RAY_STATE_FILE}" || true
  fi
}

stop_serve_app() {
  if ! command -v python >/dev/null 2>&1; then
    return 0
  fi
  local ray_address="${1:-auto}"
  python - <<PY >/dev/null 2>&1 || true
try:
    import ray
    from ray import serve

    ray.init(address="${ray_address}", ignore_reinit_error=True, logging_level="ERROR")
    try:
        serve.shutdown()
    finally:
        ray.shutdown()
except Exception:
    pass
PY
}

kill_stray_ray_local() {
  # Best-effort cleanup for leftover local Ray daemons/workers.
  local patterns=(
    "raylet"
    "gcs_server"
    "monitor.py"
    "dashboard.py"
    "dashboard_agent.py"
    "log_monitor.py"
    "runtime_env_agent.py"
    "ray::"
    "default_worker.py"
  )
  local pattern
  for pattern in "${patterns[@]}"; do
    if pgrep -f "${pattern}" >/dev/null 2>&1; then
      pkill -TERM -f "${pattern}" >/dev/null 2>&1 || true
    fi
  done
  sleep 2
  for pattern in "${patterns[@]}"; do
    if pgrep -f "${pattern}" >/dev/null 2>&1; then
      pkill -KILL -f "${pattern}" >/dev/null 2>&1 || true
    fi
  done
}

kill_from_pid_file "${GATEWAY_PID_FILE}" "gateway"
kill_from_pid_file "${SERVE_PID_FILE}" "serve runtime"

load_ray_state

# Ask Serve to shut down first so replicas have a chance to exit cleanly.
stop_serve_app "${RAY_ADDRESS}"

# Stop Ray runtime forcefully when this script created the local cluster, so
# detached Serve replicas / vLLM workers do not keep occupying GPU memory.
if [[ "${RAY_STARTED_BY_SCRIPT}" == "1" ]] && command -v ray >/dev/null 2>&1; then
  ray stop -f >/dev/null 2>&1 || true
fi

kill_stray_vllm_local
kill_stray_ray_local

if [[ -d "${STATE_DIR}" ]]; then
  printf '%s\n' "stopped" >"${STATE_DIR}/STATUS" || true
fi

rm -f "${RAY_STATE_FILE}"

echo "Stopped."

