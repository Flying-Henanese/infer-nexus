#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${ROOT_DIR}/.infer-nexus"
PID_DIR="${STATE_DIR}/pids"

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
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  fi

  rm -f "${pid_file}"
}

kill_from_pid_file "${GATEWAY_PID_FILE}" "gateway"
kill_from_pid_file "${SERVE_PID_FILE}" "serve runtime"

# Stop Ray runtime forcefully so detached Serve replicas / vLLM workers
# do not keep occupying GPU memory after the gateway/runtime parent exits.
if command -v ray >/dev/null 2>&1; then
  ray stop -f >/dev/null 2>&1 || true
fi

if [[ -d "${STATE_DIR}" ]]; then
  printf '%s\n' "stopped" >"${STATE_DIR}/STATUS" || true
fi

echo "Stopped."

