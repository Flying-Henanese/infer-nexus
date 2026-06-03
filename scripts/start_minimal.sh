#!/usr/bin/env bash
set -euo pipefail

# Minimal local bootstrap for Infer Nexus.
#
# The script is intentionally linear:
# 1. prepare the workspace state directories
# 2. optionally install / sync dependencies with uv
# 3. ensure a Ray head is available, starting one if needed
# 4. deploy the Serve runtime process
# 5. launch the FastAPI gateway
#
# Keeping these steps in one place makes local bring-up reproducible and also
# gives us a single log/pid directory to inspect when something fails.
#
# Logs:   .infer-nexus/logs/{ray,serve_runtime,gateway}.log
# PIDs:   .infer-nexus/pids/{serve_runtime,gateway}.pid
# Status: .infer-nexus/STATUS

# Resolve the repository root relative to this script so the command works
# no matter where it is invoked from.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# All runtime state is kept under a hidden repo-local directory so we do not
# clutter the project root and can cleanly inspect or remove generated artifacts.
STATE_DIR="${ROOT_DIR}/.infer-nexus"
LOG_DIR="${STATE_DIR}/logs"
PID_DIR="${STATE_DIR}/pids"

# This file records whether the current script invocation started Ray itself.
# That makes it easier for downstream tooling to understand who owns the head.
RAY_STATE_FILE="${STATE_DIR}/ray_state.env"

# Default runtime settings and startup behavior.
SETTINGS="config/settings.yaml"
RAY_ADDRESS="auto"
PROXY_LOCATION="Disabled"
INSTALL=1
RELOAD=0
CUDA_VISIBLE_DEVICES_VALUE="0,1,2,3"
NUM_GPUS="4"
PYTHON_BIN="${PYTHON_BIN:-python}"
RAY_BIN="${RAY_BIN:-ray}"
UV_BIN="${UV_BIN:-uv}"

usage() {
  cat <<'EOF'
Usage: scripts/start_minimal.sh [options]

Options:
  --settings PATH            Path to settings.yaml (default: config/settings.yaml)
  --no-install               Skip dependency sync before startup
  --install-artifacts        Also install the optional artifacts extra
  --ray-address ADDR         Ray address passed to the Serve runtime launcher
  --num-gpus N               GPU count for a locally started Ray head
  --cuda-visible-devices CSV Export CUDA_VISIBLE_DEVICES before starting Ray/runtime
  --proxy-location VALUE     Ray Serve proxy location setting
  --reload                   Enable uvicorn reload for the gateway
  -h, --help                 Show this help

Examples:
  scripts/start_minimal.sh --cuda-visible-devices 0,1,2,3 --num-gpus 4
  scripts/start_minimal.sh --no-install --ray-address auto

Environment overrides:
  PYTHON_BIN=/path/to/python  Python interpreter for gateway and Serve runtime
  RAY_BIN=/path/to/ray        Ray CLI used for `ray status` and `ray start`
  UV_BIN=/path/to/uv          uv binary used only for optional dependency sync
EOF
}

# Optional dependency set selector.
# When enabled, the script will install the service/runtime extras before
# attempting to start any processes.
# 也就是读取命令行参数的
INSTALL_ARTIFACTS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    # Allow callers to point at a non-default settings file.
    --settings) SETTINGS="$2"; shift 2;;
    # Skip `uv sync` entirely when the environment is already prepared.
    --no-install) INSTALL=0; shift;;
    # Include the optional artifacts extra when syncing dependencies.
    --install-artifacts) INSTALL_ARTIFACTS=1; shift;;
    # Override the Ray address passed into the runtime launcher.
    --ray-address) RAY_ADDRESS="$2"; shift 2;;
    # Set the GPU count for a locally spawned Ray head.
    --num-gpus) NUM_GPUS="$2"; shift 2;;
    # Export CUDA_VISIBLE_DEVICES before Ray starts so both Ray and the
    # runtime see the same GPU subset.
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2;;
    # Forward the proxy location to the Serve runtime launcher.
    --proxy-location) PROXY_LOCATION="$2"; shift 2;;
    # Developer convenience: turn on hot-reload for the gateway process.
    --reload) RELOAD=1; shift;;
    # Standard help flag.
    -h|--help) usage; exit 0;;
    # Fail fast on unknown flags so typos do not silently change behavior.
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

# Create the local state directories up front so later commands can write logs,
# pid files, and status markers without extra checks.
mkdir -p "${LOG_DIR}" "${PID_DIR}"

# Clear the Ray ownership marker from previous runs. It will be recreated if
# this invocation launches Ray itself.
rm -f "${RAY_STATE_FILE}"

# Switch to repo root so all subsequent relative paths match the project layout.
cd "${ROOT_DIR}"

# If the caller supplied a GPU allowlist, export it before starting any process.
# This keeps the Ray head, Serve runtime, and gateway aligned on the same device set.
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi

# Optionally sync the project environment before startup.
# This ensures `ray`, `serve`, `vllm`, and any extras are available.
if [[ "${INSTALL}" -eq 1 ]]; then
  if ! "${UV_BIN}" --version >/dev/null 2>&1; then
    echo "Missing '${UV_BIN}' for dependency sync. Install uv first or pass --no-install." >&2
    exit 1
  fi
  if [[ "${INSTALL_ARTIFACTS}" -eq 1 ]]; then
    "${UV_BIN}" sync --preview-features extra-build-dependencies --extra serve --extra vllm --extra artifacts
  else
    "${UV_BIN}" sync --preview-features extra-build-dependencies --extra serve --extra vllm
  fi
fi

# The runtime launcher and gateway should both use the same selected Python interpreter.
if ! "${PYTHON_BIN}" -V >/dev/null 2>&1; then
  echo "Configured PYTHON_BIN='${PYTHON_BIN}' is not executable." >&2
  exit 1
fi

# Ray is a hard dependency for the runtime deployment step.
if ! "${RAY_BIN}" --version >/dev/null 2>&1; then
  echo "Configured RAY_BIN='${RAY_BIN}' is not executable. Did you install ray in the selected environment?" >&2
  exit 1
fi

# Helper: return success when a Ray head is already reachable.
# `ray status` is a pragmatic connectivity check here because it fails when
# there is no live Ray head to talk to.
ray_is_running() {
  # "ray status" exits non-zero if it can't connect to a running head.
  "${RAY_BIN}" status >/dev/null 2>&1
}

# Start a local Ray head only if we cannot already talk to one.
# This lets the script attach to an existing cluster in shared/dev setups while
# still being self-sufficient on a fresh machine.
start_ray_head_if_needed() {
  if ray_is_running; then
    # Record that Ray was already up when we arrived.
    cat >"${RAY_STATE_FILE}" <<EOF
RAY_STARTED_BY_SCRIPT=0
RAY_ADDRESS=${RAY_ADDRESS}
EOF
    return 0
  fi

  # Build the `ray start` argument list incrementally so we only pass flags
  # that are relevant for this invocation.
  local args=(start --head --disable-usage-stats)
  if [[ -n "${NUM_GPUS}" ]]; then
    args+=(--num-gpus "${NUM_GPUS}")
  fi

  # Ray prints useful diagnostics during startup; capture them in the Ray log.
  # Note: Ray daemonizes, so the command itself returns quickly after spawn.
  (exec "${RAY_BIN}" "${args[@]}" >"${LOG_DIR}/ray.log" 2>&1) || {
    echo "Failed to start Ray head. See ${LOG_DIR}/ray.log" >&2
    exit 1
  }

  # Poll for a short period so we do not race the subsequent deployment step.
  local i
  for i in {1..40}; do
    if ray_is_running; then
      # Mark that this script created the Ray head for the current session.
      cat >"${RAY_STATE_FILE}" <<EOF
RAY_STARTED_BY_SCRIPT=1
RAY_ADDRESS=${RAY_ADDRESS}
EOF
      return 0
    fi
    sleep 0.25
  done

  echo "Ray did not become ready in time. See ${LOG_DIR}/ray.log" >&2
  exit 1
}

# Check whether a pid file points at a currently live process.
# We use pid files for the long-running gateway and the serve-runtime launcher.
pid_is_running() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" >/dev/null 2>&1
}

# Wait until a tracked process exits, but give up after a timeout.
# This is used for the Serve runtime launcher because it should exit after it
# finishes deploying the app; if it hangs, we want to fail early with context.
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

# Write a simple single-line status marker for external scripts or humans.
write_status() {
  local msg="$1"
  printf '%s\n' "${msg}" >"${STATE_DIR}/STATUS"
}

# Ensure the Ray head exists before launching anything that depends on it.
start_ray_head_if_needed

# Track the two background processes separately so we can detect and reuse
# already-running instances on subsequent invocations.
SERVE_PID_FILE="${PID_DIR}/serve_runtime.pid"
GATEWAY_PID_FILE="${PID_DIR}/gateway.pid"

STARTED_SERVE_RUNTIME=0

# The Serve runtime launcher is a one-shot deployment process. If it is already
# running, we leave it alone; otherwise we start it in the background and wait
# for it to complete initialization.
if pid_is_running "${SERVE_PID_FILE}"; then
  echo "Serve runtime already running (pid $(cat "${SERVE_PID_FILE}"))."
else
  (
    exec "${PYTHON_BIN}" scripts/run_serve_runtime.py \
      --settings "${SETTINGS}" \
      --ray-address "${RAY_ADDRESS}" \
      --proxy-location "${PROXY_LOCATION}"
  ) >"${LOG_DIR}/serve_runtime.log" 2>&1 &
  echo $! >"${SERVE_PID_FILE}"
  STARTED_SERVE_RUNTIME=1
fi

# If we started the deployment launcher ourselves, wait for it to finish
# initializing the runtime app. A timeout here usually indicates a deployment
# error that should be inspected in the runtime log.
if [[ "${STARTED_SERVE_RUNTIME}" -eq 1 ]]; then
  if ! wait_for_pid_exit "${SERVE_PID_FILE}" "serve runtime deployment" 900; then
    echo "Serve runtime failed to deploy. See ${LOG_DIR}/serve_runtime.log" >&2
    exit 1
  fi
fi

# The gateway is a long-lived HTTP process. Reuse it if it is already alive;
# otherwise start a new one and leave it running in the background.
if pid_is_running "${GATEWAY_PID_FILE}"; then
  echo "Gateway already running (pid $(cat "${GATEWAY_PID_FILE}"))."
else
  # Build the gateway command first so we can add `--reload` only when asked.
  gateway_args=("${PYTHON_BIN}" scripts/run_gateway.py --settings "${SETTINGS}")
  if [[ "${RELOAD}" -eq 1 ]]; then
    gateway_args+=(--reload)
  fi

  # Start the gateway in the background and capture its stdout/stderr in a log.
  (exec "${gateway_args[@]}") >"${LOG_DIR}/gateway.log" 2>&1 &
  echo $! >"${GATEWAY_PID_FILE}"
fi

# Publish the overall status for any wrapper scripts or tooling watching the
# local state directory.
write_status "started"

echo "Started."
echo "Logs: ${LOG_DIR}"
echo "PIDs: ${PID_DIR}"
echo "Try:"
echo "  curl http://127.0.0.1:8000/healthz"
echo "  curl http://127.0.0.1:8000/v1/models"
