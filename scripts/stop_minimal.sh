#!/usr/bin/env bash
set -euo pipefail

# 以仓库根目录为基准定位运行时状态文件，避免从任意工作目录执行时找错路径。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 所有启动脚本共享的本地状态目录。
STATE_DIR="${ROOT_DIR}/.infer-nexus"
PID_DIR="${STATE_DIR}/pids"
RAY_STATE_FILE="${STATE_DIR}/ray_state.env"

# 此 pid 文件记录 one-shot Serve runtime 启动器进程。
SERVE_PID_FILE="${PID_DIR}/serve_runtime.pid"

# 根据 pid 文件终止进程，并尽量连同进程组里的子进程一起清理掉。
# 这样可以处理像 uv / python / shell wrapper 这类“外面一层壳，里面一层真进程”的启动方式。
kill_from_pid_file() {
  local pid_file="$1"
  local name="$2"
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    # pid 文件是空的，直接清掉，避免下次误判。
    rm -f "${pid_file}"
    return 0
  fi

  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "Stopping ${name} (pid ${pid})..."
    # 先尝试对进程组发信号，尽量把子进程一起带走。
    # 这一步对通过 wrapper 启动的进程特别重要。
    kill -- "-${pid}" >/dev/null 2>&1 || true
    # 再补一发单进程信号，兼容没有独立进程组的情况。
    kill "${pid}" >/dev/null 2>&1 || true

    local i
    # 给进程一点时间做正常退出和资源释放。
    for i in {1..40}; do
      if ! kill -0 "${pid}" >/dev/null 2>&1; then
        break
      fi
      sleep 0.25
    done

    if kill -0 "${pid}" >/dev/null 2>&1; then
      echo "${name} did not exit; sending SIGKILL..."
      # 如果还没退出，就直接强杀，避免残留阻塞后续启动。
      kill -9 -- "-${pid}" >/dev/null 2>&1 || true
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  fi

  # 无论进程是否存在，pid 文件都清掉，保证状态文件不“陈旧”。
  rm -f "${pid_file}"
}

# 兜底清理本机遗留的 vLLM worker 进程。
# 这些进程有时不会随着父进程正常退出而立刻消失，因此需要按命令行特征补刀。
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

# 读取 Ray 状态文件，判断当前这套环境是不是由本脚本启动的。
# 如果是，本脚本后面可以放心执行 `ray stop -f`。
load_ray_state() {
  RAY_STARTED_BY_SCRIPT=1
  RAY_ADDRESS="auto"
  if [[ -f "${RAY_STATE_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${RAY_STATE_FILE}" || true
  fi
}

# 主动请求 Serve 关闭当前应用。
# 这一步很关键：它给 Ray Serve 的 replicas 一个“优雅退出”的机会，
# 让它们先做正常 teardown，再进入后面的强制清理阶段。
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

# 兜底清理本机残留的 Ray 相关守护进程和 worker。
# 这里的模式覆盖了 raylet / gcs / dashboard / worker / ray:: 等常见残留形态。
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

# 先停掉 Serve runtime 启动器进程。
# 注意：serve runtime 这里只是“启动器”进程，真正的副作用在 Ray Serve 集群里。
kill_from_pid_file "${SERVE_PID_FILE}" "serve runtime"

# 读取 Ray 归属信息，决定后续是否有权把本地 Ray cluster 整个停掉。
load_ray_state

# 先让 Serve 关闭应用。
# 这样做的目的，是让 Ray Serve 负责管理的 replicas 尽量走正常退出路径，
# 避免它们直接变成孤儿进程或卡在资源释放上。
stop_serve_app "${RAY_ADDRESS}"

# 如果这个脚本曾经自己拉起本地 Ray head，那么这里可以直接把整套 Ray runtime
# 强制停掉。这样即使有 detached Serve replicas、后台 actor、vLLM workers 没有
# 跟着正常退出，也会在 Ray 级别被收掉，不再继续占用 GPU / CPU 资源。
if [[ "${RAY_STARTED_BY_SCRIPT}" == "1" ]] && command -v ray >/dev/null 2>&1; then
  ray stop -f >/dev/null 2>&1 || true
fi

# 最后再做一次按特征扫描的兜底清理，处理前面步骤遗漏的残留进程。
kill_stray_vllm_local
kill_stray_ray_local

if [[ -d "${STATE_DIR}" ]]; then
  # 把总体状态写回本地状态目录，便于外部脚本判断是否已经停过。
  printf '%s\n' "stopped" >"${STATE_DIR}/STATUS" || true
fi

# 清掉 Ray 归属标记，表示这次停止流程已经结束。
rm -f "${RAY_STATE_FILE}"

echo "Stopped."
