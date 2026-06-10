#!/usr/bin/env bash
set -uo pipefail

# Check system-side dependencies for CUDA profiling, monitoring, and benchmark runs.
# This script is read-only. It reports missing tools and does not install anything.

STRICT=0
RUN_LIGHT_PROBES=1

usage() {
  cat <<'EOF'
Usage: scripts/check_cuda_perf_system_deps.sh [options]

Options:
  --strict          Treat recommended dependencies as failures.
  --no-probes       Do not run lightweight commands such as nvidia-smi queries.
  -h, --help        Show this help.

Exit code:
  0  required checks passed
  1  at least one required check failed, or recommended failed with --strict
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict) STRICT=1; shift;;
    --no-probes) RUN_LIGHT_PROBES=0; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

required_failures=0
recommended_failures=0
optional_missing=0

section() {
  printf '\n== %s ==\n' "$1"
}

pass() {
  printf '[OK  ] %s\n' "$1"
}

fail_required() {
  printf '[FAIL] %s\n' "$1"
  required_failures=$((required_failures + 1))
}

fail_recommended() {
  printf '[WARN] %s\n' "$1"
  recommended_failures=$((recommended_failures + 1))
}

missing_optional() {
  printf '[MISS] %s\n' "$1"
  optional_missing=$((optional_missing + 1))
}

check_command() {
  local level="$1"
  local command_name="$2"
  local purpose="$3"
  local path

  path="$(command -v "${command_name}" 2>/dev/null || true)"
  if [[ -n "${path}" ]]; then
    pass "${command_name}: ${path} (${purpose})"
    return 0
  fi

  case "${level}" in
    required) fail_required "${command_name}: not found (${purpose})";;
    recommended) fail_recommended "${command_name}: not found (${purpose})";;
    *) missing_optional "${command_name}: not found (${purpose})";;
  esac
  return 1
}

check_any_command() {
  local level="$1"
  local purpose="$2"
  shift 2

  local command_name
  for command_name in "$@"; do
    if command -v "${command_name}" >/dev/null 2>&1; then
      pass "${command_name}: $(command -v "${command_name}") (${purpose})"
      return 0
    fi
  done

  case "${level}" in
    required) fail_required "$*: none found (${purpose})";;
    recommended) fail_recommended "$*: none found (${purpose})";;
    *) missing_optional "$*: none found (${purpose})";;
  esac
  return 1
}

check_path() {
  local level="$1"
  local path="$2"
  local purpose="$3"

  if [[ -e "${path}" ]]; then
    pass "${path}: present (${purpose})"
    return 0
  fi

  case "${level}" in
    required) fail_required "${path}: missing (${purpose})";;
    recommended) fail_recommended "${path}: missing (${purpose})";;
    *) missing_optional "${path}: missing (${purpose})";;
  esac
  return 1
}

check_env() {
  local level="$1"
  local name="$2"
  local purpose="$3"
  local value="${!name:-}"

  if [[ -n "${value}" ]]; then
    pass "${name}=${value} (${purpose})"
    return 0
  fi

  case "${level}" in
    required) fail_required "${name}: unset (${purpose})";;
    recommended) fail_recommended "${name}: unset (${purpose})";;
    *) missing_optional "${name}: unset (${purpose})";;
  esac
  return 1
}

section "Host"
printf 'hostname: %s\n' "$(hostname 2>/dev/null || echo unknown)"
printf 'kernel  : %s\n' "$(uname -a 2>/dev/null || echo unknown)"
printf 'user    : %s\n' "$(id 2>/dev/null || whoami 2>/dev/null || echo unknown)"

section "CUDA device visibility"
check_command required nvidia-smi "GPU health, power, memory, and utilization"
check_path recommended /dev/nvidiactl "NVIDIA control device node"
check_path recommended /dev/nvidia-uvm "NVIDIA unified memory device node"
if compgen -G "/dev/nvidia[0-9]*" >/dev/null; then
  pass "/dev/nvidia[0-9]*: present ($(compgen -G "/dev/nvidia[0-9]*" | wc -l | tr -d ' ') devices)"
else
  fail_recommended "/dev/nvidia[0-9]*: missing (GPU device nodes; expected inside GPU containers)"
fi
check_env recommended CUDA_VISIBLE_DEVICES "GPU mask used by CUDA, vLLM, and Ray workers"
check_env optional NVIDIA_VISIBLE_DEVICES "GPU mask used by NVIDIA container runtime"

if [[ "${RUN_LIGHT_PROBES}" -eq 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  section "nvidia-smi probe"
  if nvidia-smi >/tmp/infer_nexus_nvidia_smi.txt 2>&1; then
    pass "nvidia-smi succeeded"
    sed -n '1,40p' /tmp/infer_nexus_nvidia_smi.txt
  else
    fail_required "nvidia-smi failed; see /tmp/infer_nexus_nvidia_smi.txt"
    sed -n '1,40p' /tmp/infer_nexus_nvidia_smi.txt
  fi

  section "nvidia-smi metrics probe"
  if nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits >/tmp/infer_nexus_nvidia_smi_query.txt 2>&1; then
    pass "nvidia-smi query-gpu succeeded"
    sed -n '1,20p' /tmp/infer_nexus_nvidia_smi_query.txt
  else
    fail_recommended "nvidia-smi query-gpu failed; see /tmp/infer_nexus_nvidia_smi_query.txt"
    sed -n '1,20p' /tmp/infer_nexus_nvidia_smi_query.txt
  fi
fi

section "CUDA and profiling tools"
check_command recommended nvcc "CUDA compiler/version probe"
check_command recommended nsys "Nsight Systems timeline profiler"
check_command recommended ncu "Nsight Compute kernel profiler"
check_command recommended nvprof "Legacy CUDA profiler, useful only on older stacks"
check_command recommended nvlink "CUDA binary/toolchain sanity check"
check_any_command recommended "NVTX command-line helpers when installed" nvtxsum nvtxpp

section "Monitoring exporters"
check_command recommended dcgmi "DCGM host diagnostics and GPU telemetry"
check_any_command recommended "DCGM exporter for Prometheus" dcgm-exporter nvidia-dcgm-exporter
check_command recommended prometheus "Prometheus metrics storage"
check_command recommended grafana-server "Grafana dashboard service"
check_command recommended node_exporter "CPU, memory, disk, and network metrics"
check_command recommended process-exporter "Ray/vLLM/gateway process metrics"

section "Benchmark and diagnostics tools"
check_command required python "run benchmark and dependency checks"
check_command recommended curl "HTTP smoke tests and metrics endpoint checks"
check_command recommended jq "JSON response inspection"
check_command recommended pidstat "per-process CPU sampling"
check_command recommended iostat "disk and device IO sampling"
check_command recommended sar "system activity reports"
check_command optional py-spy "Python CPU flamegraphs"

section "Python profiling entrypoints"
if command -v python >/dev/null 2>&1; then
  python - <<'PY'
import importlib

checks = [
    ("torch", "PyTorch CUDA runtime and profiler"),
    ("pynvml", "NVML GPU telemetry from Python"),
    ("tensorboard", "Profiler trace viewer"),
    ("prometheus_client", "Application metrics exporter"),
    ("psutil", "Process metrics sampling"),
    ("nvtx", "NVTX ranges for Nsight Systems timeline annotation"),
]

for module, purpose in checks:
    try:
        mod = importlib.import_module(module)
        print(f"[OK  ] {module}: importable ({purpose})")
        if module == "torch":
            print(f"       torch={getattr(mod, '__version__', 'unknown')} cuda_build={getattr(mod.version, 'cuda', None)} cuda_available={mod.cuda.is_available()} device_count={mod.cuda.device_count()}")
    except Exception as exc:
        print(f"[WARN] {module}: import failed: {type(exc).__name__}: {exc} ({purpose})")
PY
else
  fail_required "python: not found, skipped Python profiling import checks"
fi

section "Summary"
printf 'required_failures=%s\n' "${required_failures}"
printf 'recommended_failures=%s\n' "${recommended_failures}"
printf 'optional_missing=%s\n' "${optional_missing}"
if [[ "${required_failures}" -gt 0 ]]; then
  exit 1
fi
if [[ "${STRICT}" -eq 1 && "${recommended_failures}" -gt 0 ]]; then
  exit 1
fi
exit 0
