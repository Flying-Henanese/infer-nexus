#!/usr/bin/env bash
set -uo pipefail

# Check system-side dependencies for Ascend profiling, monitoring, and benchmark runs.
# This script is read-only. It reports missing tools and does not install anything.

STRICT=0
RUN_LIGHT_PROBES=1

usage() {
  cat <<'EOF'
Usage: scripts/check_ascend_perf_system_deps.sh [options]

Options:
  --strict          Treat recommended dependencies as failures.
  --no-probes       Do not run lightweight commands such as npu-smi info.
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

warn() {
  printf '[WARN] %s\n' "$1"
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

section "Ascend device visibility"
check_command required npu-smi "NPU health, power, memory, and utilization"
check_path required /dev/davinci_manager "Ascend device manager"
check_path required /dev/devmm_svm "Ascend shared virtual memory device"
check_path required /dev/hisi_hdc "Ascend host-device channel"
if compgen -G "/dev/davinci[0-9]*" >/dev/null; then
  pass "/dev/davinci[0-9]*: present ($(compgen -G "/dev/davinci[0-9]*" | wc -l | tr -d ' ') devices)"
else
  fail_required "/dev/davinci[0-9]*: missing (Ascend NPU device nodes)"
fi
check_env recommended ASCEND_RT_VISIBLE_DEVICES "NPU mask used by vLLM-Ascend and Ray workers"
check_env recommended RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES "prevents Ray from rewriting Ascend visibility"

if [[ "${RUN_LIGHT_PROBES}" -eq 1 ]] && command -v npu-smi >/dev/null 2>&1; then
  section "npu-smi probe"
  if npu-smi info >/tmp/infer_nexus_npu_smi_info.txt 2>&1; then
    pass "npu-smi info succeeded"
    sed -n '1,40p' /tmp/infer_nexus_npu_smi_info.txt
  else
    fail_required "npu-smi info failed; see /tmp/infer_nexus_npu_smi_info.txt"
    sed -n '1,40p' /tmp/infer_nexus_npu_smi_info.txt
  fi
fi

section "CANN and profiling tools"
check_path recommended /usr/local/Ascend/driver "Ascend driver mount/path"
check_path recommended /usr/local/Ascend/ascend-toolkit "CANN toolkit root"
check_path recommended /usr/local/Ascend/ascend-toolkit/latest "CANN toolkit latest symlink"
check_path recommended /usr/local/Ascend/ascend-toolkit/set_env.sh "CANN environment script"
check_any_command required "Ascend offline/system profiler" msprof msopprof
check_command recommended msprof "Ascend profiling command"
check_command recommended msopprof "Ascend operator profiling command"
check_command recommended hccn_tool "HCCL/network diagnostics for multi-NPU or multi-node runs"

section "Monitoring exporters"
check_any_command recommended "NPU metrics exporter for Prometheus" ascend-npu-exporter npu-exporter
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
    ("torch_npu.profiler", "Ascend PyTorch profiler"),
    ("tensorboard", "Profiler trace viewer"),
    ("prometheus_client", "Application metrics exporter"),
    ("psutil", "Process metrics sampling"),
]

for module, purpose in checks:
    try:
        mod = importlib.import_module(module)
        print(f"[OK  ] {module}: importable ({purpose})")
        if module == "torch_npu.profiler":
            print(f"       profile={hasattr(mod, 'profile')} tensorboard_trace_handler={hasattr(mod, 'tensorboard_trace_handler')}")
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
