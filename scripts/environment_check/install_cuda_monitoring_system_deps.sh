#!/usr/bin/env bash
set -euo pipefail

# Install system-side components for CUDA monitoring, profiling, and benchmark runs.
#
# This script targets Ubuntu/Debian-like CUDA hosts. It intentionally keeps
# heavyweight or repository-changing actions behind explicit flags.

DRY_RUN=0
WITH_PROMETHEUS=0
WITH_GRAFANA=0
INSTALL_DCGM_EXPORTER=0
DCGM_EXPORTER_IMAGE="${DCGM_EXPORTER_IMAGE:-nvcr.io/nvidia/k8s/dcgm-exporter:latest}"
DCGM_EXPORTER_NAME="${DCGM_EXPORTER_NAME:-dcgm-exporter}"
DCGM_EXPORTER_PORT="${DCGM_EXPORTER_PORT:-9400}"

usage() {
  cat <<'EOF'
Usage: scripts/install_cuda_monitoring_system_deps.sh [options]

Installs common CUDA monitoring/profiling system packages:
  - sysstat: pidstat, iostat, sar
  - curl, jq, ca-certificates, gnupg
  - prometheus-node-exporter
  - prometheus-process-exporter, when available from apt

Options:
  --dry-run                 Print commands without executing them.
  --with-prometheus         Install Prometheus from apt if available.
  --with-grafana            Install Grafana from apt if available.
  --install-dcgm-exporter   Start NVIDIA DCGM exporter as a Docker container.
  --dcgm-image IMAGE        DCGM exporter image (default: nvcr.io/nvidia/k8s/dcgm-exporter:latest).
  --dcgm-port PORT          Host port for DCGM exporter metrics (default: 9400).
  -h, --help                Show this help.

Notes:
  - NVIDIA driver, CUDA toolkit, Nsight Systems, and Nsight Compute are not
    installed by this script because they depend on the validated host image.
  - DCGM exporter requires Docker and NVIDIA container runtime access.
  - Run scripts/check_cuda_perf_system_deps.sh after installation.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift;;
    --with-prometheus) WITH_PROMETHEUS=1; shift;;
    --with-grafana) WITH_GRAFANA=1; shift;;
    --install-dcgm-exporter) INSTALL_DCGM_EXPORTER=1; shift;;
    --dcgm-image) DCGM_EXPORTER_IMAGE="$2"; shift 2;;
    --dcgm-port) DCGM_EXPORTER_PORT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

run() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '+ %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

run_shell() {
  local command="$1"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '+ %s\n' "${command}"
    return 0
  fi
  bash -lc "${command}"
}

need_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command '${command_name}'." >&2
    exit 1
  fi
}

sudo_cmd() {
  if [[ "$(id -u)" -eq 0 ]]; then
    run "$@"
  else
    run sudo "$@"
  fi
}

sudo_shell() {
  local command="$1"
  if [[ "$(id -u)" -eq 0 ]]; then
    run_shell "${command}"
  else
    run sudo bash -lc "${command}"
  fi
}

apt_has_package() {
  local package="$1"
  apt-cache show "${package}" >/dev/null 2>&1
}

install_apt_packages() {
  local packages=("$@")
  local available=()
  local skipped=()
  local package

  for package in "${packages[@]}"; do
    if apt_has_package "${package}"; then
      available+=("${package}")
    else
      skipped+=("${package}")
    fi
  done

  if [[ "${#skipped[@]}" -gt 0 ]]; then
    echo "Skipping packages not available from current apt sources: ${skipped[*]}"
  fi
  if [[ "${#available[@]}" -eq 0 ]]; then
    return 0
  fi
  sudo_cmd apt-get install -y "${available[@]}"
}

enable_service_if_present() {
  local service="$1"
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files "${service}.service" >/dev/null 2>&1; then
    sudo_cmd systemctl enable --now "${service}"
  else
    echo "Service '${service}' not managed by systemd or not installed; skipping enable."
  fi
}

install_dcgm_exporter_container() {
  need_command docker
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required before starting DCGM exporter." >&2
    exit 1
  fi

  if docker ps -a --format '{{.Names}}' | grep -Fxq "${DCGM_EXPORTER_NAME}"; then
    echo "Removing existing container '${DCGM_EXPORTER_NAME}'."
    run docker rm -f "${DCGM_EXPORTER_NAME}"
  fi

  run docker run -d \
    --name "${DCGM_EXPORTER_NAME}" \
    --restart unless-stopped \
    --gpus all \
    --cap-add SYS_ADMIN \
    -p "${DCGM_EXPORTER_PORT}:9400" \
    "${DCGM_EXPORTER_IMAGE}"
}

main() {
  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    echo "Detected OS: ${PRETTY_NAME:-unknown}"
  fi

  need_command apt-get
  need_command apt-cache

  sudo_cmd apt-get update

  install_apt_packages \
    ca-certificates \
    curl \
    gnupg \
    jq \
    sysstat \
    prometheus-node-exporter \
    prometheus-process-exporter

  enable_service_if_present prometheus-node-exporter
  enable_service_if_present prometheus-process-exporter

  if [[ "${WITH_PROMETHEUS}" -eq 1 ]]; then
    install_apt_packages prometheus
    enable_service_if_present prometheus
  else
    echo "Prometheus install skipped. Pass --with-prometheus to install it from apt."
  fi

  if [[ "${WITH_GRAFANA}" -eq 1 ]]; then
    install_apt_packages grafana grafana-server
    enable_service_if_present grafana-server
  else
    echo "Grafana install skipped. Pass --with-grafana to install it from apt if available."
  fi

  if [[ "${INSTALL_DCGM_EXPORTER}" -eq 1 ]]; then
    install_dcgm_exporter_container
  else
    echo "DCGM exporter container skipped. Pass --install-dcgm-exporter to start it."
  fi

  sudo_shell "command -v sysstat >/dev/null 2>&1 && sed -i 's/^ENABLED=.*/ENABLED=\"true\"/' /etc/default/sysstat || true"
  enable_service_if_present sysstat

  echo
  echo "Installation step finished."
  echo "Recommended validation:"
  echo "  bash scripts/check_cuda_perf_system_deps.sh"
  echo "  curl -fsS http://127.0.0.1:${DCGM_EXPORTER_PORT}/metrics | head"
}

main
