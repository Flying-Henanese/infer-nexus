#!/usr/bin/env bash
set -euo pipefail

# Attach an Ascend-capable Ray worker to the compose Ray head.
#
# Put model replicas here by advertising NPU resources on the worker, not on
# the head. This keeps the head mostly control-plane focused.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

RAY_ADDRESS="${RAY_ADDRESS:-ray-head:6379}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-}"
RAY_NUM_NPUS="${RAY_NUM_NPUS:-}"
RAY_BLOCK="${RAY_BLOCK:-1}"

count_csv_items() {
  local csv="$1"
  local count=0
  local item
  IFS=',' read -ra items <<< "${csv}"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "${count}"
}

if [[ -z "${RAY_NUM_NPUS}" && -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  RAY_NUM_NPUS="$(count_csv_items "${ASCEND_RT_VISIBLE_DEVICES}")"
fi
RAY_NUM_NPUS="${RAY_NUM_NPUS:-0}"

if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}"
fi

if ! command -v ray >/dev/null 2>&1; then
  echo "Missing 'ray' in PATH. The container image must include Ray." >&2
  exit 1
fi

args=(start --address "${RAY_ADDRESS}" --disable-usage-stats)

if [[ -n "${RAY_NUM_CPUS}" ]]; then
  args+=(--num-cpus "${RAY_NUM_CPUS}")
fi

if [[ "${RAY_NUM_NPUS}" != "0" ]]; then
  args+=(--resources "{\"NPU\": ${RAY_NUM_NPUS}}")
fi

if [[ "${RAY_BLOCK}" == "1" ]]; then
  args+=(--block)
fi

exec ray "${args[@]}"
