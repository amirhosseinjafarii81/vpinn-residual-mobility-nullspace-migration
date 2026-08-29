#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"
SCRIPT="${ROOT}/vpinn_gradient_conflict_stage8_adam_target_reflection.py"
STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"
STAGE5="${ROOT}/vpinn_gradient_conflict_stage5_escape_time"
STAGE6="${ROOT}/vpinn_gradient_conflict_stage6_adam_update_geometry"
STAGE7="${ROOT}/vpinn_gradient_conflict_stage7_adam_state_component_audit"

for path in "${PYTHON}" "${SCRIPT}" "${STAGE3}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing required file: ${path}" >&2
    exit 2
  fi
done

for dir in "${STAGE5}" "${STAGE6}" "${STAGE7}"; do
  if [[ ! -d "${dir}" ]]; then
    echo "ERROR: missing required result directory: ${dir}" >&2
    exit 2
  fi
done

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --device cpu \
  --stage3-script "${STAGE3}" \
  --stage5-dir "${STAGE5}" \
  --stage6-dir "${STAGE6}" \
  --stage7-dir "${STAGE7}"
