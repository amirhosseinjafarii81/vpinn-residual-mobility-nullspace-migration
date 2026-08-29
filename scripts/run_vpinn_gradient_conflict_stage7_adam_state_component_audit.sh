#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"
SCRIPT="${ROOT}/vpinn_gradient_conflict_stage7_adam_state_component_audit.py"
STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"
STAGE5="${ROOT}/vpinn_gradient_conflict_stage5_escape_time"
STAGE6="${ROOT}/vpinn_gradient_conflict_stage6_adam_update_geometry"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: Stage-7 script not found: ${SCRIPT}" >&2
  exit 2
fi

if [[ ! -f "${STAGE3}" ]]; then
  echo "ERROR: Stage-3 solver not found: ${STAGE3}" >&2
  exit 2
fi

if [[ ! -d "${STAGE5}" ]]; then
  echo "ERROR: Stage-5 result directory not found: ${STAGE5}" >&2
  exit 2
fi

if [[ ! -d "${STAGE6}" ]]; then
  echo "ERROR: Stage-6 result directory not found: ${STAGE6}" >&2
  exit 2
fi

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --device cpu \
  --stage3-script "${STAGE3}" \
  --stage5-dir "${STAGE5}" \
  --stage6-dir "${STAGE6}"
