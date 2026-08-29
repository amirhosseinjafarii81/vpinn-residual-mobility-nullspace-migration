#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"

SCRIPT="${ROOT}/vpinn_gradient_conflict_stage11_joint_descent_cone_audit.py"
STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"
STAGE5="${ROOT}/vpinn_gradient_conflict_stage5_escape_time"
STAGE9_SCRIPT="${ROOT}/vpinn_gradient_conflict_stage9_reflected_adam_continuation.py"
STAGE9_DIR="${ROOT}/vpinn_gradient_conflict_stage9_reflected_adam_continuation"
STAGE10_SCRIPT="${ROOT}/vpinn_gradient_conflict_stage10_local_feasibility_audit.py"
STAGE10_DIR="${ROOT}/vpinn_gradient_conflict_stage10_local_feasibility_audit"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

for file in "${SCRIPT}" "${STAGE3}" "${STAGE9_SCRIPT}" "${STAGE10_SCRIPT}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing required file: ${file}" >&2
    exit 2
  fi
done

for dir in "${STAGE5}" "${STAGE9_DIR}" "${STAGE10_DIR}"; do
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
  --stage9-script "${STAGE9_SCRIPT}" \
  --stage9-dir "${STAGE9_DIR}" \
  --stage10-script "${STAGE10_SCRIPT}" \
  --stage10-dir "${STAGE10_DIR}"
