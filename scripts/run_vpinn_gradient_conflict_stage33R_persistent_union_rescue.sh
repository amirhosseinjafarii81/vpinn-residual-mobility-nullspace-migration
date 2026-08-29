#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"

SCRIPT="${ROOT}/vpinn_gradient_conflict_stage33R_persistent_union_rescue.py"

STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"
STAGE29="${ROOT}/vpinn_gradient_conflict_stage29R_nonfourier_testspace_robustness.py"
STAGE29_DIR="${ROOT}/vpinn_gradient_conflict_stage29R_nonfourier_testspace_robustness"
STAGE31="${ROOT}/vpinn_gradient_conflict_stage31R_minimal_P1_refinement_rescue.py"
STAGE31_DIR="${ROOT}/vpinn_gradient_conflict_stage31R_minimal_P1_refinement_rescue"
STAGE32="${ROOT}/vpinn_gradient_conflict_stage32R_nullspace_migration_audit.py"
STAGE32_DIR="${ROOT}/vpinn_gradient_conflict_stage32R_nullspace_migration_audit"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

for file in "${SCRIPT}" "${STAGE3}" "${STAGE29}" "${STAGE31}" "${STAGE32}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing required file: ${file}" >&2
    exit 2
  fi
done

for dir in "${STAGE29_DIR}" "${STAGE31_DIR}" "${STAGE32_DIR}"; do
  if [[ ! -d "${dir}" ]]; then
    echo "ERROR: missing required result directory: ${dir}" >&2
    exit 2
  fi
done

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --device cpu \
  --stage3-script "${STAGE3}" \
  --stage29-script "${STAGE29}" \
  --stage29-dir "${STAGE29_DIR}" \
  --stage31-script "${STAGE31}" \
  --stage31-dir "${STAGE31_DIR}" \
  --stage32-script "${STAGE32}" \
  --stage32-dir "${STAGE32_DIR}"
