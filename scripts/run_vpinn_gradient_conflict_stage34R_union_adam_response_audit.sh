#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"

SCRIPT="${ROOT}/vpinn_gradient_conflict_stage34R_union_adam_response_audit.py"

STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"
STAGE29="${ROOT}/vpinn_gradient_conflict_stage29R_nonfourier_testspace_robustness.py"
STAGE31="${ROOT}/vpinn_gradient_conflict_stage31R_minimal_P1_refinement_rescue.py"
STAGE33="${ROOT}/vpinn_gradient_conflict_stage33R_persistent_union_rescue.py"
STAGE33_DIR="${ROOT}/vpinn_gradient_conflict_stage33R_persistent_union_rescue"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

for file in "${SCRIPT}" "${STAGE3}" "${STAGE29}" "${STAGE31}" "${STAGE33}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing required file: ${file}" >&2
    exit 2
  fi
done

if [[ ! -d "${STAGE33_DIR}" ]]; then
  echo "ERROR: missing Stage-33 result directory: ${STAGE33_DIR}" >&2
  exit 2
fi

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --device cpu \
  --stage3-script "${STAGE3}" \
  --stage29-script "${STAGE29}" \
  --stage31-script "${STAGE31}" \
  --stage33-script "${STAGE33}" \
  --stage33-dir "${STAGE33_DIR}"
