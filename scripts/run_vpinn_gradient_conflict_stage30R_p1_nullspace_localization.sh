#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"

SCRIPT="${ROOT}/vpinn_gradient_conflict_stage30R_p1_nullspace_localization.py"

STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"
STAGE29="${ROOT}/vpinn_gradient_conflict_stage29R_nonfourier_testspace_robustness.py"
STAGE29_DIR="${ROOT}/vpinn_gradient_conflict_stage29R_nonfourier_testspace_robustness"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

for file in "${SCRIPT}" "${STAGE3}" "${STAGE29}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing required file: ${file}" >&2
    exit 2
  fi
done

if [[ ! -d "${STAGE29_DIR}" ]]; then
  echo "ERROR: missing Stage-29 result directory: ${STAGE29_DIR}" >&2
  exit 2
fi

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --device cpu \
  --stage3-script "${STAGE3}" \
  --stage29-script "${STAGE29}" \
  --stage29-dir "${STAGE29_DIR}"
