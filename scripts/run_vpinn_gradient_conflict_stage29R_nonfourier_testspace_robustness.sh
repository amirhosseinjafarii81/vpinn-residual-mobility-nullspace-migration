#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"

SCRIPT="${ROOT}/vpinn_gradient_conflict_stage29R_nonfourier_testspace_robustness.py"

STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"
STAGE18="${ROOT}/vpinn_gradient_conflict_stage18R_frequency_transfer.py"
STAGE20="${ROOT}/vpinn_gradient_conflict_stage20R_heldout_mobility_unlock.py"
STAGE28="${ROOT}/vpinn_gradient_conflict_stage28R_reaction_diffusion_robustness.py"
STAGE28_DIR="${ROOT}/vpinn_gradient_conflict_stage28R_reaction_diffusion_robustness"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

for file in "${SCRIPT}" "${STAGE3}" "${STAGE18}" "${STAGE20}" "${STAGE28}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing required file: ${file}" >&2
    exit 2
  fi
done

if [[ ! -d "${STAGE28_DIR}" ]]; then
  echo "ERROR: missing Stage-28 result directory: ${STAGE28_DIR}" >&2
  exit 2
fi

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --device cpu \
  --stage3-script "${STAGE3}" \
  --stage18-script "${STAGE18}" \
  --stage20-script "${STAGE20}" \
  --stage28-script "${STAGE28}" \
  --stage28-dir "${STAGE28_DIR}"
