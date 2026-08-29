#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"

SCRIPT="${ROOT}/vpinn_gradient_conflict_stage22R_symmetry_sector_swap.py"

STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"
STAGE18="${ROOT}/vpinn_gradient_conflict_stage18R_frequency_transfer.py"
STAGE19="${ROOT}/vpinn_gradient_conflict_stage19R_temporal_conflict_parity.py"
STAGE20="${ROOT}/vpinn_gradient_conflict_stage20R_heldout_mobility_unlock.py"
STAGE21="${ROOT}/vpinn_gradient_conflict_stage21R_even_frequency_mobility_transfer.py"
STAGE21_DIR="${ROOT}/vpinn_gradient_conflict_stage21R_even_frequency_mobility_transfer"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

for file in "${SCRIPT}" "${STAGE3}" "${STAGE18}" "${STAGE19}" "${STAGE20}" "${STAGE21}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing required file: ${file}" >&2
    exit 2
  fi
done

if [[ ! -d "${STAGE21_DIR}" ]]; then
  echo "ERROR: missing Stage-21 result directory: ${STAGE21_DIR}" >&2
  exit 2
fi

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --device cpu \
  --stage3-script "${STAGE3}" \
  --stage18-script "${STAGE18}" \
  --stage19-script "${STAGE19}" \
  --stage20-script "${STAGE20}" \
  --stage21-script "${STAGE21}" \
  --stage21-dir "${STAGE21_DIR}"
