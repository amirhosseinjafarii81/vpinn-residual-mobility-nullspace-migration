#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"

SCRIPT="${ROOT}/vpinn_gradient_conflict_stage23R_censoring_repair_continuation.py"

STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"
STAGE18="${ROOT}/vpinn_gradient_conflict_stage18R_frequency_transfer.py"
STAGE19="${ROOT}/vpinn_gradient_conflict_stage19R_temporal_conflict_parity.py"
STAGE20="${ROOT}/vpinn_gradient_conflict_stage20R_heldout_mobility_unlock.py"
STAGE22="${ROOT}/vpinn_gradient_conflict_stage22R_symmetry_sector_swap.py"
STAGE22_DIR="${ROOT}/vpinn_gradient_conflict_stage22R_symmetry_sector_swap"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

for file in "${SCRIPT}" "${STAGE3}" "${STAGE18}" "${STAGE19}" "${STAGE20}" "${STAGE22}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing required file: ${file}" >&2
    exit 2
  fi
done

if [[ ! -d "${STAGE22_DIR}" ]]; then
  echo "ERROR: missing Stage-22 result directory: ${STAGE22_DIR}" >&2
  exit 2
fi

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --device cpu \
  --stage3-script "${STAGE3}" \
  --stage18-script "${STAGE18}" \
  --stage19-script "${STAGE19}" \
  --stage20-script "${STAGE20}" \
  --stage22-script "${STAGE22}" \
  --stage22-dir "${STAGE22_DIR}"
