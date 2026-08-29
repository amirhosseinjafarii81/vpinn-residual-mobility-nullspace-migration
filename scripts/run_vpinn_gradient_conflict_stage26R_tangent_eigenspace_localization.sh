#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"

SCRIPT="${ROOT}/vpinn_gradient_conflict_stage26R_tangent_eigenspace_localization.py"

STAGE22="${ROOT}/vpinn_gradient_conflict_stage22R_symmetry_sector_swap.py"
STAGE22_DIR="${ROOT}/vpinn_gradient_conflict_stage22R_symmetry_sector_swap"

STAGE25="${ROOT}/vpinn_gradient_conflict_stage25R_minimal_horizon_closure.py"
STAGE25_DIR="${ROOT}/vpinn_gradient_conflict_stage25R_minimal_horizon_closure"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

for file in "${SCRIPT}" "${STAGE22}" "${STAGE25}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing required file: ${file}" >&2
    exit 2
  fi
done

for dir in "${STAGE22_DIR}" "${STAGE25_DIR}"; do
  if [[ ! -d "${dir}" ]]; then
    echo "ERROR: missing required result directory: ${dir}" >&2
    exit 2
  fi
done

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --stage22-script "${STAGE22}" \
  --stage22-dir "${STAGE22_DIR}" \
  --stage25-script "${STAGE25}" \
  --stage25-dir "${STAGE25_DIR}"
