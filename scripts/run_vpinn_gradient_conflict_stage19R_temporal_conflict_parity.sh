#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"

SCRIPT="${ROOT}/vpinn_gradient_conflict_stage19R_temporal_conflict_parity.py"

STAGE3_SCRIPT="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"
STAGE3_DIR="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer"

STAGE18_SCRIPT="${ROOT}/vpinn_gradient_conflict_stage18R_frequency_transfer.py"
STAGE18_DIR="${ROOT}/vpinn_gradient_conflict_stage18R_frequency_transfer"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

for file in "${SCRIPT}" "${STAGE3_SCRIPT}" "${STAGE18_SCRIPT}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing required file: ${file}" >&2
    exit 2
  fi
done

for dir in "${STAGE3_DIR}" "${STAGE18_DIR}"; do
  if [[ ! -d "${dir}" ]]; then
    echo "ERROR: missing required directory: ${dir}" >&2
    exit 2
  fi
done

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --device cpu \
  --stage3-script "${STAGE3_SCRIPT}" \
  --stage3-dir "${STAGE3_DIR}" \
  --stage18-script "${STAGE18_SCRIPT}" \
  --stage18-dir "${STAGE18_DIR}"
