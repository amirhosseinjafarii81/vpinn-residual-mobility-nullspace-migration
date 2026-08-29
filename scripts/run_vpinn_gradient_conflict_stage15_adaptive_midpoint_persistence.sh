#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"

SCRIPT="${ROOT}/vpinn_gradient_conflict_stage15_adaptive_midpoint_persistence.py"

STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"

STAGE5="${ROOT}/vpinn_gradient_conflict_stage5_escape_time"

STAGE9_SCRIPT="${ROOT}/vpinn_gradient_conflict_stage9_reflected_adam_continuation.py"
STAGE9_DIR="${ROOT}/vpinn_gradient_conflict_stage9_reflected_adam_continuation"

STAGE10_DIR="${ROOT}/vpinn_gradient_conflict_stage10_local_feasibility_audit"
STAGE12_DIR="${ROOT}/vpinn_gradient_conflict_stage12_common_pareto_blend_audit"
STAGE13_DIR="${ROOT}/vpinn_gradient_conflict_stage13_fixed_common_blend_pilot"

STAGE14_SCRIPT="${ROOT}/vpinn_gradient_conflict_stage14_adaptive_midpoint_failure_audit.py"
STAGE14_DIR="${ROOT}/vpinn_gradient_conflict_stage14_adaptive_midpoint_failure_audit"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

for file in \
  "${SCRIPT}" \
  "${STAGE3}" \
  "${STAGE9_SCRIPT}" \
  "${STAGE14_SCRIPT}"
do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing required file: ${file}" >&2
    exit 2
  fi
done

for dir in \
  "${STAGE5}" \
  "${STAGE9_DIR}" \
  "${STAGE10_DIR}" \
  "${STAGE12_DIR}" \
  "${STAGE13_DIR}" \
  "${STAGE14_DIR}"
do
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
  --stage10-dir "${STAGE10_DIR}" \
  --stage12-dir "${STAGE12_DIR}" \
  --stage13-dir "${STAGE13_DIR}" \
  --stage14-script "${STAGE14_SCRIPT}" \
  --stage14-dir "${STAGE14_DIR}"
