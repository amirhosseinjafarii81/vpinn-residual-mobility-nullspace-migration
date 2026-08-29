#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"

SCRIPT="${ROOT}/vpinn_gradient_conflict_stage14_adaptive_midpoint_failure_audit.py"

STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"
STAGE5="${ROOT}/vpinn_gradient_conflict_stage5_escape_time"

STAGE9="${ROOT}/vpinn_gradient_conflict_stage9_reflected_adam_continuation.py"
STAGE12="${ROOT}/vpinn_gradient_conflict_stage12_common_pareto_blend_audit.py"

STAGE13_SCRIPT="${ROOT}/vpinn_gradient_conflict_stage13_fixed_common_blend_pilot.py"
STAGE13_DIR="${ROOT}/vpinn_gradient_conflict_stage13_fixed_common_blend_pilot"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

for file in \
  "${SCRIPT}" \
  "${STAGE3}" \
  "${STAGE9}" \
  "${STAGE12}" \
  "${STAGE13_SCRIPT}"
do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing required file: ${file}" >&2
    exit 2
  fi
done

for dir in \
  "${STAGE5}" \
  "${STAGE13_DIR}"
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
  --stage9-script "${STAGE9}" \
  --stage12-script "${STAGE12}" \
  --stage13-script "${STAGE13_SCRIPT}" \
  --stage13-dir "${STAGE13_DIR}"
