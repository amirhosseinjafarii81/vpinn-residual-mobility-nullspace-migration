#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"

SCRIPT="${ROOT}/vpinn_gradient_conflict_stage16_matched_escape_comparison.py"

STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"

STAGE5="${ROOT}/vpinn_gradient_conflict_stage5_escape_time"

STAGE9="${ROOT}/vpinn_gradient_conflict_stage9_reflected_adam_continuation.py"

STAGE10="${ROOT}/vpinn_gradient_conflict_stage10_local_feasibility_audit"

STAGE15_SCRIPT="${ROOT}/vpinn_gradient_conflict_stage15_adaptive_midpoint_persistence.py"
STAGE15_DIR="${ROOT}/vpinn_gradient_conflict_stage15_adaptive_midpoint_persistence"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

for file in \
  "${SCRIPT}" \
  "${STAGE3}" \
  "${STAGE9}" \
  "${STAGE15_SCRIPT}"
do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing required file: ${file}" >&2
    exit 2
  fi
done

for dir in \
  "${STAGE5}" \
  "${STAGE10}" \
  "${STAGE15_DIR}"
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
  --stage10-dir "${STAGE10}" \
  --stage15-script "${STAGE15_SCRIPT}" \
  --stage15-dir "${STAGE15_DIR}"
