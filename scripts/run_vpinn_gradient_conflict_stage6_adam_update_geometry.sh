#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"
SCRIPT="${ROOT}/vpinn_gradient_conflict_stage6_adam_update_geometry.py"
STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"
STAGE5_SCRIPT="${ROOT}/vpinn_gradient_conflict_stage5_escape_time.py"
STAGE5="${ROOT}/vpinn_gradient_conflict_stage5_escape_time"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: Stage-6 script not found: ${SCRIPT}" >&2
  exit 2
fi

if [[ ! -f "${STAGE3}" ]]; then
  echo "ERROR: Stage-3 solver not found: ${STAGE3}" >&2
  exit 2
fi

if [[ ! -f "${STAGE5_SCRIPT}" ]]; then
  echo "ERROR: Stage-5 script not found: ${STAGE5_SCRIPT}" >&2
  exit 2
fi

if [[ ! -d "${STAGE5}" ]]; then
  echo "ERROR: Stage-5 result directory not found: ${STAGE5}" >&2
  exit 2
fi

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --device cpu \
  --track-interval 25 \
  --stage3-script "${STAGE3}" \
  --stage5-script "${STAGE5_SCRIPT}" \
  --stage5-dir "${STAGE5}"
