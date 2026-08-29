#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"
SCRIPT="${ROOT}/vpinn_gradient_conflict_stage9_reflected_adam_continuation.py"
STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"
STAGE5="${ROOT}/vpinn_gradient_conflict_stage5_escape_time"
STAGE8="${ROOT}/vpinn_gradient_conflict_stage8_adam_target_reflection"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

for file in "${SCRIPT}" "${STAGE3}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing required file: ${file}" >&2
    exit 2
  fi
done

for dir in "${STAGE5}" "${STAGE8}"; do
  if [[ ! -d "${dir}" ]]; then
    echo "ERROR: missing required result directory: ${dir}" >&2
    exit 2
  fi
done

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --device cpu \
  --track-interval 25 \
  --common-endpoint-epoch 2700 \
  --stage3-script "${STAGE3}" \
  --stage5-dir "${STAGE5}" \
  --stage8-dir "${STAGE8}"
