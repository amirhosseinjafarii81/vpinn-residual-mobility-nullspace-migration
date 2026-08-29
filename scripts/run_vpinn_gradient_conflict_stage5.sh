#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"
SCRIPT="${ROOT}/vpinn_gradient_conflict_stage5_escape_time.py"
STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"
STAGE4="${ROOT}/vpinn_gradient_conflict_stage4_edge_mode_replication"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: Stage-5 script not found: ${SCRIPT}" >&2
  exit 2
fi

if [[ ! -f "${STAGE3}" ]]; then
  echo "ERROR: Stage-3 solver not found: ${STAGE3}" >&2
  exit 2
fi

if [[ ! -d "${STAGE4}" ]]; then
  echo "ERROR: Stage-4 result directory not found: ${STAGE4}" >&2
  exit 2
fi

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --device cpu \
  --seeds 0 1 2 3 4 \
  --max-epoch 4000 \
  --track-interval 25 \
  --stage3-script "${STAGE3}" \
  --stage4-dir "${STAGE4}"
