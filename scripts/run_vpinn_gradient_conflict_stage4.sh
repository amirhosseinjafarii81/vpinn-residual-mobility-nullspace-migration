#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"
SCRIPT="${ROOT}/vpinn_gradient_conflict_stage4_edge_mode_replication.py"
STAGE3="${ROOT}/vpinn_gradient_conflict_stage3_frequency_transfer.py"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: Stage-4 script not found: ${SCRIPT}" >&2
  exit 2
fi

if [[ ! -f "${STAGE3}" ]]; then
  echo "ERROR: Stage-3 solver not found: ${STAGE3}" >&2
  exit 2
fi

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --device cpu \
  --seeds 0 1 2 3 4 \
  --epochs 2500 \
  --track-interval 25 \
  --diagnostic-epochs 0 50 100 250 500 1000 1500 2000 2500
