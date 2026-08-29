#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

STAGE1="${SCRIPT_DIR}/vpinn_gradient_conflict_stage1.py"
STAGE2="${SCRIPT_DIR}/vpinn_gradient_conflict_stage2_seedreplication.py"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"

if [[ ! -f "${STAGE1}" ]]; then
  echo "ERROR: Stage-1 script not found: ${STAGE1}" >&2
  exit 2
fi

if [[ ! -f "${STAGE2}" ]]; then
  echo "ERROR: Stage-2 script not found: ${STAGE2}" >&2
  exit 2
fi

if [[ -x "${VENV_PYTHON}" ]]; then
  PYTHON="${VENV_PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "ERROR: No suitable Python interpreter found." >&2
  exit 2
fi

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd "${SCRIPT_DIR}"

printf '%s\n' "==> VPINN gradient-conflict Stage 2: five-seed exact replication"
printf 'python: %s\n' "${PYTHON}"
printf 'cwd:    %s\n' "${SCRIPT_DIR}"
printf '%s\n' "seeds:  [0,1,2,3,4]"
printf '%s\n' "device: cpu"
printf '%s\n' "NOTE: existing Stage-2 run folders are rerun by default for strict comparability."

exec "${PYTHON}" "${STAGE2}" \
  --stage1-script "${STAGE1}" \
  --device cpu \
  --seeds 0 1 2 3 4 \
  --output-dir "${SCRIPT_DIR}/vpinn_gradient_conflict_stage2_seedreplication"
