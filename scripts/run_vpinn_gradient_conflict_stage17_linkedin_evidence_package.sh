#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"

SCRIPT="${ROOT}/vpinn_gradient_conflict_stage17_linkedin_evidence_package.py"

STAGE12="${ROOT}/vpinn_gradient_conflict_stage12_common_pareto_blend_audit"
STAGE14="${ROOT}/vpinn_gradient_conflict_stage14_adaptive_midpoint_failure_audit"
STAGE15="${ROOT}/vpinn_gradient_conflict_stage15_adaptive_midpoint_persistence"
STAGE16="${ROOT}/vpinn_gradient_conflict_stage16_matched_escape_comparison"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: project Python not found/executable: ${PYTHON}" >&2
  exit 2
fi

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: missing script: ${SCRIPT}" >&2
  exit 2
fi

for dir in "${STAGE12}" "${STAGE14}" "${STAGE15}" "${STAGE16}"; do
  if [[ ! -d "${dir}" ]]; then
    echo "ERROR: missing required result directory: ${dir}" >&2
    exit 2
  fi
done

cd "${ROOT}"

exec "${PYTHON}" "${SCRIPT}" \
  --stage12-dir "${STAGE12}" \
  --stage14-dir "${STAGE14}" \
  --stage15-dir "${STAGE15}" \
  --stage16-dir "${STAGE16}"
