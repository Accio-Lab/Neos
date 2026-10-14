#!/usr/bin/env bash
NEOS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export NEOS_ROOT PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${NEOS_ROOT}:${NEOS_ROOT}/verl${PYTHONPATH:+:$PYTHONPATH}"
source "${NEOS_ROOT}/environment/config.sh"
NEOS_DRY_RUN=0
for arg in "$@"; do
  case "$arg" in --dry-run|--help|-h) NEOS_DRY_RUN=1 ;; esac
done
if [[ "$NEOS_DRY_RUN" == 1 ]]; then
  NEOS_PYTHON="${NEOS_PYTHON:-$(command -v python3)}"
else
  source "${NEOS_ROOT}/environment/activate.sh"
  NEOS_PYTHON="${NEOS_VENV}/bin/python"
fi
export NEOS_PYTHON
