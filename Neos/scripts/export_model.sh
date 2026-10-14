#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_environment.sh" "$@"
if [[ "$NEOS_DRY_RUN" != 1 ]]; then source "${NEOS_ROOT}/environment/activate_sft.sh"; fi
exec "$NEOS_PYTHON" -m neos.training.export "$@"
