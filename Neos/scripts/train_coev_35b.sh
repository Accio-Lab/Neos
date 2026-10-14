#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_environment.sh" "$@"
exec "$NEOS_PYTHON" -m neos.training.coevolution --config "${NEOS_ROOT}/configs/training/coev_35b.json" "$@"
