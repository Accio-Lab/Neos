#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_environment.sh" "$@"
exec "$NEOS_PYTHON" -m neos.serving.service --kind mei "$@"
