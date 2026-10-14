#!/usr/bin/env bash
NEOS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export NEOS_ROOT
export PYTHONDONTWRITEBYTECODE=1
NEOS_CONFIG_EXPORTS="$(PYTHONPATH="${NEOS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 -c 'from neos.configuration import main; main()' --shell)" || { return 1 2>/dev/null || exit 1; }
eval "${NEOS_CONFIG_EXPORTS}"
unset NEOS_CONFIG_EXPORTS
