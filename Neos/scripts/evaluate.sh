#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_environment.sh" "$@"
source "${NEOS_ROOT}/environment/tools.sh"
export NEOS_JUDGE_MODEL="${NEOS_JUDGE_MODEL:-qwen3.7-plus}"
export NEOS_MEI_MODEL_SPEC="${NEOS_MEI_MODEL_SPEC:-local/neos-mei}"
export NEOS_MEI_FALLBACK_MODELS="${NEOS_MEI_FALLBACK_MODELS-}"
export NEOS_LOCAL_MEI_ENABLE_THINKING=0
export NEOS_MEI_MAX_COMPLETION_TOKENS=4096
exec "$NEOS_PYTHON" -m neos.workflow.eval --execution-scheduler turn_event \
  --workers 128 --trajectory-window 128 --repeat-count 4 --max-turns 50 \
  --context-length 262144 --model-native-context-length 262144 \
  --assistant-format-profile qwen35_native_compat --tool-schema-version native_array \
  --format-failure-rollout-policy score_first_attempt "$@"
