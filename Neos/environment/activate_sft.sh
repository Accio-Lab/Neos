#!/usr/bin/env bash
                            
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/activate.sh" || return 1
export PYTHONPATH="${NEOS_ROOT}/third_party/sft_overlay:${PYTHONPATH}"
export NEOS_QWEN35_MEGATRON_ENV_ROOT="${NEOS_ROOT}/third_party/sft_overlay"
export NEOS_RUNTIME_FLAVOR=qwen35-megatron
export USE_MCORE_GDN="${USE_MCORE_GDN:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
