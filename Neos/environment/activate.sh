#!/usr/bin/env bash
                                           
NEOS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export NEOS_ROOT
source "${NEOS_ROOT}/environment/config.sh" || { return 1 2>/dev/null || exit 1; }
if [[ -z "${NEOS_VENV:-}" ]]; then
    echo "Fill NEOS_VENV in Neos/config.local.json before launching." >&2
    return 1 2>/dev/null || exit 1
fi
if [[ ! -x "${NEOS_VENV}/bin/python" ]]; then
    echo "NEOS_VENV must point to an existing Python3.12 environment." >&2
    return 1 2>/dev/null || exit 1
fi
export PATH="${NEOS_VENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
source "${NEOS_ROOT}/environment/native.sh" || { return 1 2>/dev/null || exit 1; }
export PYTHONPATH="${NEOS_ROOT}:${NEOS_ROOT}/verl:${NEOS_ROOT}/third_party/mbridge:${NEOS_ROOT}/third_party/sglang/python:${NEOS_ROOT}/third_party/ms-swift:${NEOS_ROOT}/third_party/transformers:${NEOS_ROOT}/third_party/transfer_queue:${NEOS_ROOT}/third_party/rl_overlay${PYTHONPATH:+:${PYTHONPATH}}"
export NEOS_RUNTIME_FLAVOR=qwen35
export NEOS_QWEN35_ENV_ROOT="${NEOS_ROOT}/environment"
export TMPDIR="${TMPDIR:-/tmp/neos}"
: "${NEOS_OUTPUT_ROOT:?Fill NEOS_OUTPUT_ROOT in Neos/config.local.json}"
export HF_HOME="${HF_HOME:-${NEOS_OUTPUT_ROOT}/model-cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${NEOS_OUTPUT_ROOT}/model-cache/modelscope}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR}/inductor}"
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-${TMPDIR}/torch-kernels}"
export NEOS_EVAL_ARTIFACT_ROOT="${NEOS_EVAL_ARTIFACT_ROOT:-${NEOS_OUTPUT_ROOT}/artifacts}"
export CUSTOM_CHAT_TEMPLATE_PATH="${CUSTOM_CHAT_TEMPLATE_PATH:-${NEOS_ROOT}/configs/qwen_vl_tool_chat_template.jinja}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${TMPDIR}" "${HF_DATASETS_CACHE}" "${MODELSCOPE_CACHE}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${PYTORCH_KERNEL_CACHE_PATH}"
