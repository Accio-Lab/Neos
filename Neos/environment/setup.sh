#!/usr/bin/env bash
set -Eeuo pipefail

DRY_RUN=0
REUSE_EXISTING=0
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --reuse-existing) REUSE_EXISTING=1 ;;
    --help|-h)
      printf '%s\n' 'Usage: bash environment/setup.sh [--dry-run] [--reuse-existing]' 'Set NEOS_VENV and CUDA_HOME in config.local.json first.'
      exit 0
      ;;
    *) printf 'Unknown argument: %s\n' "${arg}" >&2; exit 2 ;;
  esac
done

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/config.sh"
: "${NEOS_VENV:?Set NEOS_VENV in config.local.json}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
if command -v python3.12 >/dev/null 2>&1; then
  NEOS_SETUP_BOOTSTRAP=python3.12
else
  NEOS_SETUP_BOOTSTRAP=python3
fi
"${NEOS_SETUP_BOOTSTRAP}" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
NEOS_VENV="$("${NEOS_SETUP_BOOTSTRAP}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${NEOS_VENV}")"
if [[ "${NEOS_VENV}" == / || "${NEOS_VENV}" == "${NEOS_ROOT}" || "${NEOS_VENV}" == /usr || "${NEOS_VENV}" == /usr/local ]]; then
  echo 'Choose a dedicated virtual-environment directory.' >&2
  exit 2
fi

run() {
  if [[ "${DRY_RUN}" == 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ -e "${NEOS_VENV}" ]]; then
  if [[ "${REUSE_EXISTING}" != 1 ]]; then
    echo 'NEOS_VENV already exists. Choose a new directory or explicitly pass --reuse-existing.' >&2
    exit 2
  fi
  if [[ ! -x "${NEOS_VENV}/bin/python" ]] || { [[ ! -f "${NEOS_VENV}/pyvenv.cfg" ]] && [[ ! -d "${NEOS_VENV}/conda-meta" ]]; }; then
    echo 'The configured directory is not a virtualenv/conda environment; refusing to modify it.' >&2
    exit 2
  fi
fi

source "${NEOS_ROOT}/environment/native.sh"
if [[ "${DRY_RUN}" != 1 ]]; then
  command -v nvcc >/dev/null 2>&1 || { echo 'CUDA development toolkit not found; set CUDA_HOME.' >&2; exit 2; }
  command -v git >/dev/null 2>&1 || { echo 'git is required to build the native extensions.' >&2; exit 2; }
  command -v g++ >/dev/null 2>&1 || { echo 'A C++ compiler is required to build the native extensions.' >&2; exit 2; }
  nvcc --version | grep -q 'release 12\.9,' || { echo 'This environment targets CUDA Toolkit 12.9; set CUDA_HOME accordingly.' >&2; exit 2; }
fi
if [[ ! -e "${NEOS_VENV}" ]]; then
  run "${NEOS_SETUP_BOOTSTRAP}" -m venv --without-pip "${NEOS_VENV}"
fi
NEOS_SETUP_PYTHON="${NEOS_VENV}/bin/python"
export PATH="${NEOS_VENV}/bin:${PATH}"
export MAX_JOBS="${MAX_JOBS:-8}"
run "${NEOS_SETUP_PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 12), "The configured environment must use Python 3.12"'
if [[ "${DRY_RUN}" == 1 ]] || ! "${NEOS_SETUP_PYTHON}" -m pip --version >/dev/null 2>&1; then
  run "${NEOS_SETUP_PYTHON}" -c 'import urllib.request; source = urllib.request.urlopen("https://bootstrap.pypa.io/get-pip.py", timeout=60).read(); exec(compile(source, "get-pip.py", "exec"), {"__name__": "__main__", "__file__": "get-pip.py"})' pip==26.0.1
fi
run "${NEOS_SETUP_PYTHON}" -m pip install --upgrade -r "${NEOS_ROOT}/environment/requirements-build.txt"
run "${NEOS_SETUP_PYTHON}" -m pip install -c "${NEOS_ROOT}/environment/requirements-constraints.txt" torch==2.9.1+cu129 torchvision==0.24.1+cu129 torchaudio==2.9.1+cu129 --index-url https://download.pytorch.org/whl/cu129
if [[ "${DRY_RUN}" != 1 ]]; then NEOS_NATIVE_BUILD=1 source "${NEOS_ROOT}/environment/native.sh"; fi
run "${NEOS_SETUP_PYTHON}" -m pip install --no-build-isolation -r "${NEOS_ROOT}/environment/requirements-runtime.txt" -r "${NEOS_ROOT}/environment/requirements-support.txt"
run "${NEOS_SETUP_PYTHON}" -m pip install --no-deps --upgrade --target "${NEOS_VENV}/native/cudnn" -r "${NEOS_ROOT}/environment/requirements-native.txt"
if [[ "${DRY_RUN}" != 1 ]]; then NEOS_NATIVE_BUILD=1 source "${NEOS_ROOT}/environment/native.sh"; fi
run env APEX_CPP_EXT=1 APEX_CUDA_EXT=1 "${NEOS_SETUP_PYTHON}" -m pip install --no-deps --no-build-isolation 'apex @ git+https://github.com/NVIDIA/apex.git@b884ff5da8789691188cf8af7c06559770ddf708'
run env NVTE_FRAMEWORK=pytorch NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-1}" "${NEOS_SETUP_PYTHON}" -m pip install --no-deps --no-build-isolation 'transformer_engine @ git+https://github.com/NVIDIA/TransformerEngine.git@769ed778341a32c8c593fda391700c0a80f65f1f'
run "${NEOS_SETUP_PYTHON}" -m pip install --no-deps megatron-core==0.16.0 mcore-bridge==1.6.1
run env SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.9 "${NEOS_SETUP_PYTHON}" -m pip install --no-deps --no-build-isolation -e "${NEOS_ROOT}" -e "${NEOS_ROOT}/verl" -e "${NEOS_ROOT}/third_party/mbridge" -e "${NEOS_ROOT}/third_party/ms-swift" -e "${NEOS_ROOT}/third_party/sglang/python"

NEOS_SETUP_IMPORT_PATH="${NEOS_ROOT}:${NEOS_ROOT}/verl:${NEOS_ROOT}/third_party/mbridge:${NEOS_ROOT}/third_party/sglang/python:${NEOS_ROOT}/third_party/ms-swift:${NEOS_ROOT}/third_party/transformers:${NEOS_ROOT}/third_party/transfer_queue"
run env PYTHONPATH="${NEOS_SETUP_IMPORT_PATH}:${NEOS_ROOT}/third_party/rl_overlay" "${NEOS_SETUP_PYTHON}" -c 'import torch, sglang, transformers, ray, fla, flash_attn, flashinfer, transformer_engine.pytorch, apex, megatron.core, alibabacloud_oss_v2; assert torch.backends.cudnn.version() >= 91500, "SGLang requires cuDNN 9.15+ with Torch 2.9.1"; print("runtime imports ready; cuDNN", torch.backends.cudnn.version())'
run env PYTHONPATH="${NEOS_ROOT}/third_party/sft_overlay:${NEOS_SETUP_IMPORT_PATH}" "${NEOS_SETUP_PYTHON}" -c 'import swift, mcore_bridge, megatron.core; print("SFT imports ready")'
if [[ "${DRY_RUN}" == 1 ]]; then
  echo 'Dry run finished; no environment was created or modified.'
else
  echo 'Environment ready. Training and serving scripts load config.local.json automatically.'
fi
