#!/usr/bin/env bash

if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname -- "$(dirname -- "$(readlink -f "$(command -v nvcc)")")")"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda
  fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

if [[ ! -x "${NEOS_VENV}/bin/python" ]]; then
  return 0 2>/dev/null || exit 0
fi

NEOS_NATIVE_EXPORTS="$("${NEOS_VENV}/bin/python" -B - <<'PY'
import os
from pathlib import Path
import shlex
import sys
import sysconfig

site = Path(sysconfig.get_path('purelib'))
cudnn = Path(sys.prefix) / 'native' / 'cudnn' / 'nvidia' / 'cudnn'
if not (cudnn / 'lib').is_dir():
    cudnn = site / 'nvidia' / 'cudnn'
preferred = [cudnn / 'lib'] + [site / 'nvidia' / name / 'lib'
             for name in ('cudnn', 'cublas', 'cuda_runtime', 'cuda_nvrtc')]
preferred += [site / 'torch' / 'lib', *sorted((site / 'nvidia').glob('*/lib'))]
libraries = list(dict.fromkeys(str(path) for path in preferred if path.is_dir()))
if libraries:
    value = ':'.join(libraries + [p for p in os.getenv('LD_LIBRARY_PATH', '').split(':') if p])
    print('export LD_LIBRARY_PATH=' + shlex.quote(value))
if (cudnn / 'include').is_dir():
    for key, path in (('CUDNN_PATH', cudnn), ('CUDNN_INCLUDE_DIR', cudnn / 'include'),
                      ('CUDNN_LIBRARY_DIR', cudnn / 'lib')):
        print('export ' + key + '=' + shlex.quote(str(path)))
PY
)" || { return 1 2>/dev/null || exit 1; }
eval "${NEOS_NATIVE_EXPORTS}"
unset NEOS_NATIVE_EXPORTS

if [[ "${NEOS_NATIVE_BUILD:-0}" == 1 ]]; then
  NEOS_ARCH_EXPORTS="$("${NEOS_VENV}/bin/python" -B - <<'PY'
import os
import shlex
import torch

if torch.cuda.is_available():
    capabilities = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
    targets = {
        'TORCH_CUDA_ARCH_LIST': ';'.join(f'{major}.{minor}' for major, minor in capabilities),
        'NVTE_CUDA_ARCHS': ';'.join(f'{major}{minor}' for major, minor in capabilities),
        'FLASH_ATTN_CUDA_ARCHS': ';'.join(sorted({f'{major}0' for major, _ in capabilities})),
    }
    for key, value in targets.items():
        if not os.getenv(key):
            print('export ' + key + '=' + shlex.quote(value))
PY
)" || { return 1 2>/dev/null || exit 1; }
  eval "${NEOS_ARCH_EXPORTS}"
  unset NEOS_ARCH_EXPORTS
fi
