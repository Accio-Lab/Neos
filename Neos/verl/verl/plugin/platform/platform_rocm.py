# Copyright (c) 2026 BAAI. All rights reserved.
"""AMD ROCm/HIP platform implementation.

ROCm is largely CUDA-compatible: PyTorch on ROCm reuses the ``torch.cuda.*``
API surface via hipify, so most of ``PlatformCUDA`` works unchanged. This class
therefore subclasses ``PlatformCUDA`` and only overrides the parts that differ
on ROCm, following a "CUDA base + ROCm extensions" model (always extend the
parent via ``super()`` rather than re-implementing it).
"""

import os
import shutil

import torch

from .platform_cuda import PlatformCUDA
from .platform_manager import PlatformRegistry


@PlatformRegistry.register(platform="amd")
class PlatformROCm(PlatformCUDA):
    """Platform backend for AMD ROCm/HIP GPUs (reuses PlatformCUDA where compatible)."""

    @property
    def vendor_name(self) -> str:
        
        
        return "amd"

    def is_platform_available(self, use_smi_check=False) -> bool:
        if not hasattr(torch, "cuda"):
            return False
        
        if torch.version.hip is None:
            return False
        if use_smi_check:
            
            
            
            
            cmd = "rocm-smi"
            cmd_path = shutil.which(cmd)
            if cmd_path is None:
                
                common_paths = [
                    f"/usr/bin/{cmd}",
                    f"/usr/local/bin/{cmd}",
                    f"/opt/rocm/bin/{cmd}",
                ]
                for path in common_paths:
                    if os.path.isfile(path) and os.access(path, os.X_OK):
                        cmd_path = path
                        break
                if cmd_path is None:
                    return False
            if self.check_smi_command(cmd_path):
                return True
        return torch.cuda.is_available()

    def rollout_env_vars(self) -> dict[str, str]:
        
        
        
        
        return {
            **super().rollout_env_vars(),
            "SGLANG_USE_AITER": os.environ.get("SGLANG_USE_AITER", "1"),
        }

    def ray_noset_envvars(self) -> list[str]:
        
        
        return super().ray_noset_envvars() + [
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
            "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        ]
