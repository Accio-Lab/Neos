# Copyright (c) 2026 BAAI. All rights reserved.
# Adopted from https://github.com/microsoft/DeepSpeed/blob/master/accelerator/cuda_accelerator.py
"""NVIDIA CUDA platform implementation."""

import os
import shutil
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch
import torch.cuda

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry


@PlatformRegistry.register(platform="nvidia")
class PlatformCUDA(PlatformBase):
    """Platform backend for NVIDIA CUDA GPUs and CUDA-compatible accelerators."""

    
    
    

    @property
    def device_name(self) -> str:
        return "cuda"

    @property
    def vendor_name(self) -> str:
        return "nvidia"

    @property
    def device_module(self) -> ModuleType:
        return torch.cuda

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def is_platform_available(self, use_smi_check=False) -> bool:
        if not hasattr(torch, "cuda"):
            return False
        
        
        if torch.version.hip is not None:
            return False
        if use_smi_check:
            
            
            
            
            cmd = "nvidia-smi"
            cmd_path = shutil.which(cmd)
            if cmd_path is None:
                
                common_paths = [
                    f"/usr/bin/{cmd}",
                    f"/usr/local/bin/{cmd}",
                    f"/usr/local/cuda/bin/{cmd}",
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

    def current_device(self) -> int:
        return torch.cuda.current_device()

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def set_device(self, device_index: int) -> None:
        torch.cuda.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        torch.cuda.synchronize(device_index)

    
    
    

    def manual_seed(self, seed: int) -> None:
        torch.cuda.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.cuda.manual_seed_all(seed)

    
    
    

    def set_allocator_settings(self, settings: str) -> None:
        torch.cuda.memory._set_allocator_settings(settings)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    
    
    

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        if not torch.cuda.is_available():
            return None, None
        return torch.cuda.get_device_capability(device_index)

    
    
    

    def communication_backend_name(self) -> str:
        return "flagcx" if os.getenv("USE_FLAGCX", "0").lower() in ["1", "true"] else "nccl"

    def visible_devices_envvar(self) -> str:
        return "CUDA_VISIBLE_DEVICES"

    
    
    

    def ray_resource_name(self) -> str:
        return "GPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"num_gpus": num_gpus}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]

    
    
    

    def is_ipc_supported(self) -> bool:
        return True

    
    
    

    def rollout_env_vars(self) -> dict[str, str]:
        
        
        
        
        return {"NCCL_CUMEM_ENABLE": "0"}

    
    
    

    def get_collective_module(self) -> Any:
        try:
            from cupy.cuda import nccl

            return nccl
        except (ImportError, ModuleNotFoundError):
            return None

    
    
    

    @contextmanager
    def nvtx_range(self, msg: str):
        with torch.cuda.nvtx.range(msg):
            yield

    def profiler_start(self) -> None:
        torch.cuda.profiler.start()

    def profiler_stop(self) -> None:
        torch.cuda.profiler.stop()

    
    
    

    def cudart(self) -> Any:
        return torch.cuda.cudart()
