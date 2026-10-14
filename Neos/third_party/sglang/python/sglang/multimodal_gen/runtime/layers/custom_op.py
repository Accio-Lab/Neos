# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/custom_op.py

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
_is_cuda = current_platform.is_cuda()


class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    def __init__(self) -> None:
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs) -> Any:
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs) -> Any:
        """PyTorch-native implementation of the forward method.
        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs) -> Any:
        
        return self.forward_cuda(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> Any:
        
        return self.forward_cuda(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> Any:
        
        
        
        return self.forward_native(*args, **kwargs)

    def forward_musa(self, *args, **kwargs) -> Any:
        
        
        
        return self.forward_native(*args, **kwargs)

    def forward_oot(self, *args, **kwargs) -> Any:
        
        
        return self.forward_native(*args, **kwargs)

    def forward_npu(self, *args, **kwargs) -> Any:
        
        
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self) -> Callable:
        if _is_cuda:
            return self.forward_cuda
        elif current_platform.is_hip():
            return self.forward_hip
        elif current_platform.is_npu():
            return self.forward_npu
        elif current_platform.is_xpu():
            return self.forward_xpu
        elif current_platform.is_musa():
            return self.forward_musa
        else:
            return self.forward_native

    @classmethod
    def enabled(cls) -> bool:
        
        return True

    @staticmethod
    def default_on() -> bool:
        """
        On by default if level < CompilationLevel.PIECEWISE
        Specifying 'all' or 'none' in custom_op takes precedence.
        """
        raise NotImplementedError

    
    
    
    
    
    op_registry: dict[str, type["CustomOp"]] = {}

    
    @classmethod
    def register(cls, name: str) -> Callable:

        def decorator(op_cls):
            assert name not in cls.op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            cls.op_registry[name] = op_cls
            return op_cls

        return decorator
