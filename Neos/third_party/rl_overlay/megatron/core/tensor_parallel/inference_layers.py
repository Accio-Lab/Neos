# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.inference.communication.torch_symm_triton import (
    fused_multimem_rs_add_norm_ag,
    multimem_all_gather,
    multimem_reduce_scatter,
)
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import get_global_symmetric_memory_buffer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_tensor_model_parallel_group_if_none

try:
    import transformer_engine.pytorch.cpp_extensions as tex
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine.pytorch.distributed import (
        gather_along_first_dim,
        reduce_scatter_along_first_dim,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def _te_rms_norm_kernel(x: torch.Tensor, weight: torch.Tensor, eps: float):
    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    out, _, _ = tex.rmsnorm_fwd(
        x, weight, eps, None, None, TE_DType[x.dtype], 16, False  
    )
    out = out.view(*x_shape[:-1], -1)
    return out.to(x.dtype)


class InferenceLayerNormColumnParallelLinear(TELayerNormColumnParallelLinear):
    """
    Inference optimized version of TELayerNormColumnParallelLinear.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        stride: int = 1,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        assert HAVE_TE, "--transformer-impl=inference_optimized requires transformer engine"
        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            stride=stride,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
        )
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self.tp_size = dist.get_world_size(self.tp_group)

        assert (
            output_size % self.tp_size == 0
        ), f"output_size ({output_size}) must be divisible by tp_size ({self.tp_size})"

        self.eps = config.layernorm_epsilon

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--transformer-impl=inference_optimized requires --sequence-parallel"

        
        
        
        
        self.skip_norm_and_all_gather = False

    def _maybe_allocate_symmetric_buffer(self, x: torch.Tensor):
        """
        Attempt to allocate symmetric memory buffer for all-gather.
        """
        symm_mem_buffer_dims = list(x.size())
        symm_mem_buffer_dims[0] *= self.tp_size
        symm_mem_buffer = get_global_symmetric_memory_buffer().maybe_get_tensor(
            symm_mem_buffer_dims, dtype=x.dtype
        )
        return symm_mem_buffer

    def _all_gather(self, x: torch.Tensor, symm_mem_buffer: dict) -> None:
        """
        Attempt an NVLS all-gather into symmetric memory. If not possible,
        revert to torch dist (NCCL) all-gather.
        """
        if self.tp_size == 1:
            return x

        
        is_bf16 = x.dtype == torch.bfloat16
        
        is_hopper_or_newer = torch.cuda.get_device_properties(x.device).major >= 9
        
        has_enough_symmetric_memory = symm_mem_buffer["handle"] is not None
        can_use_custom_nvls_collectives = (
            is_bf16 and is_hopper_or_newer and has_enough_symmetric_memory
        )
        if can_use_custom_nvls_collectives:
            
            multimem_all_gather(symm_mem_buffer["tensor"], x, symm_mem_buffer["handle"])
            return symm_mem_buffer["tensor"]
        else:
            
            x, _ = gather_along_first_dim(x, process_group=self.tp_group)
            return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Forward pass.
        """
        
        
        
        
        
        symm_mem_buffer = self._maybe_allocate_symmetric_buffer(x)
        is_in_fused_mode = (
            self.skip_norm_and_all_gather
            and self.tp_size > 1
            and symm_mem_buffer["handle"] is not None
        )
        if is_in_fused_mode:
            x = symm_mem_buffer["tensor"]
        else:
            x = _te_rms_norm_kernel(x=x, weight=self.layer_norm_weight, eps=self.eps)
            x = self._all_gather(x, symm_mem_buffer)

        x = torch.matmul(x, self.weight.t())

        return x, None


class InferenceRowParallelLinear(TERowParallelLinear):
    """
    Inference optimized version of TERowParallelLinear.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        assert HAVE_TE, "--transformer-impl=inference_optimized requires transformer engine"
        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
        )
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self.tp_size = dist.get_world_size(self.tp_group)
        assert (
            input_size % self.tp_size == 0
        ), f"input_size ({input_size}) must be divisible by tp_size ({self.tp_size})"

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--transformer-impl=inference_optimized requires --sequence-parallel"

        
        
        self.next_layer_norm_weights = None
        self.config = config

    def _matmul_reduce_scatter(self, x, residual=None):
        """
        Multiplies x by the weight matrix and performs a reduce-scatter.
        It will first try to write the matmul output to symmetric memory
        and perform an NVLS multicast reduce-scatter. If that is not possible,
        it will revert to torch.dist (NCCL) reduce-scatter.
        """
        
        is_bf16 = x.dtype == torch.bfloat16
        
        is_hopper_or_newer = torch.cuda.get_device_properties(x.device).major >= 9
        
        symm_mem_buffer_dims = list(x.size())
        symm_mem_buffer_dims[-1] = self.weight.size(0)
        symm_mem_buffer = get_global_symmetric_memory_buffer().maybe_get_tensor(
            symm_mem_buffer_dims, dtype=x.dtype
        )
        has_enough_symmetric_memory = symm_mem_buffer["handle"] is not None
        can_use_custom_nvls_collectives = (
            is_bf16 and is_hopper_or_newer and has_enough_symmetric_memory
        )
        if can_use_custom_nvls_collectives:
            
            torch.matmul(x, self.weight.t(), out=symm_mem_buffer["tensor"])
            x = symm_mem_buffer["tensor"]
            
            if self.next_layer_norm_weights is None:
                output_dims = list(x.size())
                output_dims[0] = x.size(0) // self.tp_size
                output = torch.empty(output_dims, dtype=x.dtype, device=x.device)
                multimem_reduce_scatter(output, x, symm_mem_buffer["handle"])
                return output
            else:
                assert hasattr(self, "residual"), (
                    "For fused reduce-scatter + add + rms-norm + all-gather, "
                    "residual must be set via _set_residual()"
                )
                residual = self.residual
                fused_multimem_rs_add_norm_ag(
                    residual,
                    symm_mem_buffer["tensor"],
                    symm_mem_buffer["handle"],
                    residual,
                    self.next_layer_norm_weights,
                    self.config.layernorm_epsilon,
                )
                
                
                
                
                
                return residual
        else:
            
            x = torch.matmul(x, self.weight.t())
            x, _ = reduce_scatter_along_first_dim(x, tp_group=self.tp_group)
        return x

    def _set_next_layer_norm_weights(self, weights: torch.Tensor):
        """
        Set next layer norm weights for fused reduce-scatter + add + rms-norm + all-gather.
        """
        self.next_layer_norm_weights = weights

    def _set_residual(self, residual: torch.Tensor):
        """
        Set residual for fused reduce-scatter + add + rms-norm + all-gather.
        """
        self.residual = residual

    @torch.no_grad()
    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        """
        if self.tp_size == 1:
            x = torch.matmul(x, self.weight.t())
            return x, None
        else:
            x = self._matmul_reduce_scatter(x)
            return x, None
