# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import math
from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    if version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available():
        HAVE_TRITON = False
    else:
        HAVE_TRITON = tl.constexpr(version.parse(triton.__version__) >= version.parse("2.0.0"))
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    triton.heuristics = null_decorator
    tl = MagicMock()



@triton.jit
def _indices_to_multihot_kernel(
    indices_ptr,
    probs_in_indices_ptr,
    multihot_indices_ptr,  
    probs_in_multihot_ptr,
    position_map_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    '''
    Triton kernel for converting indices to multihot representation.

    Input:
        indices: [num_of_tokens, topk]
        probs_in_indices: [num_of_tokens, topk]
    Output:
        multihot_indices: [num_of_tokens, num_of_local_experts]
        probs_in_multihot: [num_of_tokens, num_of_local_experts]

    Assume that topk = 2 , num_of_local_experts = 4, num_of_tokens = 2,
    then the kernel can process the following conversion:

    Input Example:
        indices = [
                [0, 1],
                [1, 2]
            ]
        probs_in_indices = [
                [0.1, 0.2],
                [0.3, 0.4]
            ]
    Output Example:
        multihot_indices = [
                [1, 1, -1, -1],
                [-1, 1, 1, -1]
            ]
        probs_in_multihot = [
                [0.1, 0.2, 0.0, 0.0],
                [0.0, 0.3, 0.4, 0.0]
            ]
    '''
    
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1
    
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    
    row_idx = tl.program_id(0)
    indices_row = tl.load(indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)
    indices_row = tl.where(topk_row_mask, indices_row, -1)
    probs_row = tl.load(probs_in_indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)

    
    position_row = tl.where(indices_row != -1, topk_row, -1)
    
    mask = (indices_row != -1) & (indices_row < num_of_local_experts)

    row_idx_offset = row_idx * num_of_local_experts
    
    tl.store(multihot_indices_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask)
    tl.store(probs_in_multihot_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask)
    tl.store(position_map_ptr + row_idx_offset + num_exp_row, -1, mask=num_exp_row_mask)
    
    tl.debug_barrier()
    
    tl.store(multihot_indices_ptr + row_idx_offset + indices_row, 1, mask)
    tl.store(probs_in_multihot_ptr + row_idx_offset + indices_row, probs_row, mask)
    
    tl.store(position_map_ptr + row_idx_offset + indices_row, position_row, mask)



@triton.jit
def _multihot_to_indices_kernel(
    probs_in_multihot_ptr,
    position_map_ptr,
    probs_indices_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    '''
    Triton kernel for converting multihot representation to indices.

    Input:
        probs_in_multihot: [num_of_tokens, num_of_local_experts]
        position_map: [num_of_tokens, num_of_local_experts]
    Output:
        probs_indices: [num_of_tokens, topk]

    Assume that topk = 2 , num_of_local_experts = 4, num_of_tokens = 2,
    then the kernel can process the following conversion:

    Input Example:
        probs_in_multihot = [
                [0.7, 0.8, 0.0, 0.0],
                [0.0, 0.1, 0.9, 0.0]
            ]
        position_map = [
                [1, 1, -1, -1],
                [-1, 1, 1, -1]
            ]
    Output Example:
        probs_indices = [
                [0.7, 0.8],
                [0.1, 0.9]
            ]
    '''
    
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1
    
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    
    row_idx = tl.program_id(0)
    ptr_offset = row_idx * num_of_local_experts + num_exp_row
    probs_in_multihot_row = tl.load(probs_in_multihot_ptr + ptr_offset, mask=num_exp_row_mask)

    
    position_map_row = tl.load(position_map_ptr + ptr_offset, mask=num_exp_row_mask)
    position_map_row = tl.where(num_exp_row_mask, position_map_row, -1)
    mask = position_map_row != -1

    
    tl.store(probs_indices_ptr + row_idx * topk + topk_row, 0, mask=topk_row_mask)
    
    tl.debug_barrier()
    
    tl.store(probs_indices_ptr + row_idx * topk + position_map_row, probs_in_multihot_row, mask)


class IndicesToMultihot(torch.autograd.Function):
    """Convert moe topk indices to multihot representation.

    This class implements a custom forward and backward propagation
    operation for efficiently converting indices to multihot
    representation.
    It is an experimental feature and may change in future versions.
    """

    @staticmethod
    def forward(ctx, indices, probs_indices, num_of_local_experts):
        '''Forward function for IndicesToMultihot

        Convert indices to multihot representation.

        Args:
            indices: [num_of_tokens, topk]
            probs_indices: [num_of_tokens, topk]
            num_of_local_experts: int

        Returns:
            multihot_indices: [num_of_tokens, num_of_local_experts]
            probs_in_multihot: [num_of_tokens, num_of_local_experts]
        '''
        num_of_tokens = indices.shape[0]
        assert (
            indices.shape == probs_indices.shape
        ), "indices and probs_indices must have the same shape"
        topk = indices.shape[1]
        multihot_indices = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=torch.bool, device="cuda"
        )
        probs_in_multihot = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=probs_indices.dtype, device="cuda"
        )
        position_map = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=torch.int32, device="cuda"
        )
        
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(math.ceil(math.log2(num_of_local_experts)))
        grid = (num_of_tokens,)
        _indices_to_multihot_kernel[grid](
            indices,
            probs_indices,
            multihot_indices,
            probs_in_multihot,
            position_map,
            num_of_local_experts,
            num_of_local_experts_next_power_of_2,
            topk,
            topk_next_power_of_2,
            BLOCK_SIZE=32,  
            num_warps=1,
        )

        ctx.save_for_backward(position_map)
        ctx.num_of_tokens = num_of_tokens
        ctx.num_of_local_experts = num_of_local_experts
        ctx.topk = topk
        return multihot_indices, probs_in_multihot

    @staticmethod
    def backward(ctx, grad_multihot_indices, grad_probs_in_multihot):
        '''Backward function for IndicesToMultihot

        Convert multihot probs representation to indices.
        indices is ignored in the backward function.

        Args:
            grad_multihot_indices: [num_of_tokens, num_of_local_experts]
            grad_probs_in_multihot: [num_of_tokens, num_of_local_experts]

        Returns:
            grad_probs_indices: [num_of_tokens, topk]
        '''
        position_map = ctx.saved_tensors[0]
        num_of_tokens = ctx.num_of_tokens
        num_of_local_experts = ctx.num_of_local_experts
        topk = ctx.topk

        
        grad_probs_indices = torch.empty(
            (num_of_tokens, topk), dtype=grad_probs_in_multihot.dtype, device="cuda"
        )
        
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(math.ceil(math.log2(num_of_local_experts)))

        grid = (num_of_tokens,)
        _multihot_to_indices_kernel[grid](
            
            
            grad_probs_in_multihot.contiguous(),
            position_map,
            grad_probs_indices,
            num_of_local_experts,
            num_of_local_experts_next_power_of_2,
            topk,
            topk_next_power_of_2,
            BLOCK_SIZE=32,  
            num_warps=1,
        )
        return None, grad_probs_indices, None, None


def fused_indices_to_multihot(indices, probs_indices, num_of_local_experts):
    """Convert moe topk indices to multihot representation.

    This function is an experimental feature and may change in future versions.
    """
    return IndicesToMultihot.apply(indices, probs_indices, num_of_local_experts)
