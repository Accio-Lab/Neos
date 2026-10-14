# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.jit import jit_fuser
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
def _pad_routing_map_kernel(
    routing_map_ptr, output_ptr, num_tokens, pad_multiple: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    expert_idx = tl.program_id(axis=0)

    
    row_offset = expert_idx * num_tokens
    input_row_ptr = routing_map_ptr + row_offset
    output_row_ptr = output_ptr + row_offset

    
    token_indices = tl.arange(0, BLOCK_SIZE)
    token_mask = token_indices < num_tokens

    
    row = tl.load(input_row_ptr + token_indices, mask=token_mask, other=0)

    
    
    
    num_ones = tl.sum(row.to(tl.int32), axis=0)

    
    remainder = num_ones % pad_multiple
    num_to_pad = tl.where(remainder != 0, pad_multiple - remainder, 0)

    
    is_zero = row == 0
    
    zero_ranks = tl.cumsum(is_zero.to(tl.int32), axis=0)

    
    
    mask_to_flip = (zero_ranks <= num_to_pad) & is_zero

    
    output_row = tl.where(mask_to_flip, 1, row)

    
    tl.store(output_row_ptr + token_indices, output_row, mask=token_mask)


@jit_fuser
def fused_pad_routing_map(routing_map: torch.Tensor, pad_multiple: int) -> torch.Tensor:
    """Fused version of pad_routing_map.
    Args:
        routing_map (torch.Tensor): A boolean or integer tensor of shape [num_tokens,
            num_experts] indicating which tokens are routed to which experts.
        pad_multiple (int): The multiple to pad each expert's token count to.

    Returns:
        torch.Tensor: The padded routing map of shape [num_tokens, num_experts].
    """
    num_tokens, num_experts = routing_map.shape
    if num_tokens == 0:
        return routing_map

    input_map = routing_map.transpose(0, 1).contiguous().int()  

    output_map = torch.empty_like(input_map)

    
    grid = (num_experts,)
    BLOCK_SIZE = triton.next_power_of_2(num_tokens)

    _pad_routing_map_kernel[grid](
        input_map, output_map, num_tokens, pad_multiple, BLOCK_SIZE=BLOCK_SIZE
    )

    return output_map.transpose(0, 1)  
