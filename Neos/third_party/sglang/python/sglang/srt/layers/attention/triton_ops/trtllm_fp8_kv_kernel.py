"""
Fused FP8 quantization + paged KV cache write kernel for TRTLLM MHA backend.

This kernel fuses the following operations:
1. FP8 quantization of K and V tensors (from BF16/FP16 to FP8)
2. Per-token or per-page scale computation
3. Writing quantized K/V to paged KV cache layout

Performance benefits:
- Eliminates intermediate FP8 tensors in memory
- Reduces kernel launch overhead
- Better memory bandwidth utilization
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _process_kv_tensor(
    token_id,
    head_block_id,
    page_id,
    page_offset,
    input_ptr,
    cache_ptr,
    inv_scale,
    use_provided_scale: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    input_stride_token: tl.constexpr,
    input_stride_head: tl.constexpr,
    input_stride_dim: tl.constexpr,
    cache_stride_page: tl.constexpr,
    cache_stride_offset: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Process a block of heads for a single K or V tensor."""
    head_idx = head_block_id * BLOCK_HEAD
    num_heads_in_block = min(BLOCK_HEAD, num_kv_heads - head_idx)

    for dim_idx in range(0, head_dim, BLOCK_DIM):
        num_dims_in_block = min(BLOCK_DIM, head_dim - dim_idx)

        head_offsets = head_idx + tl.arange(0, BLOCK_HEAD)
        dim_offsets = dim_idx + tl.arange(0, BLOCK_DIM)

        head_mask = head_offsets < (head_idx + num_heads_in_block)
        dim_mask = dim_offsets < (dim_idx + num_dims_in_block)

        
        input_offsets = (
            token_id * input_stride_token
            + head_offsets[:, None] * input_stride_head
            + dim_offsets[None, :] * input_stride_dim
        )
        mask = head_mask[:, None] & dim_mask[None, :]

        block = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

        
        if use_provided_scale:
            block_fp8 = (block * inv_scale).to(tl.float8e4nv)
        else:
            block_fp8 = block.to(tl.float8e4nv)

        
        cache_offsets = (
            page_id * cache_stride_page
            + page_offset * cache_stride_offset
            + head_offsets[:, None] * cache_stride_head
            + dim_offsets[None, :] * cache_stride_dim
        )

        tl.store(cache_ptr + cache_offsets, block_fp8, mask=mask)


@triton.jit
def _fused_fp8_set_kv_buffer_kernel(
    
    k_ptr,  
    v_ptr,  
    
    k_cache_ptr,  
    v_cache_ptr,  
    
    cache_loc_ptr,  
    
    inv_k_scale_ptr,  
    inv_v_scale_ptr,  
    use_provided_scale: tl.constexpr,  
    
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    
    k_stride_token: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_dim: tl.constexpr,
    
    k_cache_stride_page: tl.constexpr,
    k_cache_stride_offset: tl.constexpr,
    k_cache_stride_head: tl.constexpr,
    k_cache_stride_dim: tl.constexpr,
    
    v_stride_token: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_dim: tl.constexpr,
    
    v_cache_stride_page: tl.constexpr,
    v_cache_stride_offset: tl.constexpr,
    v_cache_stride_head: tl.constexpr,
    v_cache_stride_dim: tl.constexpr,
    
    BLOCK_HEAD: tl.constexpr,  
    BLOCK_DIM: tl.constexpr,  
):
    """
    Fused FP8 quantization + paged KV cache write kernel.

    Each program processes one token-head_block-kv combination, quantizing and writing
    to the appropriate page in the KV cache.

    Grid: (num_tokens, num_head_blocks, 2) where dim2: 0=K, 1=V
    """
    
    token_id = tl.program_id(0)
    head_block_id = tl.program_id(1)
    kv_idx = tl.program_id(2)  

    
    cache_loc = tl.load(cache_loc_ptr + token_id)

    
    page_id = cache_loc // page_size
    page_offset = cache_loc % page_size

    
    if kv_idx == 0:
        
        if use_provided_scale:
            inv_scale = tl.load(inv_k_scale_ptr)
        else:
            inv_scale = 1.0
        _process_kv_tensor(
            token_id,
            head_block_id,
            page_id,
            page_offset,
            k_ptr,
            k_cache_ptr,
            inv_scale,
            use_provided_scale,
            num_kv_heads,
            head_dim,
            k_stride_token,
            k_stride_head,
            k_stride_dim,
            k_cache_stride_page,
            k_cache_stride_offset,
            k_cache_stride_head,
            k_cache_stride_dim,
            BLOCK_HEAD,
            BLOCK_DIM,
        )
    else:
        
        if use_provided_scale:
            inv_scale = tl.load(inv_v_scale_ptr)
        else:
            inv_scale = 1.0
        _process_kv_tensor(
            token_id,
            head_block_id,
            page_id,
            page_offset,
            v_ptr,
            v_cache_ptr,
            inv_scale,
            use_provided_scale,
            num_kv_heads,
            head_dim,
            v_stride_token,
            v_stride_head,
            v_stride_dim,
            v_cache_stride_page,
            v_cache_stride_offset,
            v_cache_stride_head,
            v_cache_stride_dim,
            BLOCK_HEAD,
            BLOCK_DIM,
        )


def fused_fp8_set_kv_buffer(
    k: torch.Tensor,  
    v: torch.Tensor,  
    k_cache: torch.Tensor,  
    v_cache: torch.Tensor,  
    cache_loc: torch.Tensor,  
    k_scale: Optional[
        float
    ] = None,  
    v_scale: Optional[float] = None,
    page_size: int = 16,
    use_triton: bool = True,  
) -> None:
    """
    Python wrapper for the fused FP8 quantization + paged KV cache write kernel.

    This function replicates the exact behavior of the original set_kv_buffer but with
    a fused kernel that combines FP8 quantization and cache write.

    Args:
        k: Key tensor after RoPE, can be 2D or 3D
        v: Value tensor, can be 2D or 3D
        k_cache: Paged K cache buffer in FP8
        v_cache: Paged V cache buffer in FP8
        cache_loc: Cache location for each token, shape [num_tokens]
        k_scale: Optional scalar scale for K (matching original set_kv_buffer)
        v_scale: Optional scalar scale for V (matching original set_kv_buffer)
        page_size: Number of tokens per page
        use_triton: Whether to use optimized Triton kernel
    """
    num_tokens = k.shape[0]

    
    if k_cache.ndim == 3:
        
        total_slots, num_kv_heads, head_dim = k_cache.shape
        assert (
            total_slots % page_size == 0
        ), f"total_slots ({total_slots}) must be divisible by page_size ({page_size})"
        num_pages = total_slots // page_size
    elif k_cache.ndim == 4:
        
        num_pages, ps, num_kv_heads, head_dim = k_cache.shape
        assert (
            ps == page_size
        ), f"page_size mismatch: cache has {ps}, expected {page_size}"
        total_slots = num_pages * page_size
    else:
        raise ValueError(f"Unsupported k_cache.ndim={k_cache.ndim}, expected 3 or 4")

    
    
    k_3d = None
    v_3d = None

    if k.ndim == 3:
        
        assert (
            k.shape[1] == num_kv_heads
        ), f"num_kv_heads mismatch: k.shape[1]={k.shape[1]} vs cache={num_kv_heads}"
        assert (
            k.shape[2] == head_dim
        ), f"head_dim mismatch: k.shape[2]={k.shape[2]} vs cache={head_dim}"
        assert v.shape[1] == num_kv_heads and v.shape[2] == head_dim, "v shape mismatch"

        
        k_3d = k
        v_3d = v
        
        k_2d = k.reshape(num_tokens, num_kv_heads * head_dim)
        v_2d = v.reshape(num_tokens, num_kv_heads * head_dim)
    elif k.ndim == 2:
        
        assert (
            k.shape[1] == num_kv_heads * head_dim
        ), f"k.shape[1]={k.shape[1]} != {num_kv_heads * head_dim}"
        assert (
            v.shape[1] == num_kv_heads * head_dim
        ), f"v.shape[1]={v.shape[1]} != {num_kv_heads * head_dim}"

        
        k_3d = k.view(num_tokens, num_kv_heads, head_dim)
        v_3d = v.view(num_tokens, num_kv_heads, head_dim)
        
        k_2d = k
        v_2d = v
    else:
        raise ValueError(f"Unsupported k.ndim={k.ndim}, expected 2 or 3")

    
    if k_cache.ndim == 3:
        
        stride_slot = k_cache.stride(0)
        stride_head = k_cache.stride(1)
        stride_dim = k_cache.stride(2)

        k_cache_stride_page = stride_slot * page_size
        k_cache_stride_offset = stride_slot
        k_cache_stride_head = stride_head
        k_cache_stride_dim = stride_dim

        v_stride_slot = v_cache.stride(0)
        v_stride_head = v_cache.stride(1)
        v_stride_dim = v_cache.stride(2)

        v_cache_stride_page = v_stride_slot * page_size
        v_cache_stride_offset = v_stride_slot
        v_cache_stride_head = v_stride_head
        v_cache_stride_dim = v_stride_dim
    else:
        
        k_cache_stride_page = k_cache.stride(0)
        k_cache_stride_offset = k_cache.stride(1)
        k_cache_stride_head = k_cache.stride(2)
        k_cache_stride_dim = k_cache.stride(3)

        v_cache_stride_page = v_cache.stride(0)
        v_cache_stride_offset = v_cache.stride(1)
        v_cache_stride_head = v_cache.stride(2)
        v_cache_stride_dim = v_cache.stride(3)

    
    use_provided_scale = k_scale is not None and v_scale is not None

    if use_triton and num_tokens > 0:
        
        
        k_stride_token = k_3d.stride(0)
        k_stride_head = k_3d.stride(1)
        k_stride_dim = k_3d.stride(2)

        v_stride_token = v_3d.stride(0)
        v_stride_head = v_3d.stride(1)
        v_stride_dim = v_3d.stride(2)

        
        BLOCK_HEAD = min(num_kv_heads, 8)  
        BLOCK_DIM = min(head_dim, 128)  

        
        num_head_blocks = (num_kv_heads + BLOCK_HEAD - 1) // BLOCK_HEAD

        
        
        
        
        grid = (num_tokens, num_head_blocks, 2)

        device = k_3d.device

        def _to_tensor_scale(scale):
            """Convert scale to 0-D CUDA tensor (accepts Python float or Tensor)."""
            if isinstance(scale, torch.Tensor):
                return scale.to(device=device, dtype=torch.float32)
            else:
                
                return torch.tensor(float(scale), device=device, dtype=torch.float32)

        
        
        
        if use_provided_scale:
            k_scale_tensor = _to_tensor_scale(k_scale)
            v_scale_tensor = _to_tensor_scale(v_scale)

            
            inv_k_scale = (1.0 / k_scale_tensor).to(device=device, dtype=torch.float32)
            inv_v_scale = (1.0 / v_scale_tensor).to(device=device, dtype=torch.float32)

            inv_k_scale_ptr = inv_k_scale
            inv_v_scale_ptr = inv_v_scale
        else:
            
            
            
            
            inv_k_scale_ptr = k_3d
            inv_v_scale_ptr = k_3d

        
        _fused_fp8_set_kv_buffer_kernel[grid](
            k_3d,
            v_3d,
            k_cache,
            v_cache,
            cache_loc,
            inv_k_scale_ptr,
            inv_v_scale_ptr,
            use_provided_scale,
            num_kv_heads,
            head_dim,
            page_size,
            k_stride_token,
            k_stride_head,
            k_stride_dim,
            k_cache_stride_page,
            k_cache_stride_offset,
            k_cache_stride_head,
            k_cache_stride_dim,
            v_stride_token,
            v_stride_head,
            v_stride_dim,
            v_cache_stride_page,
            v_cache_stride_offset,
            v_cache_stride_head,
            v_cache_stride_dim,
            BLOCK_HEAD=BLOCK_HEAD,
            BLOCK_DIM=BLOCK_DIM,
        )
    else:
        
        _naive_fp8_set_kv_buffer(
            k_2d, v_2d, k_cache, v_cache, cache_loc, k_scale, v_scale, page_size
        )


def _naive_fp8_set_kv_buffer(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_loc: torch.Tensor,
    k_scale: Optional[float],
    v_scale: Optional[float],
    page_size: int,
) -> None:
    """
    Naive fallback implementation that mimics the original set_kv_buffer logic.

    This directly replicates the behavior of MHATokenToKVPool.set_kv_buffer:
    1. Apply scale (if k.dtype != cache.dtype and scale is provided)
    2. Convert to FP8
    3. Write to cache at cache_loc

    Args:
        k: [num_tokens, num_kv_heads * head_dim], already reshaped to 2D
        v: [num_tokens, num_kv_heads * head_dim], already reshaped to 2D
        k_cache: [total_slots, num_kv_heads, head_dim] or [num_pages, page_size, num_kv_heads, head_dim]
        v_cache: Same shape as k_cache
        cache_loc: [num_tokens]
        k_scale: Optional scale for K
        v_scale: Optional scale for V
        page_size: Tokens per page
    """
    num_tokens = k.shape[0]

    
    if k_cache.ndim == 3:
        num_kv_heads = k_cache.shape[1]
        head_dim = k_cache.shape[2]
    elif k_cache.ndim == 4:
        num_kv_heads = k_cache.shape[2]
        head_dim = k_cache.shape[3]
    else:
        raise ValueError(f"Unsupported k_cache.ndim={k_cache.ndim}")

    
    
    store_dtype = k_cache.dtype
    if store_dtype == torch.uint8:
        
        dtype = torch.float8_e4m3fn  
    else:
        dtype = store_dtype  

    
    
    if k.dtype != dtype:
        
        k = k.clone()
        v = v.clone()

        if k_scale is not None:
            k.div_(k_scale)  
        if v_scale is not None:
            v.div_(v_scale)  

        k = k.to(dtype)
        v = v.to(dtype)

    
    if store_dtype == torch.uint8 and dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        k = k.view(torch.uint8)
        v = v.view(torch.uint8)

    
    k = k.view(num_tokens, num_kv_heads, head_dim)
    v = v.view(num_tokens, num_kv_heads, head_dim)

    
    if k_cache.ndim == 3:
        
        k_cache[cache_loc] = k
        v_cache[cache_loc] = v
    else:
        
        
        page_ids = cache_loc // page_size
        page_offsets = cache_loc % page_size
        k_cache[page_ids, page_offsets] = k
        v_cache[page_ids, page_offsets] = v
