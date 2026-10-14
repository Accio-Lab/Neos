from typing import Optional, Tuple

import torch
import triton
import triton.language as tl



@triton.jit
def _act_quant_kernel(
    X_ptr,
    Y_ptr,
    S_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for activation quantization.

    Each block processes BLOCK_M rows and group_size columns.
    """
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    
    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size

    
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    
    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=1)  

    
    amax = tl.maximum(amax, 1e-4)

    
    if round_scale:
        
        
        
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv

    
    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)

    
    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    
    s_cols = pid_n
    s_ptrs = S_ptr + rows * (N // group_size) + s_cols
    s_mask = row_mask
    tl.store(s_ptrs, scale, mask=s_mask)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with Triton.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)

    
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    y_flat = y.view(-1, N)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    s_flat = s.view(-1, N // block_size)

    
    BLOCK_M = 32
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None

    _act_quant_kernel[grid](
        x_flat,
        y_flat,
        s_flat,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=0 if round_scale else 2,
    )

    return y, s


@triton.jit
def _get_valid_kv_indices_kernel(
    page_table_ptr,  
    kv_indptr_ptr,  
    kv_indices_ptr,  
    bs: tl.constexpr,
    topk: tl.constexpr,
):
    """
    Extract valid indices (non -1) from page_table into kv_indices.
    Each program handles one batch.
    """
    batch_id = tl.program_id(0)

    
    dst_start = tl.load(kv_indptr_ptr + batch_id)

    
    src_offset = batch_id * topk
    offsets = tl.arange(0, topk)
    indices = tl.load(page_table_ptr + src_offset + offsets)

    
    mask = indices != -1

    
    
    prefix_sum = tl.cumsum(mask.to(tl.int32), axis=0) - 1

    
    dst_positions = dst_start + prefix_sum
    tl.store(kv_indices_ptr + dst_positions, indices, mask=mask)


def get_valid_kv_indices(
    page_table_1: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    bs: int,
):
    """
    Extract valid indices from page_table_1 into kv_indices buffer.

    Args:
        page_table_1: [bs, topk] page table with -1 as invalid
        kv_indptr: [bs + 1] cumulative count of valid indices per batch
        kv_indices: [bs * topk] pre-allocated output buffer
        bs: batch size
    """
    topk = page_table_1.shape[1]
    grid = (bs,)
    _get_valid_kv_indices_kernel[grid](
        page_table_1,
        kv_indptr,
        kv_indices,
        bs,
        topk,
    )
