# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Standalone MXFP8 quantization kernel with fused scale swizzle.

One block per token. Quantizes BF16 → FP8 e4m3 and writes scales directly
in cuBLAS 2D blocked (swizzled) layout. No FP4, no triton_kernels dependency.

Usage:
    from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize
    data, swizzled_scales, total_scale_bytes = mxfp8_quantize(x_bf16)
    # data: [M, K] float8_e4m3fn
    # swizzled_scales: 1D uint8 in cuBLAS blocked layout
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def _mxfp8_quant_swizzle_kernel(
    out_ptr,  
    scale_ptr,  
    src_ptr,  
    K,  
    n_col_blocks,  
    REAL_GROUPS: tl.constexpr,  
    BLOCK_K: tl.constexpr,  
    BLOCK_GROUPS: tl.constexpr,  
):
    """Each triton block quantizes one row → FP8 e4m3, write scales directly in swizzled layout.

    We use round up in scale calculation. see: Mishra et al.,
    Recipes for Pre-training LLMs with MXFP8 (https://arxiv.org/pdf/2506.08027)

    The implementation borrows code from the triton upstream MXFP downcast kernel:
    https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_downcast_to_mxfp.py

    Note on swizzled scale layout (torch.nn.functional.SwizzleType.SWIZZLE_32_4_4):

        Background: In MXFP8, every group of 32 elements shares one 1-byte scale
        (an e8m0 exponent). For an [M, K] matrix, this gives an [M, K//32] scale
        matrix. cuBLAS doesn't read these scales in simple row-major order — it
        expects a "swizzled" layout optimized for its internal access patterns.

        Step 1 — Divide into macro-tiles:
            The scale matrix is partitioned into 128-row x 4-col macro-tiles.
            Each tile is stored as a contiguous 512-byte (128 x 4) block.

        Step 2 — Interleave within each tile:
            Within a macro-tile, the 128 rows are NOT stored sequentially.
            Instead, they are split into 4 groups of 32 rows:
                group 0: rows   0- 31
                group 1: rows  32- 63
                group 2: rows  64- 95
                group 3: rows  96-127

            Rows with the same position within their group (same "sub_row")
            are placed next to each other. So the memory layout is:

            Concretely, for sub_row=0:
                byte 0:  row  0, col 0
                byte 1:  row  0, col 1
                byte 2:  row  0, col 2
                byte 3:  row  0, col 3
                byte 4:  row 32, col 0
                byte 5:  row 32, col 1
                byte 6:  row 32, col 2
                byte 7:  row 32, col 3
                byte 8:  row 64, col 0
                ...
                byte 15: row 96, col 3

        The formula to map logical (row, col) → byte offset:
            tile_idx = (row // 128) * n_col_blocks + (col // 4)
            sub_row  = row % 32
            group    = (row % 128) // 32
            local_col = col % 4
            offset   = tile_idx * 512 + sub_row * 16 + group * 4 + local_col

    """
    row = tl.program_id(0)
    src_row = src_ptr + row * K
    out_row = out_ptr + row * K

    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    
    x = tl.load(src_row + offs, mask=mask, other=0.0).to(tl.float32)

    
    x_grouped = tl.reshape(x, [BLOCK_GROUPS, 32])
    abs_grouped = tl.abs(x_grouped)
    max_vals = tl.max(abs_grouped, axis=1)

    
    
    dequant_scale = max_vals / 448.0
    
    
    
    
    dequant_exp = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    
    dequant_rounded = dequant_exp.to(tl.float32, bitcast=True)
    
    quant_scale = tl.where(dequant_rounded == 0, 0.0, 1.0 / dequant_rounded)

    
    quantized = x_grouped * quant_scale[:, None]
    quantized_flat = tl.reshape(quantized, [BLOCK_K])
    out_fp8 = quantized_flat.to(tl.float8e4nv)

    
    tl.store(out_row + offs, out_fp8, mask=mask)

    
    scale_exp = (dequant_exp >> 23).to(tl.uint8)
    col_offs = tl.arange(0, BLOCK_GROUPS)
    col_mask = col_offs < REAL_GROUPS

    
    
    
    
    
    
    
    
    macro_row_block = row // 128
    macro_col_block = col_offs // 4
    local_row = row % 128
    local_col = col_offs % 4
    group = local_row // 32
    sub_row = local_row % 32
    tile_idx = macro_row_block * n_col_blocks + macro_col_block
    swizzled_offs = tile_idx * 512 + sub_row * 16 + group * 4 + local_col

    tl.store(scale_ptr + swizzled_offs, scale_exp, mask=col_mask)


def mxfp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to MXFP8 with fused scale swizzle.

    Args:
        x: [M, K] tensor in bf16/fp16/fp32. K must be divisible by 32.

    Returns:
        (data, swizzled_scales):
            data: [M, K] float8_e4m3fn
            swizzled_scales: 1D tensor in cuBLAS blocked layout (uint8/e8m0)
    """
    assert x.is_cuda and x.dim() == 2
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    M, K = x.shape
    assert K % 32 == 0, f"K ({K}) must be divisible by 32"

    scale_cols = K // 32
    n_row_blocks = _ceil_div(M, 128)
    n_col_blocks = _ceil_div(scale_cols, 4)
    total_scale_bytes = n_row_blocks * n_col_blocks * 512

    out_data = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    out_scale = torch.zeros(total_scale_bytes, dtype=torch.uint8, device=x.device)

    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_GROUPS = BLOCK_K // 32

    _mxfp8_quant_swizzle_kernel[(M,)](
        out_data,
        out_scale,
        x,
        K,
        n_col_blocks,
        REAL_GROUPS=scale_cols,
        BLOCK_K=BLOCK_K,
        BLOCK_GROUPS=BLOCK_GROUPS,
    )

    return out_data, out_scale.view(torch.float8_e8m0fnu)
