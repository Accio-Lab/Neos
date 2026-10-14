from typing import Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import torch
from einops import rearrange

from sglang.jit_kernel.diffusion.cutedsl.common.reduce import (
    cta_reduce_sum,
    warp_reduce_sum,
)


@cute.jit
def apply_norm_cta(
    norm_type: cutlass.Constexpr,
    num_warps: cutlass.Constexpr,
    tidx: cutlass.Int32,
    tXrX: cute.Tensor,
    tWrW: Optional[cute.Tensor],
    tBrB: Optional[cute.Tensor],
    D: Union[cutlass.Int32, cutlass.Constexpr],
    eps: Union[cutlass.Float32, cutlass.Constexpr],
) -> cute.Tensor:
    if cutlass.const_expr(norm_type == "rms"):
        return apply_rmsnorm_cta(num_warps, tidx, tXrX, tWrW, D, eps)
    else:
        return apply_layernorm_cta(num_warps, tidx, tXrX, tWrW, tBrB, D, eps)


@cute.jit
def apply_rmsnorm_cta(
    num_warps: Union[cutlass.Int32, cutlass.Constexpr],
    tidx: cutlass.Int32,
    tXrX: cute.Tensor,
    tWrW: Optional[cute.Tensor],
    D: Union[cutlass.Int32, cutlass.Constexpr],
    eps: Union[cutlass.Float32, cutlass.Constexpr],
) -> cute.Tensor:
    """
    RMSNorm:
      y[i] = x[i] / sqrt(sum(x ^ 2) / D + eps) * w[i]
    """
    val = cute.Float32(0.0)
    for idx in range(cute.size(tXrX)):
        
        x_fp32 = tXrX[idx].to(cutlass.Float32)
        val += x_fp32 * x_fp32
    val = warp_reduce_sum(val)
    acc_sq = cta_reduce_sum(val, num_warps, tidx)
    factor = cute.rsqrt(acc_sq / D + eps)
    tNrN = cute.make_fragment_like(tXrX)
    if cutlass.const_expr(isinstance(tWrW, cute.Tensor)):
        tNrN.store((tXrX.load() * factor * tWrW.load()).to(tNrN.element_type))
    else:
        tNrN.store((tXrX.load() * factor).to(tNrN.element_type))
    return tNrN


@cute.jit
def apply_layernorm_cta(
    num_warps: Union[cutlass.Int32, cutlass.Constexpr],
    tidx: cutlass.Int32,
    tXrX: cute.Tensor,
    tWrW: Optional[cute.Tensor],
    tBrB: Optional[cute.Tensor],
    D: Union[cutlass.Int32, cutlass.Constexpr],
    eps: Union[cutlass.Float32, cutlass.Constexpr],
) -> cute.Tensor:
    """
    LayerNorm:
        mean = sum(x) / D
        var  = sum((x - mean) ^ 2) / D
        y[i] = (x[i] - mean) / sqrt(var + eps) * w[i] + b[i]
    """
    
    val = cute.Float32(0.0)
    for idx in range(cute.size(tXrX)):
        
        val += tXrX[idx].to(cutlass.Float32)
    val = warp_reduce_sum(val)
    val = cta_reduce_sum(val, num_warps, tidx)
    mean = val / D
    
    val = cute.Float32(0.0)
    for idx in range(cute.size(tXrX)):
        
        x_fp32 = tXrX[idx].to(cutlass.Float32)
        val += (x_fp32 - mean) * (x_fp32 - mean)
    val = warp_reduce_sum(val)
    val = cta_reduce_sum(val, num_warps, tidx)
    factor = cute.rsqrt(val / D + eps)
    
    tNrN = cute.make_fragment_like(tXrX)
    if cutlass.const_expr(
        isinstance(tWrW, cute.Tensor) and isinstance(tBrB, cute.Tensor)
    ):
        tNrN.store(
            ((tXrX.load() - mean) * factor * tWrW.load() + tBrB.load()).to(
                tNrN.element_type
            )
        )
    else:
        tNrN.store(((tXrX.load() - mean) * factor).to(tNrN.element_type))
    return tNrN
























def broadcast_tensor_for_bsfd(
    tensor: Union[Optional[torch.Tensor], int],
    B: int,
    S: int,
    D: int,
) -> Union[Optional[torch.Tensor], int]:
    """
    Broadcast to (B, S, D) without memory copy for following shapes:
    - [D], [1, D], [1, 1, D], [B, D], [B, 1, D], [B, S, D].
    """

    
    if not isinstance(tensor, torch.Tensor):
        return tensor

    if tensor.ndim == 1:
        
        if tensor.numel() == 1:
            return tensor
        return rearrange(tensor, "d -> 1 1 d").expand(B, S, D)
    if tensor.ndim == 2:
        return rearrange(tensor, "b d -> b 1 d").expand(B, S, D)
    if tensor.ndim == 3:
        return tensor.expand(B, S, D)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"BSFD broadcast: unsupported tensor ndim: {tensor.ndim}.")


@cute.jit
def tensor_slice_for_bsfd(
    mV: cute.Tensor,
    thr_copy: cute.ThrCopy,
    batch_id: cutlass.Int32,
    seq_id: cutlass.Int32,
    S: Union[cutlass.Int32, cutlass.Constexpr],
    D: Union[cutlass.Int32, cutlass.Constexpr],
) -> Tuple[cute.Tensor, cute.Tensor]:
    """
    Slice a BSFD-compatible tensor into a per-thread gmem tile and rmem fragment.

    Given a logical (batch_id, seq_id), this helper selects the corresponding
    D-length slice from `mV` and prepares it for vectorized copy.
    """
    gV: cute.Tensor
    if cutlass.const_expr(cute.is_static(mV.layout) and cute.size(mV.layout) == 1):
        
        
        layout = cute.make_layout(shape=((1, 1), (1,)))
        tVgV = cute.make_tensor(mV.iterator, layout)
        tVrV = cute.make_rmem_tensor(layout, mV.element_type)
        return tVgV, tVrV

    
    
    if cutlass.const_expr(len(mV.shape) == 1):
        gV = mV
    elif cutlass.const_expr(len(mV.shape) == 3):
        gV = cute.local_tile(mV, tiler=(1, 1, D), coord=(batch_id, seq_id, 0))
        gV = gV[0, 0, None]
    elif cutlass.const_expr(len(mV.shape) == 4):
        
        
        frame_len = S // mV.shape[1]
        frame_id = seq_id // frame_len
        gV = cute.local_tile(mV, tiler=(1, 1, 1, D), coord=(batch_id, frame_id, 0, 0))
        gV = gV[0, 0, 0, None]
    else:
        raise NotImplementedError(f"BSFD slice: unsupported shape {mV.shape}.")
    tVgV = thr_copy.partition_S(gV)
    tVrV = cute.make_fragment_like(tVgV, tVgV.element_type)
    return tVgV, tVrV
