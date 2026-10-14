# SPDX-License-Identifier: Apache-2.0
# Adapt from https://github.com/KwaiVGI/VMoBA/blob/main/src/vmoba.py

import random
import time
from typing import Tuple

import torch

try:
    from flash_attn import (  
        flash_attn_varlen_func,
    )
    from flash_attn.flash_attn_interface import (
        _flash_attn_varlen_backward,
        _flash_attn_varlen_forward,
    )
except ImportError:

    def _unsupported(*args, **kwargs):
        raise ImportError(
            "flash-attn is not installed. Please install it, e.g., `pip install flash-attn`."
        )

    _flash_attn_varlen_forward = _unsupported
    _flash_attn_varlen_backward = _unsupported
    flash_attn_varlen_func = _unsupported

from functools import lru_cache

from einops import rearrange


@lru_cache(maxsize=16)
def calc_chunks(cu_seqlen, moba_chunk_size):
    """
    Calculate chunk boundaries.

    For vision tasks we include all chunks (even the last one which might be shorter)
    so that every chunk can be selected.
    """
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    batch_num_chunk = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size
    cu_num_chunk = torch.ones(
        batch_num_chunk.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_chunk.dtype,
    )
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0)
    num_chunk = cu_num_chunk[-1]
    chunk_sizes = torch.full(
        (num_chunk + 1,), moba_chunk_size, dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_sizes[0] = 0
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * moba_chunk_size
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size
    cu_chunk = chunk_sizes.cumsum(dim=-1, dtype=torch.int32)
    chunk_to_batch = torch.zeros(
        (num_chunk,), dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_to_batch[cu_num_chunk[1:-1]] = 1
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=torch.int32)

    
    filtered_chunk_indices = torch.arange(
        num_chunk, device=cu_seqlen.device, dtype=torch.int32
    )
    num_filtered_chunk = num_chunk

    return cu_chunk, filtered_chunk_indices, num_filtered_chunk, chunk_to_batch





def _select_threshold_query_head(
    gate: torch.Tensor,
    valid_gate_mask: torch.Tensor,
    gate_self_chunk_mask: torch.Tensor,
    simsum_threshold: float,
) -> torch.Tensor:
    """
    Selects chunks for each <query, head> pair based on threshold.
    Normalization and sorting happen along the chunk dimension (dim=0).
    """
    C, H, S = gate.shape
    eps = 1e-6

    
    gate_masked = torch.where(valid_gate_mask, gate, -torch.inf)  
    gate_min_val = torch.where(valid_gate_mask, gate, torch.inf)  

    row_min = gate_min_val.amin(dim=0)  
    row_max = gate_masked.amax(dim=0)  
    denom = row_max - row_min
    denom = torch.where(
        denom <= eps, torch.ones_like(denom), denom
    )  

    gate_norm = (gate - row_min.unsqueeze(0)) / denom.unsqueeze(0)
    gate_norm = torch.where(valid_gate_mask, gate_norm, 0.0)  

    
    self_norm = (gate_norm * gate_self_chunk_mask).sum(dim=0)  

    
    total_norm_sum = gate_norm.sum(dim=0)  
    remain_ratio = simsum_threshold - self_norm / (total_norm_sum + eps)  
    remain_ratio = torch.clamp(
        remain_ratio, min=0.0
    )  

    
    others_norm = gate_norm.clone()
    others_norm[gate_self_chunk_mask] = 0.0

    
    sorted_norm, sorted_idx = torch.sort(
        others_norm, descending=True, dim=0
    )  

    
    cumsum_others = sorted_norm.cumsum(dim=0)  

    
    ratio = cumsum_others / (total_norm_sum.unsqueeze(0) + eps)  
    cond = ratio >= remain_ratio.unsqueeze(0)  
    any_cond = cond.any(dim=0)  
    
    cutoff = torch.where(
        any_cond,
        cond.float().argmax(dim=0),
        torch.full_like(any_cond, fill_value=C - 1),
    )  

    
    idx_range = torch.arange(C, device=gate.device).view(-1, 1, 1)  
    sorted_mask = idx_range <= cutoff.unsqueeze(0)  

    
    others_mask = torch.zeros_like(gate, dtype=torch.bool)
    others_mask.scatter_(0, sorted_idx, sorted_mask)

    
    final_gate_mask = valid_gate_mask & (others_mask | gate_self_chunk_mask)

    return final_gate_mask


def _select_threshold_block(
    gate: torch.Tensor,
    valid_gate_mask: torch.Tensor,
    gate_self_chunk_mask: torch.Tensor,
    simsum_threshold: float,
) -> torch.Tensor:
    """
    Selects <query, head> pairs for each block based on threshold.
    Normalization and sorting happen across the head and sequence dimensions (dim=1, 2).
    """
    C, H, S = gate.shape
    HS = H * S
    eps = 1e-6

    
    gate_masked = torch.where(valid_gate_mask, gate, -torch.inf)  
    gate_min_val = torch.where(valid_gate_mask, gate, torch.inf)  

    block_max = gate_masked.amax(dim=(1, 2), keepdim=True)  
    block_min = gate_min_val.amin(dim=(1, 2), keepdim=True)  
    block_denom = block_max - block_min
    block_denom = torch.where(
        block_denom <= eps, torch.ones_like(block_denom), block_denom
    )  

    gate_norm = (gate - block_min) / block_denom  
    gate_norm = torch.where(valid_gate_mask, gate_norm, 0.0)  

    
    self_norm_entries = gate_norm * gate_self_chunk_mask  
    
    self_norm_sum_per_block = self_norm_entries.sum(dim=(1, 2))  

    
    total_norm_sum_per_block = gate_norm.sum(dim=(1, 2))  
    remain_ratio = simsum_threshold - self_norm_sum_per_block / (
        total_norm_sum_per_block + eps
    )  
    remain_ratio = torch.clamp(remain_ratio, min=0.0)  

    
    others_norm = gate_norm.clone()
    others_norm[gate_self_chunk_mask] = 0.0  

    
    others_flat = others_norm.contiguous().view(C, HS)  
    sorted_others_flat, sorted_indices_flat = torch.sort(
        others_flat, dim=1, descending=True
    )  

    
    cumsum_others_flat = sorted_others_flat.cumsum(dim=1)  

    
    ratio_flat = cumsum_others_flat / (
        total_norm_sum_per_block.unsqueeze(1) + eps
    )  
    cond_flat = ratio_flat >= remain_ratio.unsqueeze(1)  
    any_cond = cond_flat.any(dim=1)  
    
    cutoff_flat = torch.where(
        any_cond,
        cond_flat.float().argmax(dim=1),
        torch.full_like(any_cond, fill_value=HS - 1),
    )  

    
    idx_range_flat = torch.arange(HS, device=gate.device).unsqueeze(0)  
    sorted_mask_flat = idx_range_flat <= cutoff_flat.unsqueeze(1)  

    
    others_mask_flat = torch.zeros_like(others_flat, dtype=torch.bool)  
    others_mask_flat.scatter_(1, sorted_indices_flat, sorted_mask_flat)
    others_mask = others_mask_flat.view(C, H, S)  

    
    final_gate_mask = valid_gate_mask & (others_mask | gate_self_chunk_mask)

    return final_gate_mask


def _select_threshold_overall(
    gate: torch.Tensor,
    valid_gate_mask: torch.Tensor,
    gate_self_chunk_mask: torch.Tensor,
    simsum_threshold: float,
) -> torch.Tensor:
    """
    Selects <chunk, query, head> triplets globally based on threshold.
    Normalization and sorting happen across all valid entries.
    """
    C, H, S = gate.shape
    CHS = C * H * S
    eps = 1e-6

    
    gate_masked = torch.where(valid_gate_mask, gate, -torch.inf)  
    gate_min_val = torch.where(valid_gate_mask, gate, torch.inf)  

    overall_max = gate_masked.max()  
    overall_min = gate_min_val.min()  
    overall_denom = overall_max - overall_min
    overall_denom = torch.where(
        overall_denom <= eps,
        torch.tensor(1.0, device=gate.device, dtype=gate.dtype),
        overall_denom,
    )

    gate_norm = (gate - overall_min) / overall_denom  
    gate_norm = torch.where(valid_gate_mask, gate_norm, 0.0)  

    
    self_norm_entries = gate_norm * gate_self_chunk_mask  
    
    self_norm_sum_overall = self_norm_entries.sum()  

    
    total_norm_sum_overall = gate_norm.sum()  
    remain_ratio = simsum_threshold - self_norm_sum_overall / (
        total_norm_sum_overall + eps
    )  
    remain_ratio = torch.clamp(remain_ratio, min=0.0)  

    
    others_norm = gate_norm.clone()
    others_norm[gate_self_chunk_mask] = 0.0  

    
    others_flat = others_norm.flatten()  
    valid_others_mask_flat = (
        valid_gate_mask.flatten() & ~gate_self_chunk_mask.flatten()
    )  

    
    valid_others_indices = torch.where(valid_others_mask_flat)[0]
    valid_others_values = others_flat[valid_others_indices]

    sorted_others_values, sort_perm = torch.sort(
        valid_others_values, descending=True
    )  
    sorted_original_indices = valid_others_indices[
        sort_perm
    ]  

    
    cumsum_others_values = sorted_others_values.cumsum(dim=0)  

    
    ratio_values = cumsum_others_values / (
        total_norm_sum_overall + eps
    )  
    cond_values = ratio_values >= remain_ratio  
    any_cond = cond_values.any()  

    
    cutoff_idx_in_sorted = torch.where(
        any_cond,
        cond_values.float().argmax(dim=0),
        torch.tensor(
            len(sorted_others_values) - 1, device=gate.device, dtype=torch.long
        ),
    )

    
    
    selected_other_indices = sorted_original_indices[: cutoff_idx_in_sorted + 1]

    
    others_mask_flat = torch.zeros_like(others_flat, dtype=torch.bool)  
    if selected_other_indices.numel() > 0:  
        others_mask_flat[selected_other_indices] = True
    others_mask = others_mask_flat.view(C, H, S)  

    
    final_gate_mask = valid_gate_mask & (others_mask | gate_self_chunk_mask)

    return final_gate_mask


def _select_threshold_head_global(
    gate: torch.Tensor,
    valid_gate_mask: torch.Tensor,
    gate_self_chunk_mask: torch.Tensor,
    simsum_threshold: float,
) -> torch.Tensor:
    """
    Selects <chunk, query> globally for each head based on threshold.
    """
    C, H, S = gate.shape
    eps = 1e-6

    
    gate_masked = torch.where(valid_gate_mask, gate, -torch.inf)
    gate_min_val = torch.where(valid_gate_mask, gate, torch.inf)

    max_per_head = gate_masked.amax(dim=(0, 2), keepdim=True)  
    min_per_head = gate_min_val.amin(dim=(0, 2), keepdim=True)  
    denom = max_per_head - min_per_head
    denom = torch.where(denom <= eps, torch.ones_like(denom), denom)

    gate_norm = (gate - min_per_head) / denom
    gate_norm = torch.where(valid_gate_mask, gate_norm, 0.0)  

    
    self_norm_sum = (gate_norm * gate_self_chunk_mask).sum(dim=(0, 2))  

    
    total_norm_sum = gate_norm.sum(dim=(0, 2))  

    
    remain_ratio = simsum_threshold - self_norm_sum / (total_norm_sum + eps)  
    remain_ratio = torch.clamp(remain_ratio, min=0.0)

    
    others_norm = gate_norm.clone()
    others_norm[gate_self_chunk_mask] = 0.0  

    
    CS = C * S
    others_flat = others_norm.permute(1, 0, 2).reshape(H, CS)  
    valid_flat = (
        (valid_gate_mask & ~gate_self_chunk_mask).permute(1, 0, 2).reshape(H, CS)
    )  

    
    masked_flat = torch.where(valid_flat, others_flat, torch.zeros_like(others_flat))
    sorted_vals, sorted_idx = torch.sort(
        masked_flat, dim=1, descending=True
    )  

    cumsum_vals = sorted_vals.cumsum(dim=1)  
    ratio_vals = cumsum_vals / (total_norm_sum.unsqueeze(1) + eps)  
    cond = ratio_vals >= remain_ratio.unsqueeze(1)  

    has_cutoff = cond.any(dim=1)  
    default = torch.full((H,), CS - 1, device=gate.device, dtype=torch.long)
    cutoff = torch.where(has_cutoff, cond.float().argmax(dim=1), default)  

    idx_range = torch.arange(CS, device=gate.device).unsqueeze(0)  
    sorted_mask = idx_range <= cutoff.unsqueeze(1)  

    selected_flat = torch.zeros_like(valid_flat)  
    selected_flat.scatter_(1, sorted_idx, sorted_mask)  

    
    others_mask = selected_flat.reshape(H, C, S).permute(1, 0, 2)  

    
    final_gate_mask = valid_gate_mask & (gate_self_chunk_mask | others_mask)

    return final_gate_mask


class MixedAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    ):
        ctx.max_seqlen = max_seqlen
        ctx.moba_chunk_size = moba_chunk_size
        ctx.softmax_scale = softmax_scale = q.shape[-1] ** (-0.5)

        
        
        self_attn_out_sh, self_attn_lse_hs, _, _ = _flash_attn_varlen_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self_attn_cu_seqlen,
            cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
        )
        
        moba_attn_out, moba_attn_lse_hs, _, _ = _flash_attn_varlen_forward(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
        )

        self_attn_lse_sh = self_attn_lse_hs.t().contiguous()
        moba_attn_lse = moba_attn_lse_hs.t().contiguous()

        output = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        output_2d = output.view(-1, q.shape[2])

        max_lse_1d = self_attn_lse_sh.view(-1)
        max_lse_1d = max_lse_1d.index_reduce(
            0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
        )
        self_attn_lse_sh = self_attn_lse_sh - max_lse_1d.view_as(self_attn_lse_sh)
        moba_attn_lse = (
            moba_attn_lse.view(-1)
            .sub(max_lse_1d.index_select(0, moba_q_sh_indices))
            .reshape_as(moba_attn_lse)
        )

        mixed_attn_se_sh = self_attn_lse_sh.exp()
        moba_attn_se = moba_attn_lse.exp()

        mixed_attn_se_sh.view(-1).index_add_(
            0, moba_q_sh_indices, moba_attn_se.view(-1)
        )
        mixed_attn_lse_sh = mixed_attn_se_sh.log()

        
        factor = (self_attn_lse_sh - mixed_attn_lse_sh).exp()  
        self_attn_out_sh = self_attn_out_sh * factor.unsqueeze(-1)
        output_2d += self_attn_out_sh.reshape_as(output_2d)

        
        mixed_attn_lse = (
            mixed_attn_lse_sh.view(-1)
            .index_select(0, moba_q_sh_indices)
            .view_as(moba_attn_lse)
        )
        factor = (moba_attn_lse - mixed_attn_lse).exp()  
        moba_attn_out = moba_attn_out * factor.unsqueeze(-1)
        raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
        output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out)
        output = output.to(q.dtype)
        mixed_attn_lse_sh = mixed_attn_lse_sh + max_lse_1d.view_as(mixed_attn_se_sh)
        ctx.save_for_backward(
            output,
            mixed_attn_lse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        )

        return output

    @staticmethod
    def backward(ctx, d_output):

        max_seqlen = ctx.max_seqlen
        moba_chunk_size = ctx.moba_chunk_size
        softmax_scale = ctx.softmax_scale

        (
            output,
            mixed_attn_vlse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        ) = ctx.saved_tensors

        d_output = d_output.contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _ = _flash_attn_varlen_backward(
            dout=d_output,
            q=q,
            k=k,
            v=v,
            out=output,
            softmax_lse=mixed_attn_vlse_sh.t().contiguous(),
            dq=dq,
            dk=dk,
            dv=dv,
            cu_seqlens_q=self_attn_cu_seqlen,
            cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
            window_size_left=-1,
            window_size_right=-1,
        )

        headdim = q.shape[-1]
        d_moba_output = (
            d_output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )
        moba_output = (
            output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )

        mixed_attn_vlse = (
            mixed_attn_vlse_sh.view(-1).index_select(0, moba_q_sh_indices).view(1, -1)
        )

        dmq = torch.empty_like(moba_q)
        dmkv = torch.empty_like(moba_kv)
        _ = _flash_attn_varlen_backward(
            dout=d_moba_output,
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            out=moba_output,
            softmax_lse=mixed_attn_vlse,
            dq=dmq,
            dk=dmkv[:, 0],
            dv=dmkv[:, 1],
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
            window_size_left=-1,
            window_size_right=-1,
        )

        return dq, dk, dv, None, dmq, dmkv, None, None, None, None, None


def moba_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
    select_mode: str = "threshold",  
    simsum_threshold: float = 0.25,
    threshold_type: str = "query_head",
) -> torch.Tensor:
    """
    Accelerated MOBA attention for vision tasks with proper LSE normalization.

    This version:
      - Splits KV into chunks.
      - For each query head, selects the top-k relevant KV chunks (including the self chunk)
        by amplifying the diagonal (self-chunk) logits.
      - Aggregates the attention outputs from the selected chunks using a log-sum-exp
        reduction so that attending to each query over the selected chunks is equivalent
        to the original algorithm.
    """
    
    kv = torch.stack((k, v), dim=1)
    seqlen, num_head, head_dim = q.shape

    
    cu_chunk, filtered_chunk_indices, num_filtered_chunk, chunk_to_batch = calc_chunks(
        cu_seqlens, moba_chunk_size
    )

    self_attn_cu_seqlen = cu_chunk

    
    moba_topk = min(moba_topk, num_filtered_chunk)

    
    chunk_starts = cu_chunk[filtered_chunk_indices]  
    chunk_ends = cu_chunk[filtered_chunk_indices + 1]  
    chunk_lengths = chunk_ends - chunk_starts  
    max_chunk_len = int(chunk_lengths.max().item())

    range_tensor = torch.arange(
        max_chunk_len, device=kv.device, dtype=chunk_starts.dtype
    ).unsqueeze(0)
    indices = chunk_starts.unsqueeze(1) + range_tensor
    indices = torch.clamp(indices, max=kv.shape[0] - 1)
    valid_mask = range_tensor < chunk_lengths.unsqueeze(1)
    gathered = kv[indices.view(-1)].view(
        num_filtered_chunk, max_chunk_len, *kv.shape[1:]
    )
    gathered = gathered * valid_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).type_as(
        gathered
    )

    
    key_values = gathered[
        :, :, 0
    ].float()  
    valid_mask_exp = valid_mask.unsqueeze(-1).unsqueeze(-1)
    key_sum = (key_values * valid_mask_exp).sum(dim=1)
    divisor = valid_mask.sum(dim=1).unsqueeze(-1).unsqueeze(-1)
    key_gate_weight = key_sum / divisor  

    
    q_float = q.float()
    
    gate = torch.bmm(
        key_gate_weight.permute(1, 0, 2), q_float.permute(1, 0, 2).transpose(1, 2)
    ).permute(1, 0, 2)

    
    gate_seq_idx = (
        torch.arange(seqlen, device=q.device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(num_filtered_chunk, seqlen)
    )
    chunk_start = cu_chunk[filtered_chunk_indices]  
    chunk_end = cu_chunk[filtered_chunk_indices + 1]  
    gate_self_chunk_mask = (
        (
            (gate_seq_idx >= chunk_start.unsqueeze(1))
            & (gate_seq_idx < chunk_end.unsqueeze(1))
        )
        .unsqueeze(1)
        .expand(-1, num_head, -1)
    )
    amplification_factor = 1e9  
    origin_gate = gate.clone()
    gate = gate.clone()
    if select_mode == "topk":
        gate[gate_self_chunk_mask] += amplification_factor

    
    batch_starts = cu_seqlens[chunk_to_batch[filtered_chunk_indices]]
    batch_ends = cu_seqlens[chunk_to_batch[filtered_chunk_indices] + 1]
    gate_batch_start_mask = gate_seq_idx < batch_starts.unsqueeze(1)
    gate_batch_end_mask = gate_seq_idx >= batch_ends.unsqueeze(1)
    gate_inf_mask = gate_batch_start_mask | gate_batch_end_mask
    gate.masked_fill_(gate_inf_mask.unsqueeze(1), -float("inf"))

    if select_mode == "topk":
        
        valid_gate_mask = gate != -float("inf")
        if threshold_type == "query_head":
            
            
            _, gate_topk_idx = torch.topk(
                gate, k=moba_topk, dim=0, largest=True, sorted=False
            )
            gate_idx_mask = torch.zeros_like(gate, dtype=torch.bool)
            gate_idx_mask.scatter_(0, gate_topk_idx, True)
            gate_mask = valid_gate_mask & gate_idx_mask
        elif threshold_type == "overall":
            
            C, H, S = gate.shape
            flat_gate = gate.flatten()
            flat_mask = valid_gate_mask.flatten()
            flat_gate_masked = torch.where(flat_mask, flat_gate, -float("inf"))
            
            vals, idx = torch.topk(
                flat_gate_masked, k=moba_topk * H * S, largest=True, sorted=False
            )
            others_mask_flat = torch.zeros_like(flat_mask, dtype=torch.bool)
            others_mask_flat[idx] = True
            gate_mask = (valid_gate_mask.flatten() & others_mask_flat).view(gate.shape)
        elif threshold_type == "head_global":
            
            C, H, S = gate.shape
            CS = C * S
            flat_gate = gate.permute(1, 0, 2).reshape(H, CS)
            flat_valid = valid_gate_mask.permute(1, 0, 2).reshape(H, CS)
            flat_gate_masked = torch.where(
                flat_valid, flat_gate, torch.full_like(flat_gate, -float("inf"))
            )
            
            _, topk_idx = torch.topk(
                flat_gate_masked, k=moba_topk * S, dim=1, largest=True, sorted=False
            )
            gate_idx_flat = torch.zeros_like(flat_valid, dtype=torch.bool)
            gate_idx_flat.scatter_(1, topk_idx, True)
            gate_mask = gate_idx_flat.reshape(H, C, S).permute(1, 0, 2)
        else:
            raise ValueError(
                f"Invalid threshold_type for topk: {threshold_type}. "
                "Choose 'query_head', 'block', or 'overall'."
            )
    elif select_mode == "threshold":
        
        valid_gate_mask = gate != -float("inf")  
        if threshold_type == "query_head":
            gate_mask = _select_threshold_query_head(
                gate, valid_gate_mask, gate_self_chunk_mask, simsum_threshold
            )
        elif threshold_type == "block":
            gate_mask = _select_threshold_block(
                gate, valid_gate_mask, gate_self_chunk_mask, simsum_threshold
            )
        elif threshold_type == "overall":
            gate_mask = _select_threshold_overall(
                gate, valid_gate_mask, gate_self_chunk_mask, simsum_threshold
            )
        elif threshold_type == "head_global":
            gate_mask = _select_threshold_head_global(
                gate, valid_gate_mask, gate_self_chunk_mask, simsum_threshold
            )
        else:
            raise ValueError(
                f"Invalid threshold_type: {threshold_type}. Choose 'query_head', 'block', or 'overall'."
            )
    else:
        raise ValueError(
            f"Invalid select_mode: {select_mode}. Choose 'topk' or 'threshold'."
        )

    
    gate_mask = gate_mask & ~gate_self_chunk_mask
    
    if gate_mask.sum() == 0:
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=False
        )

    
    
    moba_q_indices = gate_mask.reshape(gate_mask.shape[0], -1).nonzero(as_tuple=True)[
        -1
    ]  
    moba_q_sh_indices = (moba_q_indices % seqlen) * num_head + (
        moba_q_indices // seqlen
    )
    moba_q = (
        rearrange(q, "s h d -> (h s) d").index_select(0, moba_q_indices).unsqueeze(1)
    )

    
    moba_seqlen_q = gate_mask.sum(dim=-1).flatten()
    q_zero_mask = moba_seqlen_q == 0
    valid_expert_mask = ~q_zero_mask
    if q_zero_mask.sum() > 0:
        moba_seqlen_q = moba_seqlen_q[valid_expert_mask]
    moba_cu_seqlen_q = torch.cat(
        (
            torch.tensor([0], device=q.device, dtype=moba_seqlen_q.dtype),
            moba_seqlen_q.cumsum(dim=0),
        ),
        dim=0,
    ).to(torch.int32)

    
    experts_tensor = rearrange(gathered, "nc cl two h d -> (nc h) cl two d")
    valid_expert_lengths = (
        chunk_lengths.unsqueeze(1)
        .expand(num_filtered_chunk, num_head)
        .reshape(-1)
        .to(torch.int32)
    )
    if q_zero_mask.sum() > 0:
        experts_tensor = experts_tensor[valid_expert_mask]
        valid_expert_lengths = valid_expert_lengths[valid_expert_mask]

    seq_range = torch.arange(
        experts_tensor.shape[1], device=experts_tensor.device
    ).unsqueeze(0)
    mask = seq_range < valid_expert_lengths.unsqueeze(1)
    moba_kv = experts_tensor[mask]  
    moba_kv = moba_kv.unsqueeze(2)  

    moba_cu_seqlen_kv = torch.cat(
        [
            torch.zeros(1, device=experts_tensor.device, dtype=torch.int32),
            valid_expert_lengths.cumsum(dim=0),
        ],
        dim=0,
    ).to(torch.int32)

    assert (
        moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape
    ), f"Mismatch between moba_cu_seqlen_kv.shape and moba_cu_seqlen_q.shape: {moba_cu_seqlen_kv.shape} vs {moba_cu_seqlen_q.shape}"

    return MixedAttention.apply(
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    )


def process_moba_input(
    x,
    patch_resolution,
    chunk_size,
):
    """
    Process inputs for the attention function.

    Args:
        x (torch.Tensor): Input tensor with shape [batch_size, num_patches, num_heads, head_dim].
        patch_resolution (tuple): Tuple containing the patch resolution (t, h, w).
        chunk_size (int): Size of the chunk. (maybe tuple or int, according to chunk type)

    Returns:
        torch.Tensor: Processed input tensor.
    """
    if isinstance(chunk_size, float) or isinstance(chunk_size, int):
        moba_chunk_size = int(chunk_size * patch_resolution[1] * patch_resolution[2])
    else:
        assert isinstance(
            chunk_size, (Tuple, list)
        ), f"chunk_size should be a tuple, list, or int, now it is: {type(chunk_size)}"
        if len(chunk_size) == 2:
            assert (
                patch_resolution[1] % chunk_size[0] == 0
                and patch_resolution[2] % chunk_size[1] == 0
            ), f"spatial patch_resolution {patch_resolution[1:]} should be divisible by 2d chunk_size {chunk_size}"
            nch, ncw = (
                patch_resolution[1] // chunk_size[0],
                patch_resolution[2] // chunk_size[1],
            )
            x = rearrange(
                x,
                "b (t nch ch ncw cw) n d -> b (nch ncw t ch cw) n d",
                t=patch_resolution[0],
                nch=nch,
                ncw=ncw,
                ch=chunk_size[0],
                cw=chunk_size[1],
            )
            moba_chunk_size = patch_resolution[0] * chunk_size[0] * chunk_size[1]
        elif len(chunk_size) == 3:
            assert (
                patch_resolution[0] % chunk_size[0] == 0
                and patch_resolution[1] % chunk_size[1] == 0
                and patch_resolution[2] % chunk_size[2] == 0
            ), f"patch_resolution {patch_resolution} should be divisible by 3d chunk_size {chunk_size}"
            nct, nch, ncw = (
                patch_resolution[0] // chunk_size[0],
                patch_resolution[1] // chunk_size[1],
                patch_resolution[2] // chunk_size[2],
            )
            x = rearrange(
                x,
                "b (nct ct nch ch ncw cw) n d -> b (nct nch ncw ct ch cw) n d",
                nct=nct,
                nch=nch,
                ncw=ncw,
                ct=chunk_size[0],
                ch=chunk_size[1],
                cw=chunk_size[2],
            )
            moba_chunk_size = chunk_size[0] * chunk_size[1] * chunk_size[2]
        else:
            raise ValueError(
                f"chunk_size should be a int, or a tuple of length 2 or 3, now it is: {len(chunk_size)}"
            )

    return x, moba_chunk_size


def process_moba_output(
    x,
    patch_resolution,
    chunk_size,
):
    if isinstance(chunk_size, float) or isinstance(chunk_size, int):
        pass
    elif len(chunk_size) == 2:
        x = rearrange(
            x,
            "b (nch ncw t ch cw) n d -> b (t nch ch ncw cw) n d",
            nch=patch_resolution[1] // chunk_size[0],
            ncw=patch_resolution[2] // chunk_size[1],
            t=patch_resolution[0],
            ch=chunk_size[0],
            cw=chunk_size[1],
        )
    elif len(chunk_size) == 3:
        x = rearrange(
            x,
            "b (nct nch ncw ct ch cw) n d -> b (nct ct nch ch ncw cw) n d",
            nct=patch_resolution[0] // chunk_size[0],
            nch=patch_resolution[1] // chunk_size[1],
            ncw=patch_resolution[2] // chunk_size[2],
            ct=chunk_size[0],
            ch=chunk_size[1],
            cw=chunk_size[2],
        )

    return x



def generate_data(batch_size, seqlen, num_head, head_dim, dtype):
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.cuda.current_device()

    q = torch.randn((batch_size, seqlen, num_head, head_dim), requires_grad=True).to(
        dtype=dtype, device="cuda"
    )
    k = torch.randn((batch_size, seqlen, num_head, head_dim), requires_grad=True).to(
        dtype=dtype, device="cuda"
    )
    v = torch.randn((batch_size, seqlen, num_head, head_dim), requires_grad=True).to(
        dtype=dtype, device="cuda"
    )
    print(f"q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
    cu_seqlens = torch.arange(
        0, q.shape[0] * q.shape[1] + 1, q.shape[1], dtype=torch.int32, device="cuda"
    )
    max_seqlen = q.shape[1]
    q = rearrange(q, "b s ... -> (b s) ...")
    k = rearrange(k, "b s ... -> (b s) ...")
    v = rearrange(v, "b s ... -> (b s) ...")

    return q, k, v, cu_seqlens, max_seqlen


def test_attn_varlen_moba_speed(
    batch,
    head,
    seqlen,
    head_dim,
    moba_chunk_size,
    moba_topk,
    dtype=torch.bfloat16,
    select_mode="threshold",
    simsum_threshold=0.25,
    threshold_type="query_head",
):
    """Speed test comparing flash_attn vs moba_attention"""
    
    q, k, v, cu_seqlen, max_seqlen = generate_data(batch, seqlen, head, head_dim, dtype)
    print(
        f"batch:{batch} head:{head} seqlen:{seqlen} chunk:{moba_chunk_size} topk:{moba_topk} select_mode: {select_mode} simsum_threshold:{simsum_threshold}"
    )
    vo_grad = torch.randn_like(q)

    
    warmup_iters = 3
    perf_test_iters = 10

    
    for _ in range(warmup_iters):
        o = flash_attn_varlen_func(
            q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, causal=False
        )
        torch.autograd.backward(o, vo_grad)

    torch.cuda.synchronize()
    start_flash = time.perf_counter()
    for _ in range(perf_test_iters):
        o = flash_attn_varlen_func(
            q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, causal=False
        )
        torch.autograd.backward(o, vo_grad)

    torch.cuda.synchronize()
    time_flash = (time.perf_counter() - start_flash) / perf_test_iters * 1000

    
    for _ in range(warmup_iters):
        om = moba_attn_varlen(
            q,
            k,
            v,
            cu_seqlen,
            max_seqlen,
            moba_chunk_size=moba_chunk_size,
            moba_topk=moba_topk,
            select_mode=select_mode,
            simsum_threshold=simsum_threshold,
            threshold_type=threshold_type,
        )
        torch.autograd.backward(om, vo_grad)

    torch.cuda.synchronize()
    start_moba = time.perf_counter()
    for _ in range(perf_test_iters):
        om = moba_attn_varlen(
            q,
            k,
            v,
            cu_seqlen,
            max_seqlen,
            moba_chunk_size=moba_chunk_size,
            moba_topk=moba_topk,
            select_mode=select_mode,
            simsum_threshold=simsum_threshold,
            threshold_type=threshold_type,
        )
        torch.autograd.backward(om, vo_grad)

    torch.cuda.synchronize()
    time_moba = (time.perf_counter() - start_moba) / perf_test_iters * 1000

    print(f"Flash: {time_flash:.2f}ms, MoBA: {time_moba:.2f}ms")
    print(f"Speedup:  {time_flash / time_moba:.2f}x")


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=1 \
    python -u csrc/attn/vmoba_attn/vmoba/vmoba.py
    """
    test_attn_varlen_moba_speed(
        batch=1,
        head=12,
        seqlen=32760,
        head_dim=128,
        moba_chunk_size=32760 // 3 // 6 // 4,
        moba_topk=3,
        select_mode="threshold",
        simsum_threshold=0.3,
        threshold_type="query_head",
    )
