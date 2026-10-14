# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_combined.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py

# ruff: noqa: E501

import torch
import triton
from einops import rearrange
from packaging import version

from .ssd_bmm import _bmm_chunk_fwd
from .ssd_chunk_scan import _chunk_scan_fwd
from .ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd, chunk_state_varlen
from .ssd_state_passing import _state_passing_fwd

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


def is_int_pow_2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def _mamba_chunk_scan_combined_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    chunk_indices=None,
    chunk_offsets=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    state_dtype=None,
    out=None,
):
    assert is_int_pow_2(chunk_size), "chunk_size must be integer power of 2"
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if (
        x.stride(-1) != 1 and x.stride(1) != 1
    ):  
        x = x.contiguous()
    if (
        z is not None and z.stride(-1) != 1 and z.stride(1) != 1
    ):  
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        if cu_seqlens is None:
            assert initial_states.shape == (batch, nheads, headdim, dstate)
        else:
            assert initial_states.shape == (
                len(cu_seqlens) - 1,
                nheads,
                headdim,
                dstate,
            )

    
    
    
    
    
    
    
    

    
    
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit
    )

    
    
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)

    
    
    
    
    
    
    
    
    
    
    
    states, final_states = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum,
        initial_states=(
            rearrange(initial_states, "... p n -> ... (p n)")
            if initial_states is not None
            else None
        ),
        seq_idx=seq_idx,
        chunk_size=chunk_size,
        out_dtype=state_dtype if state_dtype is not None else C.dtype,
        is_cont_batched=cu_seqlens is not None,
        chunk_offsets=chunk_offsets,
    )
    states, final_states = (
        rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]
    )

    
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)

    
    
    
    
    
    
    
    
    
    
    out_x = _chunk_scan_fwd(
        CB,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        D=D,
        z=z,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        initial_states=initial_states,
        out=out,
    )
    if cu_seqlens is None:
        return out_x, dt, dA_cumsum, states, final_states
    else:
        assert (
            batch == 1
        ), "passing cu_seqlens to get the varlen states is only supported if batch dimension is 1"
        varlen_states = chunk_state_varlen(
            B.squeeze(0),
            x.squeeze(0),
            dt.squeeze(0),
            dA_cumsum.squeeze(0),
            cu_seqlens,
            states.squeeze(0),
            initial_states=initial_states,
        )
        return out_x, dt, dA_cumsum, states, final_states, varlen_states


def mamba_chunk_scan_combined(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    chunk_indices=None,
    chunk_offsets=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    out=None,
    return_final_states=False,
    return_varlen_states=False,
    state_dtype=None,
):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        cu_seqlens: (num_sequences + 1) or None, only used if return_varlen_states is True
        dt_softplus: Whether to apply softplus to dt
        out: Preallocated output tensor
        state_dtype: The data type of the ssm state
    """

    if not return_varlen_states:
        cu_seqlens = None
    else:
        assert (
            cu_seqlens is not None
        ), "cu_seqlens must be provided if return_varlen_states is True"
    out_x, dt_out, dA_cumsum, states, final_states, *rest = (
        _mamba_chunk_scan_combined_fwd(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            cu_seqlens=cu_seqlens,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            out=out,
            state_dtype=state_dtype,
        )
    )
    if not return_varlen_states:
        if not return_final_states:
            return
        else:
            return final_states
    else:
        varlen_states = rest[0]
        return (
            (varlen_states)
            if not return_final_states
            else (final_states, varlen_states)
        )
