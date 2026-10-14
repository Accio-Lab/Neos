# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from:
#   https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py
# Adapted from vLLM project (Apache-2.0).

import torch
import triton
from packaging import version

from .ssd_bmm import _bmm_chunk_fwd
from .ssd_chunk_scan import _chunk_scan_fwd
from .ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd
from .ssd_state_passing import _state_passing_fwd

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


def is_int_pow_2(n):
    """Return True if n is a positive integer power of 2."""
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def _mamba_chunk_scan_combined_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    return_intermediate_states=False,
    seq_idx=None,
    cu_chunk_seqlens=None,
    last_chunk_indices=None,
    intermediate_chunk_indices=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    state_dtype=None,
):
    assert is_int_pow_2(chunk_size), "chunk_size must be integer power of 2"
    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (
        seqlen,
        ngroups,
        dstate,
    ), f"B.shape={B.shape} != ({seqlen}, {ngroups}, {dstate})"
    assert dt.shape == (seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (cu_chunk_seqlens.shape[0] - 1,)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(0) != 1:  
        x = x.contiguous()
    if (
        z is not None and z.stride(-1) != 1 and z.stride(0) != 1
    ):  
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    assert cu_chunk_seqlens is not None, "Assuming varlen input - must supply cu_chunk_seqlens"
    assert last_chunk_indices is not None, "last_chunk_indices must be provided"

    if initial_states is not None:
        num_seqs = last_chunk_indices.shape[0]
        assert initial_states.shape == (num_seqs, nheads, headdim, dstate)

    
    
    
    
    
    
    
    

    
    
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt,
        A,
        chunk_size,
        cu_chunk_seqlens,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )

    
    
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, cu_chunk_seqlens, states_in_fp32=True)

    
    
    
    
    
    
    states = _state_passing_fwd(
        states.flatten(-2),  
        dA_cumsum,  
        cu_chunk_seqlens,
        initial_states=(
            initial_states.flatten(-2) if initial_states is not None else None
        ),  
        seq_idx=seq_idx,
        out_dtype=state_dtype if state_dtype is not None else C.dtype,
    )
    states = states.unflatten(-1, (-1, dstate))

    
    CB = _bmm_chunk_fwd(C, B, chunk_size, cu_chunk_seqlens, output_dtype=torch.float32)

    
    
    
    
    
    
    
    
    
    
    _chunk_scan_fwd(
        CB,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        cu_chunk_seqlens,
        out,  
        seq_idx,
        D=D,
        z=z,
        initial_states=initial_states,
    )

    if return_intermediate_states:
        return states

    final_states = states[last_chunk_indices]
    if intermediate_chunk_indices is not None:
        intermediate_states = states[intermediate_chunk_indices]
        return final_states, intermediate_states
    else:
        return final_states


def mamba_chunk_scan_combined_varlen(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    cu_chunk_seqlens,
    last_chunk_indices,
    seq_idx,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_intermediate_states=False,
    intermediate_chunk_indices=None,
    state_dtype=None,
):
    """
    Argument:
        x: (seqlen, nheads, headdim)
        dt: (seqlen, nheads)
        A: (nheads)
        B: (seqlen, ngroups, dstate)
        C: (seqlen, ngroups, dstate)
        chunk_size: int
        cu_chunk_seqlens: (nchunks + 1,)
        last_chunk_indices: (batch,)
        seq_idx: (nchunks,)
        out: (seqlen, nheads, headdim) preallocated output tensor
        D: (nheads, headdim) or (nheads,)
        z: (seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        dt_softplus: Whether to apply softplus to dt
        intermediate_chunk_indices: (N,) optional int64 tensor of chunk indices at which to
            extract intermediate SSM states. When provided, returns (final_states,
            intermediate_states) instead of just final_states.
        state_dtype: The data type of the ssm state
    Return:
        varlen_states: (batch, nheads, headdim, dstate), or
        (varlen_states, intermediate_states) if intermediate_chunk_indices is provided
    """

    assert seq_idx is not None

    varlen_states = _mamba_chunk_scan_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        out,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        return_intermediate_states=return_intermediate_states,
        seq_idx=seq_idx,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        intermediate_chunk_indices=intermediate_chunk_indices,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        state_dtype=state_dtype,
    )

    return varlen_states
