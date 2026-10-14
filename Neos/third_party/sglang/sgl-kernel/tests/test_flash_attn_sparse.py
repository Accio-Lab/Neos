import math
from typing import List, Optional, Tuple

import pytest
import torch
from einops import rearrange, repeat
from sgl_kernel.sparse_flash_attn import (
    convert_vertical_slash_indexes,
    convert_vertical_slash_indexes_mergehead,
    sparse_attn_func,
)
from test_flash_attention import construct_local_mask, is_fa3_supported


def ref_attn(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        lse: (batch_size, nheads, seqlen_q)
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))

    lse_ref = scores.logsumexp(dim=-1)

    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf")
        )
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    
    
    if query_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    dropout_scaling = 1.0 / (1 - dropout_p)
    
    
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)

    return output.to(dtype=dtype_og), lse_ref


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        
        q = query[start_idx : start_idx + query_len].clone()
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.skipif(
    not is_fa3_supported(),
    reason="flash_attn at sgl-kernel is only supported on sm90 or sm80",
)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "seq_lens",
    [
        (1, 1),
        (1, 1024),
        (1, 2048),
        (1023, 2049),
        (1023, 1023),
        (32, 32),
        (65, 65),
        (129, 129),
    ],
)
@pytest.mark.parametrize("num_heads", [1, 2, 4])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("NNZ_S", [0, 1, 2, 3, 7, 15, 32])
@torch.inference_mode()
def test_sparse_attention(
    batch_size,
    seq_lens,
    num_heads,
    head_size,
    dtype,
    NNZ_S,
) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    block_size_M = 64
    block_size_N = 64
    seqlen_q, seqlen_k = seq_lens
    q = torch.randn(
        batch_size, seqlen_q, num_heads, head_size, dtype=dtype, requires_grad=False
    )
    k = torch.randn(
        batch_size, seqlen_k, num_heads, head_size, dtype=dtype, requires_grad=False
    )
    v = torch.randn(
        batch_size, seqlen_k, num_heads, head_size, dtype=dtype, requires_grad=False
    )
    NUM_ROWS = (seqlen_q + block_size_M - 1) // block_size_M
    if NNZ_S * block_size_N > seqlen_k:
        return
    NNZ_V = seqlen_k - NNZ_S * block_size_N
    block_count = torch.tensor(
        [NNZ_S] * batch_size * NUM_ROWS * num_heads, dtype=torch.int32
    ).reshape(batch_size, num_heads, NUM_ROWS)
    column_count = torch.tensor(
        [NNZ_V] * batch_size * NUM_ROWS * num_heads, dtype=torch.int32
    ).reshape(batch_size, num_heads, NUM_ROWS)
    block_offset = torch.tensor(
        [[i * block_size_N for i in range(NNZ_S)]] * batch_size * NUM_ROWS * num_heads,
        dtype=torch.int32,
    ).reshape(batch_size, num_heads, NUM_ROWS, NNZ_S)
    column_index = torch.tensor(
        [[NNZ_S * block_size_N + i for i in range(NNZ_V)]]
        * batch_size
        * NUM_ROWS
        * num_heads,
        dtype=torch.int32,
    ).reshape(batch_size, num_heads, NUM_ROWS, NNZ_V)
    out, lse = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        return_softmax_lse=True,
    )

    ref_out, ref_lse = ref_attn(q, k, v)

    torch.testing.assert_close(
        out, ref_out, atol=2e-2, rtol=1e-2
    ), f"{torch.max(torch.abs(out - ref_out))}"
    torch.testing.assert_close(
        lse, ref_lse, atol=2e-2, rtol=1e-2
    ), f"{torch.max(torch.abs(lse - ref_lse))}"




@pytest.mark.skipif(
    not is_fa3_supported(),
    reason="flash_attn at sgl-kernel is only supported on sm90 or sm80",
)
@pytest.mark.parametrize("causal", [True, False])
def test_convert_vertical_slash_indexes(causal):
    
    q_seqlens = torch.tensor([4], dtype=torch.int32, device="cuda")  
    kv_seqlens = torch.tensor([4], dtype=torch.int32, device="cuda")
    vertical_indexes = torch.tensor(
        [[[1, 3]]], dtype=torch.int32, device="cuda"
    )  
    slash_indexes = torch.tensor(
        [[[2]]], dtype=torch.int32, device="cuda"
    )  
    context_size = 4
    block_size_M = 2
    block_size_N = 2

    
    block_count, block_offset, column_count, column_index = (
        convert_vertical_slash_indexes(
            q_seqlens,
            kv_seqlens,
            vertical_indexes,
            slash_indexes,
            context_size,
            block_size_M,
            block_size_N,
            causal=causal,
        )
    )

    
    
    
    
    
    
    
    

    expected_column_index = torch.tensor(
        [[[[0, 0], [0, 0]]]], dtype=torch.int32, device="cuda"
    )

    
    if not causal:
        
        expected_column_index = torch.tensor(
            [[[[1, 0], [1, 3]]]], dtype=torch.int32, device="cuda"
        )

    
    assert torch.equal(column_index, expected_column_index)



@pytest.mark.skipif(
    not is_fa3_supported(),
    reason="flash_attn at sgl-kernel is only supported on sm90 or sm80",
)
@pytest.mark.parametrize("causal", [True, False])
def test_convert_vertical_slash_indexes_mergehead(causal):
    
    q_seqlens = torch.tensor([4], dtype=torch.int32, device="cuda")
    kv_seqlens = torch.tensor([4], dtype=torch.int32, device="cuda")
    vertical_indexes = torch.tensor(
        [
            [
                [1, 3],  
                [2, 0],  
            ]
        ],
        dtype=torch.int32,
        device="cuda",
    )  
    slash_indexes = torch.tensor(
        [
            [
                [2, 0],  
                [1, 3],  
            ]
        ],
        dtype=torch.int32,
        device="cuda",
    )  
    vertical_indices_count = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
    slash_indices_count = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    context_size = 4
    block_size_M = 2
    block_size_N = 2

    
    block_count, block_offset, column_count, column_index = (
        convert_vertical_slash_indexes_mergehead(
            q_seqlens,
            kv_seqlens,
            vertical_indexes,
            slash_indexes,
            vertical_indices_count,
            slash_indices_count,
            context_size,
            block_size_M,
            block_size_N,
            causal=causal,
        )
    )

    
    
    
    

    expected_column_index = torch.tensor(
        [[[[1, 0], [1, 3]], [[-1079459945, -1077788999], [-1080050043, -1104625879]]]],
        dtype=torch.int32,
        device="cuda",
    )

    if not causal:
        
        expected_column_index = torch.tensor(
            [[[[1, 0], [1, 3]], [[2, -1077788999], [2, -1104625879]]]],
            dtype=torch.int32,
            device="cuda",
        )

    
    assert torch.equal(column_index, expected_column_index)


































































































if __name__ == "__main__":
    pytest.main([__file__])
