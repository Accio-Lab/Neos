import torch

from ..generation.continuous_batching import PagedAttentionCache
from ..modeling_flash_attention_utils import lazy_import_paged_flash_attention


def paged_attention_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,  
    cache: PagedAttentionCache,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor | dict[str, torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int | dict[str, int],
    block_table: torch.Tensor | None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Performs the forward pass of attention with paged key-value cache. This function handles the cache updates and
    performs the attention computation. For decode-only batches (when block_table is provided), uses
    `flash_attn_with_kvcache` for fused attention + cache update. Otherwise uses `flash_attn_varlen_func`.
    See the [paged attention guide](https://huggingface.co/docs/transformers/en/paged_attention) for more details.

    Args:
        q: (1, nheads, total_q, headdim), where total_q = total number of query tokens in the batch.
        k: (1, nheads_k, total_k, headdim), where total_k = total number of key tokens in the batch.
        v: (1, nheads_k, total_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seq_lens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seq_lens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        block_table: (num_groups, batch_size, max_blocks_per_seq), dtype int32. Block table for paged KV cache.
            If provided, uses flash_attn_with_kvcache for fused attention + cache update. For each request, the block
            table is a vector of size (max_blocks_per_seq,) with indices indicating the physical location of the cache
            to read from and write to. The kernel, using the cache_seqlens for that request, knows how much cache to
            read and dispatches the read using the block table. Same for the write. If a request has fewer than
            max_blocks_per_seq blocks, the block table is padded with -1s to indicate that the block is not allocated.
    """
    
    flash_attn_varlen_func, flash_attn_with_kvcache = lazy_import_paged_flash_attention(
        module.config._attn_implementation
    )

    
    sliding_window = (-1, -1) if not getattr(module, "sliding_window", False) else (module.sliding_window - 1, 0)
    layer_type = "full_attention" if sliding_window == (-1, -1) else "sliding_attention"
    if isinstance(cu_seq_lens_k, dict):
        cu_seq_lens_k = cu_seq_lens_k[layer_type]
        max_seqlen_k = max_seqlen_k[layer_type]

    
    if block_table is None:
        
        k, v = cache.update(
            key_states=k,
            value_states=v,
            layer_idx=module.layer_idx,
            read_index=kwargs["read_index"],
            write_index=kwargs["write_index"],
        )
        custom_kwargs = {"s_aux": kwargs.get("s_aux")} if "s_aux" in kwargs else {}
        attn_output = flash_attn_varlen_func(
            q.transpose(1, 2).squeeze(0).contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seq_lens_q.to(torch.int32),
            cu_seq_lens_k.to(torch.int32).clone(),
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=module.scaling,
            causal=True,  
            window_size=sliding_window,  
            **custom_kwargs,
        )
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

    
    else:
        flash_kwargs = {"s_aux": kwargs["s_aux"]} if "s_aux" in kwargs else {}  
        attn_output = _paged_decode_forward(
            module, q, k, v, cache, cu_seq_lens_k, sliding_window, flash_attn_with_kvcache, block_table, **flash_kwargs
        )
    return attn_output, None


@torch.compiler.disable
def _paged_decode_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: PagedAttentionCache,
    cu_seq_lens_k: torch.Tensor,
    sliding_window: tuple[int, int],
    flash_attn_with_kvcache,
    block_table: torch.Tensor,
    **flash_kwargs,
) -> torch.Tensor:
    """Decode fast path using flash_attn_with_kvcache. Disabled because FA3 has issue with tracing this."""
    
    group_idx, layer_idx_in_group = cache.layer_index_to_group_indices[module.layer_idx]
    
    k_cache = cache.key_cache[layer_idx_in_group].view(-1, cache.block_size, cache.num_key_value_heads, cache.head_dim)
    v_cache = cache.value_cache[layer_idx_in_group].view(
        -1, cache.block_size, cache.num_key_value_heads, cache.head_dim
    )
    
    q = q.permute(2, 0, 1, 3).contiguous()
    k = k.permute(2, 0, 1, 3).contiguous()
    v = v.permute(2, 0, 1, 3).contiguous()
    
    
    batch_size = k.size(0)
    cache_seqlens = (cu_seq_lens_k[1 : batch_size + 1] - cu_seq_lens_k[:batch_size] - 1).to(torch.int32)
    
    flash_kwargs[cache.get_block_table_key(flash_attn_with_kvcache)] = block_table[group_idx]
    
    attn_output = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        softmax_scale=module.scaling,
        causal=True,
        window_size=sliding_window,
        **flash_kwargs,
    )
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0]
    
    return attn_output.squeeze(1)
