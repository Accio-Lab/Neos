"""Compatibility fix for Qwen3.5 gated GQA when TP exceeds KV heads.

Some Megatron versions all-gather the Q/gate/K/V projection for replicated
KV groups, partition Q back to the local TP rank, but return the full gate.
Keep the corresponding gate heads, not an arbitrary reshape/truncation.
This is installed only by the opt-in Neos Megatron launch path.
"""

from functools import wraps


def partition_output_gate(query, gate, *, tp_size, kv_heads, tp_rank):
    """Align gate with already partitioned query; preserve autograd views."""
    if gate.shape == query.shape:
        return gate
    if (
        query.ndim != 4
        or gate.ndim != 4
        or query.shape[:2] != gate.shape[:2]
        or query.shape[-1] != gate.shape[-1]
        or kv_heads <= 0
        or tp_size <= kv_heads
        or tp_size % kv_heads
        or not 0 <= tp_rank < tp_size
    ):
        raise ValueError(f"Unexpected Qwen3.5 query/gate layout: Q={query.shape}, gate={gate.shape}")
    replicas = tp_size // kv_heads
    if gate.shape[2] != query.shape[2] * replicas:
        raise ValueError(f"Unexpected replicated gate heads: Q={query.shape}, gate={gate.shape}")
    first = (tp_rank % replicas) * query.shape[2]
    return gate[:, :, first : first + query.shape[2], :]


def patch_qwen35_attention_output_gate():
    from megatron.core import parallel_state
    from mbridge.models.qwen3_5.attention import Qwen3_5VLSelfAttention

    original = Qwen3_5VLSelfAttention.get_query_key_value_tensors
    if getattr(original, "_verl_gate_tp_partition", False):
        return False

    @wraps(original)
    def get_query_key_value_tensors(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if len(result) != 4:
            return result
        query, key, value, gate = result
        gate = partition_output_gate(
            query, gate, tp_size=self.world_size,
            kv_heads=self.config.num_query_groups,
            tp_rank=parallel_state.get_tensor_model_parallel_rank(),
        )
        return query, key, value, gate

    get_query_key_value_tensors._verl_gate_tp_partition = True
    Qwen3_5VLSelfAttention.get_query_key_value_tensors = get_query_key_value_tensors
    print("qwen35_attention_output_gate_tp_partition=enabled", flush=True)
    return True
