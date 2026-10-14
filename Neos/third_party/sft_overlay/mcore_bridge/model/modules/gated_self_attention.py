# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from megatron.core.parallel_state import get_tensor_model_parallel_rank
from megatron.core.tensor_parallel import all_gather_last_dim_from_tensor_parallel_region
from megatron.core.transformer.attention import SelfAttention

try:
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    SplitAlongDim = None


class GatedSelfAttention(SelfAttention):

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, *args, **kwargs):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        assert self.config.num_query_groups is not None
        if getattr(self, 'world_size', None) is not None and self.config.num_query_groups < self.world_size:
            
            
            
            
            
            
            
            
            
            
            
            mixed_qkv = all_gather_last_dim_from_tensor_parallel_region(mixed_qkv)
            idx = get_tensor_model_parallel_rank() // (self.world_size // self.config.num_query_groups)
            size = mixed_qkv.size()[-1] // self.config.num_query_groups
            mixed_qkv = mixed_qkv[:, :, idx * size:(idx + 1) * size]

        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            ((self.num_attention_heads_per_partition // self.num_query_groups_per_partition * 2 + 2)
             * self.hidden_size_per_attention_head),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        split_arg_list = [
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition
             * self.hidden_size_per_attention_head * 2),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            
            
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:

            
            
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        if getattr(self, 'world_size', None) is not None and self.config.num_query_groups < self.world_size:
            
            
            
            idx = get_tensor_model_parallel_rank() % (self.world_size // self.config.num_query_groups)
            size = query.shape[2] // (self.world_size // self.config.num_query_groups)
            query = query[:, :, idx * size:(idx + 1) * size, :]
        query, gate = query[:, :, ::2], query[:, :, 1::2]
        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value, gate
