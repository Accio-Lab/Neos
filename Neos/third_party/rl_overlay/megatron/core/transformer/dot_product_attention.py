# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.dot_product_attention_context_parallel import (
    AttentionFuncionWithContextParallel,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    attention_mask_func,
    is_layer_window_attention,
    make_sharded_tensors_for_checkpoint,
)
from megatron.core.utils import divide


class DotProductAttention(MegatronModule):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models:
    https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        if self.config.context_parallel_size > 1:
            assert attention_dropout is None and self.config.attention_dropout == 0.0, (
                f'DotProductAttention with context parallelism does not support attention dropout,'
                f' but got {self.config.context_parallel_size=},'
                f' {attention_dropout=}, and {self.config.attention_dropout=}.'
            )

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        else:
            assert hasattr(
                pg_collection, 'tp'
            ), "DotProductAttention pg_collection must have tp process group"
        self.pg_collection = pg_collection
        self.tp_group = self.pg_collection.tp

        world_size = pg_collection.tp.size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        coeff = None
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.softmax_scale /= coeff

        if is_layer_window_attention(
            self.config.window_size, self.config.window_attn_skip_freq, layer_number
        ):
            window_size = self.config.window_size
        else:
            window_size = None

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
            window_size=window_size,
        )

        
        
        
        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

        if self.config.softmax_type == "vanilla":
            self.softmax_offset = None
        elif self.config.softmax_type == "off-by-one":
            self.softmax_offset = torch.zeros(
                self.num_attention_heads_per_partition,
                device=torch.cuda.current_device(),
                dtype=self.config.params_dtype,
            )
        elif self.config.softmax_type == "learnable":
            self.register_parameter(
                "softmax_offset",
                torch.nn.Parameter(
                    torch.empty(
                        self.num_attention_heads_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=self.config.params_dtype,
                    )
                ),
            )
            if config.perform_initialization:
                self.softmax_offset = config.init_method(self.softmax_offset)
        else:
            raise ValueError("Softmax type not supported")

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Forward."""
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )
        assert attention_bias is None, "Attention bias is not supported for DotProductAttention."

        
        
        

        
        
        
        

        
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        if self.config.context_parallel_size > 1:
            output = AttentionFuncionWithContextParallel.apply(
                query,
                key,
                value,
                attention_mask,
                self.config.attention_dropout,
                self.softmax_scale,
                parallel_state.get_context_parallel_group(),
            )
            output = output.view(query.shape[0], query.shape[1], self.hidden_size_per_partition)
            return output

        
        output_size = (query.size(1), query.size(2), query.size(0), key.size(0))

        
        
        
        
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu"
        )

        
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),  
            key.transpose(0, 1).transpose(1, 2),  
            beta=0.0,
            alpha=self.softmax_scale,
        )

        
        attention_scores = matmul_result.view(*output_size)

        
        
        

        
        attention_probs: Tensor = self.scale_mask_softmax(
            attention_scores, attention_mask, self.softmax_offset
        )
        
        

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        
        
        

        
        

        
        output_size = (value.size(1), value.size(2), query.size(0), value.size(3))

        
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)

        
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        
        context = context.view(*output_size)

        
        context = context.permute(2, 0, 1, 3).contiguous()

        
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict for the learnable softmax offset parameter"""
        if self.config.softmax_type == "learnable":
            state_dict = self.state_dict(prefix="", keep_vars=True)
        else:
            state_dict = {}
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {'softmax_offset': 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata['dp_cp_group'],
        )
