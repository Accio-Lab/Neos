# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import List

import torch
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig


@dataclass
class Qwen2VLTransformerConfig(TransformerConfig):
    transformer_impl: str = "transformer_engine"
    rotary_base: int = None
    rotary_scaling_factor: int = None
    max_position_embeddings: int = None
    moe_aux_loss_coeff: float = 0.0
    mrope_section: List[int] = field(default_factory=lambda: [16, 24, 24])

    
    
    
    


def get_vision_model_config(config):
    
    

    
    
    if config.num_layers in [28, 36]:
        config.ffn_hidden_size = 3420
    else:
        config.ffn_hidden_size = 3456

    from megatron.core import parallel_state

    
    
    
    
    
    
    config.num_layers = 32  
    config.num_attention_heads = 16  
    config.add_bias_linear = True  
    config.add_qkv_bias = True  
    config.hidden_size = 1280  
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0

    
    
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.num_query_groups = config.num_attention_heads  
    config.layernorm_zero_centered_gamma = False  
    config.apply_query_key_layer_scaling = False  
    config.bias_activation_fusion = False  
    config.bias_dropout_fusion = False  
    config.attention_softmax_in_fp32 = True  
    
    config.seq_length = 1

    config.tp_comm_overlap = False
    config.sequence_parallel = False
    config.temporal_patch_size = 2
    config.patch_size = 14
    config.in_channels = 3
    config.spatial_merge_size = 2

    config.fullatt_block_indexes = [7, 15, 23, 31]
    config._qwen2_5_vl_window_size = 112
    config.context_parallel_size = 1
    config.pipeline_model_parallel_size = 1
    config.num_layers_in_first_pipeline_stage = None
    config.num_layers_in_last_pipeline_stage = None
    config.virtual_pipeline_model_parallel_size = 1
    config.pipeline_model_parallel_layout = None
    return config


def get_vision_projection_config(config, embed_dim, spatial_merge_size):
    
    
    
    
    
    
    
    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.add_bias_linear = True
    config.ffn_hidden_size = embed_dim * (spatial_merge_size**2)
    config.activation_func = torch.nn.functional.gelu
    config.tp_comm_overlap = False
    config.sequence_parallel = False
    config.context_parallel_size = 1
    return config
