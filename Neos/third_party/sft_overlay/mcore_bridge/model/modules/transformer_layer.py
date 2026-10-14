# Copyright (c) ModelScope Contributors. All rights reserved.
import enum
import inspect
import torch
from contextlib import contextmanager
from functools import partial
from megatron.core.extensions.transformer_engine import TEFusedMLP
from megatron.core.models.common.embeddings import rope_utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (gather_from_sequence_parallel_region,
                                                    scatter_to_sequence_parallel_region)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer as McoreTransformerLayer
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules, get_transformer_layer_offset
from megatron.core.utils import get_pg_rank
from typing import Optional

from mcore_bridge.utils import get_logger, split_cp_inputs

try:
    from megatron.core.transformer.enums import CudaGraphScope
except ImportError:

    class CudaGraphScope(enum.Enum):
        """Cuda Graph Scope - defines which parts of the model to capture."""

        full_iteration = 1  
        attn = 2  
        mlp = 3  
        moe = 4  
        moe_router = 5  
        moe_preprocess = 6  
        mamba = 7  


logger = get_logger()


class TransformerLayer(McoreTransformerLayer):

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: Optional[int] = None,
        **kwargs,
    ):
        self.submodules_config = submodules
        super(McoreTransformerLayer, self).__init__(config=config, vp_stage=vp_stage)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        
        
        
        
        
        
        
        if is_mtp_layer or not add_layer_offset:
            self.layer_number = layer_number
        else:
            self.layer_number = layer_number + get_transformer_layer_offset(self.config, vp_stage,
                                                                            get_pg_rank(pg_collection.pp))
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout
        self.is_mtp_layer = is_mtp_layer

        
        
        self.input_layernorm = submodules.input_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                
                attention_optional_kwargs['cp_comm_type'] = config.cp_comm_type[self.layer_number - 1]
            else:
                attention_optional_kwargs['cp_comm_type'] = config.cp_comm_type

        attention_optional_kwargs['pg_collection'] = pg_collection
        if pp_layer_offset is not None:
            attention_optional_kwargs['pp_layer_offset'] = pp_layer_offset

        
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        
        self.pre_cross_attn_layernorm = submodules.pre_cross_attn_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config)

        
        self.pre_mlp_layernorm = submodules.pre_mlp_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        
        self.mlp = self._build_mlp(submodules.mlp)
        
        self.mlp_bda = build_module(submodules.mlp_bda)
        self.is_moe_layer = isinstance(self.mlp, MoELayer)

        self.recompute_input_layernorm = False
        self.recompute_pre_mlp_layernorm = False
        self.recompute_mlp = False
        if self.config.recompute_granularity == 'selective':
            assert self.config.recompute_modules is not None
            if 'layernorm' in self.config.recompute_modules:
                if not isinstance(self.input_layernorm, IdentityOp):
                    self.recompute_input_layernorm = True
                    if self.config.fp8 or self.config.fp4:
                        self.self_attention.set_for_recompute_input_layernorm()

                def can_recompute_pre_mlp_layernorm_for_cudagraph():
                    if (not self.is_moe_layer or CudaGraphScope.moe_router not in self.config.cuda_graph_scope
                            or self.config.cuda_graph_impl == 'local'):
                        
                        return True
                    if (self.config.moe_shared_expert_intermediate_size is not None
                            and self.config.moe_shared_expert_overlap):
                        
                        
                        
                        logger.warning(
                            'pre_mlp_layernorm recompute is not supported with moe router '
                            'cudagraph + shared expert overlap. Disabling pre_mlp_layernorm '
                            'recompute.', )
                        return False
                    if CudaGraphScope.moe_preprocess in self.config.cuda_graph_scope and (
                            self.config.moe_token_dispatcher_type == 'alltoall' or self.config.moe_latent_size):
                        
                        
                        
                        
                        
                        return True
                    logger.warning(
                        'pre_mlp_layernorm recompute is only supported with moe router + '
                        'preprocess cudagraph will alltoall token dispatcher or latent MoE. '
                        'Disabling pre_mlp_layernorm recompute.', )
                    return False

                if (not isinstance(self.pre_mlp_layernorm, IdentityOp)
                        and can_recompute_pre_mlp_layernorm_for_cudagraph()):
                    self.recompute_pre_mlp_layernorm = True
                    if self.config.fp8 or self.config.fp4:
                        if isinstance(self.mlp, MoELayer):
                            self.mlp.set_for_recompute_pre_mlp_layernorm()
                        else:
                            from megatron.core.extensions.transformer_engine import set_save_original_input

                            set_save_original_input(self.mlp.linear_fc1)
            if 'mlp' in self.config.recompute_modules:
                if not self.is_moe_layer:
                    self.recompute_mlp = True
        if hasattr(self, '_set_offload_modules'):
            from megatron.core.transformer.transformer_layer import _get_offloading_interface
            self._set_offload_modules()
            self.off_interface = _get_offloading_interface()
            self.mlp_norm_manager = None
        else:
            if hasattr(self.config, 'fine_grained_activation_offloading'):
                self.offload_attn_norm = (
                    self.config.fine_grained_activation_offloading and 'attn_norm' in self.config.offload_modules
                    and not isinstance(self.input_layernorm, IdentityOp))
                self.offload_mlp_norm = (
                    self.config.fine_grained_activation_offloading and 'mlp_norm' in self.config.offload_modules
                    and not isinstance(self.pre_mlp_layernorm, IdentityOp))

        
        
        
        
        
        
        self.bias_dropout_add_exec_handler = torch.enable_grad

    @contextmanager
    def _patch_apply_rotary_pos_emb(self):
        if self.config.apply_rope_fusion:
            yield
            return

        _origin_apply_rotary_pos_emb_bshd = rope_utils._apply_rotary_pos_emb_bshd

        def _apply_rotary_pos_emb_bshd(
            t: torch.Tensor,
            freqs: torch.Tensor,
            rotary_interleaved: bool = False,
            mla_rotary_interleaved: Optional[bool] = None,
            multi_latent_attention: Optional[bool] = None,
            mscale: float = 1.0,
            inverse: bool = False,
            mla_output_remove_interleaving: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            """Apply rotary positional embedding to input tensor T.

            check https://kexue.fm/archives/8265 for detailed formulas

            Args:
                t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
                freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

            Returns:
                Tensor: The input tensor after applying RoPE
            """
            mscale = self.config.attention_scaling
            rot_dim = freqs.shape[-1]

            
            t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
            if multi_latent_attention is None:
                multi_latent_attention = self.config.multi_latent_attention
            if mla_rotary_interleaved is None:
                mla_rotary_interleaved = multi_latent_attention
            if mla_rotary_interleaved:
                x1 = t[..., 0::2]
                x2 = t[..., 1::2]
                t = torch.cat((x1, x2), dim=-1)

            
            
            cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
            sin_ = (torch.sin(freqs) * mscale).to(t.dtype)
            if inverse:
                sin_ = -sin_

            t = (t * cos_) + (rope_utils._rotate_half(t, rotary_interleaved) * sin_)
            
            
            
            if mla_rotary_interleaved and mla_output_remove_interleaving:
                x1, x2 = torch.chunk(t, 2, dim=-1)
                t = torch.stack((x1, x2), dim=-1).flatten(start_dim=-2)

            return torch.cat((t, t_pass), dim=-1)

        rope_utils._apply_rotary_pos_emb_bshd = _apply_rotary_pos_emb_bshd
        try:
            yield
        finally:
            rope_utils._apply_rotary_pos_emb_bshd = _origin_apply_rotary_pos_emb_bshd

    def _build_mlp(self, mlp_spec):
        pg_collection = self.pg_collection
        additional_mlp_kwargs = {}
        if isinstance(mlp_spec, partial):
            if 'layer_number' in inspect.signature(mlp_spec.func).parameters:
                additional_mlp_kwargs['layer_number'] = self.layer_number
            mlp = mlp_spec(
                config=self.config,
                pg_collection=pg_collection,
                is_mtp_layer=self.is_mtp_layer,
                **additional_mlp_kwargs)
        else:
            
            from mcore_bridge.model.gpts.glm4 import Glm4MLP
            from mcore_bridge.model.mm_gpts.gemma4 import Gemma4MLP, Gemma4MoELayer

            
            
            
            
            if isinstance(mlp_spec, ModuleSpec):
                if mlp_spec.module in (MoELayer, Gemma4MoELayer, TEGroupedMLP, SequentialMLP):
                    additional_mlp_kwargs['pg_collection'] = pg_collection
                    
                    if mlp_spec.module == MoELayer and 'is_mtp_layer' in inspect.signature(MoELayer).parameters:
                        additional_mlp_kwargs['is_mtp_layer'] = self.is_mtp_layer
                elif mlp_spec.module in (MLP, Glm4MLP):
                    assert hasattr(pg_collection, 'tp'), 'TP process group is required for MLP in TransformerLayer'
                    additional_mlp_kwargs['tp_group'] = pg_collection.tp
                elif mlp_spec.module == Gemma4MLP:
                    additional_mlp_kwargs['layer_number'] = self.layer_number
                elif TEFusedMLP is not None and mlp_spec.module == TEFusedMLP:
                    assert hasattr(pg_collection,
                                   'tp'), 'TP process group is required for TEFusedMLP in TransformerLayer'
                    additional_mlp_kwargs['tp_group'] = pg_collection.tp
                else:
                    logger.warning_once(f'Unknown MLP type: {mlp_spec.module}. Using default kwargs.')
            mlp = build_module(mlp_spec, config=self.config, **additional_mlp_kwargs)
        if hasattr(mlp, 'set_layer_number'):
            mlp.set_layer_number(self.layer_number)
        return mlp

    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        
        padding_mask = kwargs.pop('padding_mask', None)
        mlp_kwargs = {}
        if padding_mask is not None:
            kwargs['padding_mask'] = padding_mask
            mlp_kwargs['padding_mask'] = padding_mask
        with self._patch_apply_rotary_pos_emb():
            hidden_states, context = self._forward_attention(*args, **kwargs)
        
        mlp_padding_free = self.config.mlp_padding_free and 'attention_mask' in kwargs
        mask = None
        enable_sp = self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1
        pad_size = 0
        if mlp_padding_free and hidden_states.shape[1] > 1:
            if enable_sp:
                hidden_states = gather_from_sequence_parallel_region(hidden_states, tensor_parallel_output_grad=False)
            mlp_kwargs.pop('padding_mask', None)
            mask = ((~kwargs['attention_mask']).sum(dim=(1, 2)) > 0).t()
            if self.config.context_parallel_size > 1:
                mask = split_cp_inputs(mask, None, 0)
            hidden_states = hidden_states[mask][:, None]
            if enable_sp:
                tp_size = self.config.tensor_model_parallel_size
                num_tokens = hidden_states.shape[0]
                remainder = num_tokens % tp_size
                if remainder != 0:
                    pad_size = tp_size - remainder
                    hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, 0, 0, pad_size))
                hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        output = self._forward_mlp(hidden_states, kwargs.get('inference_context', None), **mlp_kwargs)
        if mask is not None:
            if enable_sp:
                output = gather_from_sequence_parallel_region(output, tensor_parallel_output_grad=False)
                if pad_size > 0:
                    output = output[:-pad_size]
            new_output = output.new_zeros((*mask.shape, output.shape[-1]))
            new_output[mask] = output.squeeze(1)
            output = new_output
            if enable_sp:
                output = scatter_to_sequence_parallel_region(output)
        return output, context
