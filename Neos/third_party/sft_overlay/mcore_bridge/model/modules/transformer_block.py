# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from contextlib import nullcontext
from megatron.core import tensor_parallel
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import te_checkpoint
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_block import TransformerBlock as McoreTransformerBlock
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.utils import WrappedTensor, deprecate_inference_params, get_pg_rank, make_viewless_tensor
from typing import List, Optional, Set, Union, cast

try:
    from megatron.core.typed_torch import apply_module
except ImportError:
    apply_module = None

try:
    from megatron.core.transformer.hyper_connection import HyperConnectionModule, learned_output_contract
except ImportError:
    pass


class _TensorIdx:
    """Sentinel that marks a position in the flatten schema as a tensor index."""
    __slots__ = ('idx', )

    def __init__(self, idx):
        self.idx = idx


def _checkpoint_flatten(obj, tensors):
    """Recursively flatten a nested structure (dict/tuple/list/Tensor) into a schema.

    Tensors are appended to `tensors` and replaced in the schema by a _TensorIdx sentinel.
    Non-tensor leaves (int, bool, None, str, ...) are stored as-is.
    The schema mirrors the original structure and is captured in the checkpoint closure.
    """
    if torch.is_tensor(obj):
        idx = len(tensors)
        tensors.append(obj)
        return _TensorIdx(idx)
    elif isinstance(obj, dict):
        
        for k, v in obj.items():
            obj[k] = _checkpoint_flatten(v, tensors)
        return obj
    elif isinstance(obj, (tuple, list)):
        return type(obj)(_checkpoint_flatten(v, tensors) for v in obj)
    else:
        return obj  


def _checkpoint_unflatten(schema, tensors):
    """Reconstruct the original structure from a schema and a flat tensors list."""
    if isinstance(schema, _TensorIdx):
        return tensors[schema.idx]
    elif isinstance(schema, dict):
        
        for k, v in schema.items():
            schema[k] = _checkpoint_unflatten(v, tensors)
        return schema
    elif isinstance(schema, (tuple, list)):
        return type(schema)(_checkpoint_unflatten(v, tensors) for v in schema)
    else:
        return schema  



class TransformerBlock(McoreTransformerBlock):

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_bias: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_quantization_context: bool,
        padding_mask: Optional[torch.Tensor] = None,
        extract_layer_indices: Optional[Set[int]] = None,
        layer_offset: int = 0,
        **kwargs,
    ):
        """Forward method with activation checkpointing.

        Args:
            extract_layer_indices (Set[int], optional): Global layer
                indices (across all pipeline stages) from which to
                extract features.
            layer_offset (int): The global layer offset for the current
                pipeline stage. Used to convert local layer indices to
                global indices when checking extract_layer_indices.

        Returns:
            If extract_layer_indices is empty: hidden_states tensor
            If extract_layer_indices is non-empty: (hidden_states, intermediate_hidden_states) tuple
        """
        if extract_layer_indices is None:
            extract_layer_indices = set()
        intermediate_hidden_states: List[torch.Tensor] = []

        def custom(start: int, end: int):

            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                padding_mask=None,
                **kwargs,
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)

                    
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(self.config, layer.layer_number - 1)
                        
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(self.config, layer.layer_number - 1)
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with inner_quantization_context:
                        hidden_states, context = self._layer_forward(
                            layer,
                            hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                            **kwargs,
                        )
                return hidden_states, context

            return custom_forward

        
        _ckpt_attention_mask = attention_mask
        _ckpt_rotary_pos_emb = rotary_pos_emb
        extra_kwargs_keys = tuple(kwargs.keys())
        _extra_flat_tensors = []
        _extra_schemas = [_checkpoint_flatten(v, _extra_flat_tensors) for v in kwargs.values()]

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""

            def wrapped_forward(hidden_states, context, context_mask, padding_mask, *extra_flat):
                rebuilt = [_checkpoint_unflatten(s, extra_flat) for s in _extra_schemas]
                extra_kwargs = dict(zip(extra_kwargs_keys, rebuilt))
                return forward_func(
                    hidden_states,
                    _ckpt_attention_mask,
                    context,
                    context_mask,
                    _ckpt_rotary_pos_emb,
                    padding_mask,
                    **extra_kwargs,
                )

            
            if self.config.fp8 or self.config.fp4:
                return te_checkpoint(
                    wrapped_forward,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                    context,
                    context_mask,
                    padding_mask,
                    *_extra_flat_tensors,
                )
            else:
                return tensor_parallel.checkpoint(
                    wrapped_forward,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    context,
                    context_mask,
                    padding_mask,
                    *_extra_flat_tensors,
                )

        if self.config.recompute_method == 'uniform':
            
            
            
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                chunk_end = min(layer_idx + self.config.recompute_num_layers, self.num_layers_per_pipeline_rank)
                hidden_states, context = checkpoint_handler(custom(layer_idx, chunk_end))

                
                
                for idx in range(layer_idx, chunk_end):
                    if (idx + layer_offset) in extract_layer_indices:
                        
                        
                        if idx == chunk_end - 1:
                            intermediate_hidden_states.append(hidden_states)

                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            
            
            
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                
                
                
                
                if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (layer_idx >= recompute_skip_num_layers
                        and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(hidden_states, attention_mask, context,
                                                                              context_mask, rotary_pos_emb, **kwargs)

                
                if (layer_idx + layer_offset) in extract_layer_indices:
                    intermediate_hidden_states.append(hidden_states)
        else:
            raise ValueError('Invalid activation recompute method.')

        
        if len(extract_layer_indices) > 0:
            return hidden_states, intermediate_hidden_states

        return hidden_states

    def _layer_forward(self, layer, hidden_states, **kwargs):
        return layer(hidden_states=hidden_states, **kwargs)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, WrappedTensor],
        attention_mask: Optional[torch.Tensor],
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        rotary_pos_cos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        extract_layer_indices: Optional[Set[int]] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        dynamic_inference_decode_only: Optional[bool] = None,
        **kwargs,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.
            extract_layer_indices (Set[int], optional): A set of global
                layer indices (0-based across all pipeline stages) from
                which to extract intermediate hidden states. If
                non-empty, the forward pass will collect hidden_states
                after each specified layer.
            dynamic_inference_decode_only: Optional[bool]: If true, indicates that the current
                inference context is for decode-only. This args is only used to uniquely
                identify decode and non-decode cuda graph runners in the cuda graph manager.

        Returns:
            Union[Tensor, Tuple[Tensor, List[Tensor]]]:
                - If extract_layer_indices is None or empty: Returns the output hidden states tensor
                  of shape [s, b, h].
                - If extract_layer_indices is non-empty: Returns a tuple
                  of (hidden_states, intermediate_hidden_states) where
                  intermediate_hidden_states is a list of tensors
                  corresponding to hidden states after each layer in
                  extract_layer_indices.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)
        
        
        

        
        if extract_layer_indices is None:
            extract_layer_indices = set()
        intermediate_hidden_states: List[torch.Tensor] = []

        
        
        pp_group = self.pg_collection.pp if hasattr(self.pg_collection, 'pp') else None
        layer_offset = get_transformer_layer_offset(self.config, self.vp_stage, get_pg_rank(pp_group))

        
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            
            hidden_states = self.input_tensor

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        enable_hyper_connections = getattr(self.config, 'enable_hyper_connections', False)
        
        
        if enable_hyper_connections and self.pre_process:
            hidden_states = HyperConnectionModule.input_expand(hidden_states,
                                                               self.num_residual_streams)  

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        
        
        
        
        
        
        if self.config.fp8:
            use_outer_quantization_context = self.config.fp8_recipe == Fp8Recipe.delayed
            use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            outer_quantization_context = (
                get_fp8_context(self.config) if use_outer_quantization_context else nullcontext())
        elif self.config.fp4:
            use_outer_quantization_context = False
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()
        else:
            
            use_outer_quantization_context = False
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        
        
        use_mhc_recompute = (
            self.training and enable_hyper_connections and self.config.recompute_granularity == 'selective'
            and 'mhc' in self.config.recompute_modules)
        if hasattr(self, '_build_mhc_recompute_layer_plan'):
            mhc_layer_managers, mhc_is_last_in_recompute_block = self._build_mhc_recompute_layer_plan(use_mhc_recompute)
        else:
            mhc_layer_managers = [None] * len(self.layers)
            mhc_is_last_in_recompute_block = [False] * len(self.layers)
        with rng_context, outer_quantization_context:
            
            if self.config.recompute_granularity == 'full' and self.training:
                checkpointed_result = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                    padding_mask=padding_mask,
                    extract_layer_indices=extract_layer_indices,
                    layer_offset=layer_offset,
                    **kwargs,
                )
                
                if len(extract_layer_indices) > 0:
                    
                    hidden_states, intermediate_hidden_states = checkpointed_result
                else:
                    
                    hidden_states = checkpointed_result
            else:
                for l_no, layer in enumerate(self.layers):
                    
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(self.config, layer.layer_number - 1)
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(self.config, layer.layer_number - 1)
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    mhc_manager = mhc_layer_managers[l_no]
                    if mhc_manager is not None:
                        mhc_manager.is_last_layer_in_recompute_block = (mhc_is_last_in_recompute_block[l_no])
                        kwargs['mhc_recompute_manager'] = mhc_manager
                    with self.offload_context, inner_quantization_context:
                        hidden_states, context = self._layer_forward(
                            layer,
                            hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            rotary_pos_cos_sin=rotary_pos_cos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                            padding_mask=padding_mask,
                            **kwargs)
                    if mhc_manager is not None:
                        self._finalize_mhc_recompute_layer(
                            mhc_manager=mhc_manager,
                            hidden_states=hidden_states,
                            is_last_in_recompute_block=mhc_is_last_in_recompute_block[l_no],
                        )

                    if (torch.is_grad_enabled() and self.config.cpu_offloading
                            and self.group_prefetch_offload_commit_async is not None):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

                    
                    if (l_no + layer_offset) in extract_layer_indices:
                        intermediate_hidden_states.append(hidden_states)

        
        mhc_multistream = None
        if enable_hyper_connections and self.has_final_layernorm_in_this_stage():
            
            if self.config.mtp_num_layers is not None:
                assert (len(extract_layer_indices) == 0), 'Feature extraction is not supported with mHC + MTP.'
                mhc_multistream = hidden_states
            
            
            hidden_states = learned_output_contract(
                hidden_states,
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
                self.config.num_residual_streams,
                self.config.layernorm_epsilon,
            )

        
        if self.final_layernorm is not None:
            if apply_module is None:
                hidden_states = self.final_layernorm(hidden_states)
            else:
                hidden_states = apply_module(self.final_layernorm)(cast(torch.Tensor, hidden_states))
            
            
            
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        
        
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()

        if len(extract_layer_indices) > 0:
            return hidden_states, intermediate_hidden_states

        
        
        if mhc_multistream is not None:
            return hidden_states, mhc_multistream

        return hidden_states
