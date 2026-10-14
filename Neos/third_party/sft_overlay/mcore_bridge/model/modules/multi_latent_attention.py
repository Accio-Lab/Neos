import torch
import torch.nn.functional as F
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.tensor_parallel.mappings import (gather_from_sequence_parallel_region,
                                                    gather_from_tensor_model_parallel_region,
                                                    scatter_to_sequence_parallel_region)
from megatron.core.transformer.multi_latent_attention import MLASelfAttention as McoreMLASelfAttention
from megatron.core.utils import deprecate_inference_params


class MLASelfAttention(McoreMLASelfAttention):

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        rotary_pos_emb=None,
        *,
        inference_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        
        
        assert (hidden_states.ndim == 3), f'hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D'

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        
        
        
        if self.config.q_lora_rank is not None:
            
            
            
            
            q_compressed, _ = self.linear_q_down_proj(hidden_states)

            
            
            
            
            
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)
        else:
            q_compressed = hidden_states

        
        
        
        
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1)
            if self.config.sequence_parallel:
                
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1)
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)

        if self.config.q_lora_rank is not None:
            
            q_compressed = self.q_layernorm(q_compressed)

        kv_compressed = self.kv_layernorm(kv_compressed)

        if packed_seq_params is not None:
            
            
            
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
            k_pos_emb = k_pos_emb.squeeze(1)

        
        
        
        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [s, b, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [s, b, ...] or [t, ...] for two cases.
            """
            if self.config.q_lora_rank is not None:
                
                
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                
                
                q, _ = self.linear_q_proj(q_compressed)

            
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

            
            kv, _ = self.linear_kv_up_proj(kv_compressed)

            
            kv = kv.view(
                *kv.size()[:-1],
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
            )

            q_len = q.size()[0]
            if inference_context is not None:
                
                sequence_start = inference_context.sequence_len_offset
                sequence_end = sequence_start + q_len
                rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
            

            
            k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            
            
            q_no_pe, q_pos_emb = torch.split(q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1)

            
            
            k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)
            
            
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
            )
            
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
            )

            
            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

            
            if k_pos_emb.ndim == 4:
                k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)
            else:
                assert k_pos_emb.ndim == 3
                k_pos_emb = k_pos_emb.expand(-1, self.num_attention_heads_per_partition, -1)
            key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            return query, key, value

        if self.recompute_up_proj:
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            query, key, value = self.qkv_up_checkpoint.checkpoint(qkv_up_proj_and_rope_apply, q_compressed,
                                                                  kv_compressed, k_pos_emb, rotary_pos_emb)
        else:
            query, key, value = qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb)

        return query, key, value, q_compressed, kv_compressed

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """Forward pass for multi-latent attention"""
        assert attention_bias is None, 'Attention bias should not be passed into MLA.'
        assert (rotary_pos_cos is None and rotary_pos_sin is None), 'MLA does not support Flash Decoding'

        

        inference_context = deprecate_inference_params(inference_context, inference_params)

        
        
        
        
        
        
        query, key, value, q_compressed, kv_compressed = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            rotary_pos_emb=rotary_pos_emb,
            inference_context=inference_context,
        )

        
        
        
        
        query, key, value, _, attn_mask_type, _ = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None)

        
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        
        
        
        
        thd_qkv_format = packed_seq_params and packed_seq_params.qkv_format == 'thd'
        v_dim = value.shape[-1]
        if thd_qkv_format and query.shape[-1] != v_dim:
            value = F.pad(value, [0, query.shape[-1] - v_dim])
            self.core_attention.hidden_size_per_attention_head_v = value.shape[-1]
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params)
        else:
            extra_kwargs = {}
            if self.config.experimental_attention_variant == 'dsa':
                if packed_seq_params is not None or self.config.context_parallel_size > 1:
                    raise ImportError('Please install the megatron-core main branch to support `DSAttention` '
                                      'padding_free/context parallelism: '
                                      '`pip install git+https://github.com/NVIDIA/Megatron-LM.git`')
                
                
                extra_kwargs['x'] = hidden_states
                extra_kwargs['qr'] = q_compressed
                
                packed_seq_params = (packed_seq_params, rotary_pos_emb)
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=attn_mask_type,
                **extra_kwargs,
            )
        if thd_qkv_format:
            if core_attn_out.ndim == 2:
                core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-1], -1, value.shape[-1])
            if query.shape[-1] != v_dim:
                core_attn_out = core_attn_out[..., :v_dim]
            
            
            
            
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        
        
        
        output, bias = self.linear_proj(core_attn_out)

        return output, bias
