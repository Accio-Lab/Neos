# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

# there is some bug in mcore 0.12, so we need to patch it
# 1. `get_query_key_value_tensors` in `multi_latent_attention.py` works wrong when packed_seq_params is not None


def apply_patch():
    import megatron.core
    import torch
    import torch.nn.functional as F
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.transformer.multi_latent_attention import (
        MLASelfAttention,
        MultiLatentAttention,
        apply_rotary_pos_emb,
        deprecate_inference_params,
        gather_from_sequence_parallel_region,
        gather_from_tensor_model_parallel_region,
        scatter_to_sequence_parallel_region,
    )
    from packaging import version

    mcore_ge_013 = version.parse(megatron.core.__version__) >= version.parse("0.13.0")
    mcore_ge_0162 = version.parse(megatron.core.__version__) >= version.parse("0.16.2")

    def patch_get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        
        
        assert hidden_states.ndim == 3, f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"

        inference_context = deprecate_inference_params(inference_context, inference_params)

        
        
        
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        
        mscale = 1.0
        if self.config.rope_type == "rope":
            packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
            try:
                
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
            except TypeError:
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len)

        
        
        
        if self.config.q_lora_rank is not None:
            
            
            
            
            q_compressed, _ = self.linear_q_down_proj(hidden_states)

            
            
            
            
            
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)

            q_compressed = self.q_layernorm(q_compressed)
        else:
            q_compressed = hidden_states

        
        
        
        
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if self.config.sequence_parallel:
                
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)

        kv_compressed = self.kv_layernorm(kv_compressed)

        
        
        
        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            if self.config.q_lora_rank is not None:
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                
                q, _ = self.linear_q_proj(q_compressed)

            q_len, bsz, _ = q.size()

            
            q = q.view(q_len, bsz, self.num_attention_heads_per_partition, self.q_head_dim)

            
            kv, _ = self.linear_kv_up_proj(kv_compressed)

            
            kv = kv.view(
                q_len,
                bsz,
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
            )

            cp_size = parallel_state.get_context_parallel_world_size()
            if inference_context is not None:
                
                sequence_start = inference_context.sequence_len_offset
                sequence_end = sequence_start + q_len
                rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
            elif packed_seq_params is None or cp_size == 1:
                
                
                
                
                
                
                
                
                rotary_pos_emb = rotary_pos_emb[0:q_len]

            
            k_pos_emb = torch.unsqueeze(k_pos_emb, 2)

            
            q_no_pe, q_pos_emb = torch.split(q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1)

            
            k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
                q_pos_emb = q_pos_emb.squeeze(1)
                k_pos_emb = k_pos_emb.squeeze(1)
                q_no_pe = q_no_pe.squeeze(1)
                k_no_pe = k_no_pe.squeeze(1)
                value = value.squeeze(1)
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                mscale=mscale,
            )
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=mscale,
            )

            
            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)
            if packed_seq_params is not None:
                k_pos_emb = k_pos_emb.expand(-1, self.num_attention_heads_per_partition, -1)
                key = torch.cat([k_no_pe, k_pos_emb], dim=-1)
            else:
                
                k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)
                key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            return query, key, value

        if self.recompute_up_proj:
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            query, key, value = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )
        else:
            query, key, value = qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb)

        return query, key, value

    def patch_forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ):
        """Forward pass for multi-latent attention"""
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert rotary_pos_cos is None and rotary_pos_sin is None, "MLA does not support Flash Decoding"

        

        inference_context = deprecate_inference_params(inference_context, inference_params)

        
        
        
        
        
        
        qkv = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )
        query, key, value = qkv[:3]
        q_compressed = None
        
        if len(qkv) > 4:
            q_compressed = qkv[3]
            

        
        
        
        
        if mcore_ge_013:
            query, key, value, _, attn_mask_type, _ = self._adjust_key_value_for_inference(
                inference_context, query, key, value, rotary_pos_emb=None
            )
        else:
            query, key, value, _, attn_mask_type = self._adjust_key_value_for_inference(
                inference_context, query, key, value, rotary_pos_emb=None
            )

        
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        
        
        
        
        orig_v_dim = value.shape[-1] if value is not None else None
        thd_packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        need_v_pad = (
            thd_packed_seq
            and getattr(self.config, "experimental_attention_variant", None) is None
            and value is not None
            and query.shape[-1] != orig_v_dim
        )
        if need_v_pad:
            
            value = F.pad(value, [0, query.shape[-1] - orig_v_dim])
            self.core_attention.hidden_size_per_attention_head_v = value.shape[-1]
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            extra_kwargs = {}
            if getattr(self.config, "experimental_attention_variant", None) == "dsa":
                
                
                extra_kwargs["x"] = hidden_states
                extra_kwargs["qr"] = q_compressed
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=attn_mask_type,
                **extra_kwargs,
            )
        if thd_packed_seq:
            if need_v_pad:
                if core_attn_out.ndim == 2:
                    core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-1], -1, value.shape[-1])
                core_attn_out = core_attn_out[..., :orig_v_dim]
            
            
            
            
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        
        
        
        output, bias = self.linear_proj(core_attn_out)

        return output, bias

    
    
    
    if not mcore_ge_013:
        MLASelfAttention.get_query_key_value_tensors = patch_get_query_key_value_tensors

    if not mcore_ge_0162:
        MultiLatentAttention.forward = patch_forward


def apply_patch_mbridge():
    try:
        from megatron.core.utils import get_tensor_model_parallel_group_if_none
    except ImportError:
        import warnings

        import megatron.core.utils
        import torch
        from megatron.core import parallel_state

        def get_tensor_model_parallel_group_if_none(tp_group, is_expert=False, check_initialized=True):
            """Issue a deprecation warning if tp_group is None and return the default tp group."""
            if not torch.distributed.is_initialized():
                return None
            if tp_group is None:
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    warnings.warn(
                        "Warning: tp_group is None, using default tp group. Passing tp_group will be mandatory soon",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                if is_expert:
                    tp_group = parallel_state.get_expert_tensor_parallel_group(check_initialized=check_initialized)
                else:
                    tp_group = parallel_state.get_tensor_model_parallel_group(check_initialized=check_initialized)
            return tp_group

        megatron.core.utils.get_tensor_model_parallel_group_if_none = get_tensor_model_parallel_group_if_none


def apply_patch_megatron_v012_with_torch_v28_v29() -> None:
    
    
    import inspect
    import logging
    import os
    from pathlib import Path

    import megatron.core
    import torch
    from megatron.core.dist_checkpointing.strategies.async_utils import _disable_gc
    from megatron.core.dist_checkpointing.strategies.filesystem_async import _process_memory
    from packaging import version
    from torch import multiprocessing as mp
    from torch.distributed.checkpoint.filesystem import _write_item

    if (
        version.parse(torch.__version__).base_version not in ("2.8.0", "2.9.0")
        or version.parse(megatron.core.__version__).base_version != "0.12.1"
    ):
        return

    WriteBucket = tuple[Path, str, tuple[list, list]]

    @staticmethod
    @_disable_gc()
    def write_preloaded_data_patch(
        transform_list,
        local_proc_idx: int,
        write_bucket: WriteBucket,
        results_queue: mp.SimpleQueue,
        count_queue: mp.JoinableQueue,
        use_fsync: bool,
        **kwargs,
    ) -> None:
        """
        Performs actual data saving to storage.

        Args:
            local_proc_idx (int): index of a local process that performs writing
            write_bucket (WriteBucket): data to write to storage
            results_queue (mp.Queue): queue to return the write results
                to the proxy checkpoint process.
            count_queue (mp.JoinableQueue): queue to marks worker task as completed
            use_fsync (bool): if True, calls os.fsync at the end of saving

        Returns: None, the write result are put into the `queue`
        """
        logger = logging.getLogger(__name__)
        logger.debug(f"{local_proc_idx} started")
        mem_before = _process_memory()
        use_msc = kwargs.get("use_msc", False)
        local_results = []
        try:
            file_name, storage_key, (bytes_data, tensor_data) = write_bucket
            extra_kwargs = {}
            if "serialization_format" in inspect.signature(_write_item).parameters:
                from torch.distributed.checkpoint.filesystem import SerializationFormat

                extra_kwargs["serialization_format"] = SerializationFormat.TORCH_SAVE
            if use_msc:
                import multistorageclient as msc

                open_file = msc.open
            else:
                open_file = open
            with open_file(file_name, "wb") as stream:
                for write_item, data in bytes_data:
                    local_results.append(
                        _write_item(*transform_list, stream, data, write_item, storage_key, **extra_kwargs)
                    )

                for write_item, tensor in tensor_data:
                    assert tensor.is_cpu
                    local_results.append(
                        _write_item(*transform_list, stream, tensor, write_item, storage_key, **extra_kwargs)
                    )

                if use_fsync:
                    if use_msc:
                        stream.fsync()
                    else:
                        os.fsync(stream.fileno())
            local_output = (local_proc_idx, local_results)
        except Exception as e:
            logger.debug(f"{local_proc_idx} failed")
            local_output = (local_proc_idx, e)  # type: ignore[assignment]

        results_queue.put(local_output)
        
        count_queue.get()
        count_queue.task_done()

        mem_after = _process_memory()
        logger.debug(f"{local_proc_idx} consumed: {mem_after - mem_before}, before: {mem_before}, after: {mem_after}")

    from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

    FileSystemWriterAsync.write_preloaded_data = write_preloaded_data_patch


def apply_mtp_inference_patch():
    from megatron.core.models.gpt.gpt_model import GPTModel

    _original_postprocess = GPTModel._postprocess

    def _patched(self, *args, **kwargs):
        original_mtp_num_layers = self.config.mtp_num_layers
        if not self.config.mtp_num_layers:
            self.config.mtp_num_layers = None
        try:
            return _original_postprocess(self, *args, **kwargs)
        finally:
            self.config.mtp_num_layers = original_mtp_num_layers

    GPTModel._postprocess = _patched





def apply_patch_megatron_recomputation_backward():
    import megatron.core.tensor_parallel.random as rd
    import torch

    _fork_rng = rd._fork_rng
    _set_all_rng_states = rd._set_all_rng_states
    detach_variable = rd.detach_variable
    gather_split_1d_tensor = rd.gather_split_1d_tensor
    safely_set_viewless_tensor_data = rd.safely_set_viewless_tensor_data

    @staticmethod
    def patch_backward(ctx, *args):
        """Backward pass."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")
        inputs = ctx.saved_tensors
        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(inputs[0], gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape))

        with _fork_rng():
            
            _set_all_rng_states(*ctx.rng_states)

            
            detached_inputs = detach_variable(inputs)

            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        
        outputs, args = zip(
            *filter(lambda x: torch.is_tensor(x[0]) and x[0].requires_grad, zip(outputs, args, strict=False)),
            strict=False,
        )
        torch.autograd.backward(outputs, args)
        
        grads = tuple(
            inp.grad.clone()
            if isinstance(inp, torch.Tensor) and inp.grad is not None
            else inp.grad
            if isinstance(inp, torch.Tensor)
            else inp
            for inp in detached_inputs
        )
        cur_stream = torch.cuda.current_stream()
        
        for t in detached_inputs:
            if isinstance(t, torch.Tensor) and t.requires_grad:
                t.record_stream(cur_stream)
                t.untyped_storage().resize_(0)
                if t.grad is not None:
                    t.grad.record_stream(cur_stream)
                    t.grad.untyped_storage().resize_(0)
        
        return (None, None) + grads

    rd.CheckpointFunction.backward = patch_backward
