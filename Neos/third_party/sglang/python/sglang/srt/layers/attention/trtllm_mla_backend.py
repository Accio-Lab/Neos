from __future__ import annotations

"""
Support attention backend for TRTLLM MLA kernels from flashinfer.
"""

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton
import triton.language as tl

from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAAttnBackend,
    FlashInferMLAMultiStepDraftBackend,
)
from sglang.srt.layers.attention.utils import (
    concat_mla_absorb_q_general,
    create_flashmla_kv_indices_triton,
    get_num_page_per_block_flashmla,
    mla_quantize_and_rope_for_fp8,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_flashinfer_available, is_float4_e2m1fn_x2

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)


DEFAULT_WORKSPACE_SIZE_MB = 150  







TRTLLM_BLOCK_CONSTRAINT = 128


@triton.jit
def pad_draft_extend_query_kernel(
    q_ptr,  
    padded_q_ptr,  
    seq_lens_q_ptr,  
    cumsum_ptr,  
    batch_size,
    max_seq_len,
    num_heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for padding draft extended query tensor with parallelized head and dim processing."""
    
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    batch_id = batch_seq_pid // max_seq_len
    seq_pos = batch_seq_pid % max_seq_len

    if batch_id >= batch_size:
        return

    
    seq_len = tl.load(seq_lens_q_ptr + batch_id)

    if seq_pos >= seq_len:
        return

    
    input_start = tl.load(cumsum_ptr + batch_id)
    input_pos = input_start + seq_pos

    
    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, num_heads)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)

    
    input_offset = (
        input_pos * num_heads * head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    
    data = tl.load(
        q_ptr + input_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    
    output_offset = (
        batch_id * max_seq_len * num_heads * head_dim
        + seq_pos * num_heads * head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    
    tl.store(
        padded_q_ptr + output_offset,
        data,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def unpad_draft_extend_output_kernel(
    raw_out_ptr,  
    output_ptr,  
    accept_length_ptr,  
    cumsum_ptr,  
    batch_size,
    token_per_batch,
    tp_q_head_num,
    v_head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for unpadding draft extended output tensor with parallelized head and dim processing."""
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    batch_id = batch_seq_pid // token_per_batch
    seq_pos = batch_seq_pid % token_per_batch

    if batch_id >= batch_size:
        return

    
    accept_len = tl.load(accept_length_ptr + batch_id)

    if seq_pos >= accept_len:
        return

    
    output_start = tl.load(cumsum_ptr + batch_id)
    output_pos = output_start + seq_pos

    
    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, tp_q_head_num)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, v_head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)

    
    input_offset = (
        batch_id * token_per_batch * tp_q_head_num * v_head_dim
        + seq_pos * tp_q_head_num * v_head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    
    data = tl.load(
        raw_out_ptr + input_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    output_offset = (
        output_pos * tp_q_head_num * v_head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    
    tl.store(
        output_ptr + output_offset,
        data,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


def _quantize_fp8_qkv(q, k, v, layer):
    q = q.to(torch.float8_e4m3fn)

    k_scale = getattr(layer, "k_scale_float", None)
    if k_scale is None:
        k_scale = 1.0
    if k_scale != 1.0:
        assert hasattr(layer, "k_scale"), "k_scale is not set"
        k_2d, _ = scaled_fp8_quant(
            k.reshape(-1, k.shape[-1]).contiguous(), layer.k_scale
        )
        k = k_2d.reshape(k.shape)
    else:
        k = k.to(torch.float8_e4m3fn)

    v_scale = getattr(layer, "v_scale_float", None)
    if v_scale is None:
        v_scale = 1.0
    if v_scale != 1.0:
        assert hasattr(layer, "v_scale"), "v_scale is not set"
        v_2d, _ = scaled_fp8_quant(
            v.reshape(-1, v.shape[-1]).contiguous(), layer.v_scale
        )
        v = v_2d.reshape(v.shape)
    else:
        v = v.to(torch.float8_e4m3fn)

    return q, k, v, k_scale, v_scale


global_zero_init_workspace_buffer = None


@dataclass
class TRTLLMMLAPrefillMetadata:
    """Metadata for TRTLLM MLA prefill operations."""

    max_seq_len: int
    cum_seq_lens: torch.Tensor
    seq_lens: torch.Tensor
    fallback_to_flashinfer_impl: bool = False


@dataclass
class TRTLLMMLADecodeMetadata:
    """Metadata for TRTLLM MLA decode operations."""

    block_kv_indices: Optional[torch.Tensor] = None
    max_seq_len_k: Optional[int] = None
    max_seq_len_q: Optional[int] = None
    sum_seq_lens_q: Optional[int] = None
    cu_seqlens_q: Optional[torch.Tensor] = None
    seq_lens_q: Optional[torch.Tensor] = None
    seq_lens_k: Optional[torch.Tensor] = None


class TRTLLMMLABackend(FlashInferMLAAttnBackend):
    """TRTLLM MLA attention kernel from flashinfer."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
        )

        config = model_runner.model_config

        
        self.num_q_heads = config.num_attention_heads // get_attention_tp_size()
        self.num_kv_heads = config.get_num_kv_heads(get_attention_tp_size())
        self.num_local_heads = config.num_attention_heads // get_attention_tp_size()

        
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim

        
        self.scaling = config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        
        self.workspace_size = DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024
        global global_zero_init_workspace_buffer
        if global_zero_init_workspace_buffer is None:
            global_zero_init_workspace_buffer = torch.zeros(
                self.workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_zero_init_workspace_buffer

        
        self.decode_cuda_graph_metadata = {}
        self.decode_cuda_graph_kv_indices = None
        self.padded_q_buffer = None
        self.unpad_output_buffer = None
        self.forward_prefill_metadata: Optional[TRTLLMMLAPrefillMetadata] = None
        self.forward_decode_metadata: Union[TRTLLMMLADecodeMetadata, None] = None

        self.disable_chunked_prefix_cache = (
            get_global_server_args().disable_chunked_prefix_cache
        )

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

    def _calc_padded_blocks(self, max_seq_len: int) -> int:
        """
        Calculate padded block count that satisfies both TRT-LLM and Triton constraints.

        Args:
            max_seq_len: Maximum sequence length in tokens

        Returns:
            Number of blocks padded to satisfy all constraints
        """
        blocks = triton.cdiv(max_seq_len, self.page_size)

        
        
        
        trtllm_constraint = TRTLLM_BLOCK_CONSTRAINT // self.page_size
        triton_constraint = get_num_page_per_block_flashmla(self.page_size)
        constraint_lcm = math.lcm(trtllm_constraint, triton_constraint)

        if blocks % constraint_lcm != 0:
            blocks = triton.cdiv(blocks, constraint_lcm) * constraint_lcm
        return blocks

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create block KV indices tensor using Triton kernel.

        Args:
            batch_size: Batch size
            max_blocks: Maximum number of blocks per sequence
            req_pool_indices: Request pool indices
            seq_lens: Sequence lengths
            device: Target device

        Returns:
            Block KV indices tensor
        """
        block_kv_indices = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )

        create_flashmla_kv_indices_triton[(batch_size,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.stride(0),
            max_blocks,
            PAGED_SIZE=self.page_size,
        )

        return block_kv_indices

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """Initialize CUDA graph state for TRTLLM MLA."""

        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)

        self.decode_cuda_graph_kv_indices = torch.full(
            (max_bs, max_blocks_per_seq), -1, dtype=torch.int32, device=self.device
        )
        num_tokens_per_bs = max_num_tokens // max_bs

        if is_float4_e2m1fn_x2(self.data_type):
            
            self.store_dtype = torch.uint8
            self.padded_q_buffer = torch.zeros(
                (max_bs, num_tokens_per_bs // 2, self.num_q_heads, self.kv_cache_dim),
                dtype=self.store_dtype,
                device=self.device,
            )

            
            self.unpad_output_buffer = torch.zeros(
                (max_num_tokens // 2, self.num_q_heads, 512),
                dtype=self.store_dtype,
                device=self.device,
            )
        else:
            
            self.padded_q_buffer = torch.zeros(
                (max_bs, num_tokens_per_bs, self.num_q_heads, self.kv_cache_dim),
                dtype=self.data_type,
                device=self.device,
            )

            
            self.unpad_output_buffer = torch.zeros(
                (max_num_tokens, self.num_q_heads, 512),
                dtype=self.data_type,
                device=self.device,
            )

        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Initialize metadata for CUDA graph capture."""

        
        if (
            not forward_mode.is_decode_or_idle()
            and not forward_mode.is_target_verify()
            and not forward_mode.is_draft_extend(include_v2=True)
        ):
            return super().init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

        metadata = TRTLLMMLADecodeMetadata()

        if forward_mode.is_target_verify():
            seq_lens = seq_lens + self.num_draft_tokens
            metadata.seq_lens_k = torch.zeros(
                (bs,), dtype=torch.int32, device=seq_lens.device
            )
            metadata.seq_lens_k.copy_(seq_lens.to(dtype=torch.int32))
        elif forward_mode.is_draft_extend(include_v2=True):
            num_tokens_per_bs = num_tokens // bs
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.sum_seq_lens_q = num_tokens_per_bs * bs
            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                num_tokens_per_bs,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            metadata.seq_lens_q = torch.full(
                (bs,), num_tokens_per_bs, dtype=torch.int32, device=seq_lens.device
            )
            
            
            
            
            
            seq_lens = seq_lens - metadata.seq_lens_q + metadata.max_seq_len_q
            metadata.seq_lens_k = torch.zeros(
                (bs,), dtype=torch.int32, device=seq_lens.device
            )
            metadata.seq_lens_k.copy_(seq_lens.to(dtype=torch.int32))

        
        
        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)
        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_blocks_per_seq]

        create_flashmla_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.stride(0),
            max_blocks_per_seq,
            PAGED_SIZE=self.page_size,
        )

        metadata.block_kv_indices = block_kv_indices
        metadata.max_seq_len_k = self.max_context_len

        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Replay CUDA graph with new inputs."""
        
        if (
            not forward_mode.is_decode_or_idle()
            and not forward_mode.is_target_verify()
            and not forward_mode.is_draft_extend(include_v2=True)
        ):
            return super().init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

        metadata = self.decode_cuda_graph_metadata[bs]

        if forward_mode.is_target_verify():
            seq_lens = seq_lens[:bs] + self.num_draft_tokens
            metadata.seq_lens_k.copy_(seq_lens.to(dtype=torch.int32))
            del seq_lens_sum  
        elif forward_mode.is_draft_extend(include_v2=True):
            accept_length = spec_info.accept_length[:bs]
            if spec_info.accept_length_cpu:
                metadata.max_seq_len_q = max(spec_info.accept_length_cpu[:bs]) + 1
                metadata.sum_seq_lens_q = sum(spec_info.accept_length_cpu[:bs]) + bs
            else:
                metadata.max_seq_len_q = 1
                metadata.sum_seq_lens_q = bs
            
            extend_seq_lens = accept_length + 1
            metadata.cu_seqlens_q[1:].copy_(
                torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32)
            )
            metadata.seq_lens_q.copy_(extend_seq_lens)
            
            seq_lens = seq_lens[:bs] - metadata.seq_lens_q + metadata.max_seq_len_q
            metadata.seq_lens_k.copy_(seq_lens.to(torch.int32))

        
        create_flashmla_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices[:bs],
            seq_lens,
            None,
            metadata.block_kv_indices,
            self.req_to_token.stride(0),
            metadata.block_kv_indices.shape[1],
            PAGED_SIZE=self.page_size,
        )

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Get the fill value for sequence lengths in CUDA graph."""
        return 1

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize the metadata for a forward pass."""
        
        if (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            
            
            
            has_prefix = any(forward_batch.extend_prefix_lens_cpu)
            fallback_to_flashinfer_impl = (
                self.disable_chunked_prefix_cache and has_prefix
            ) or is_in_piecewise_cuda_graph()
            if fallback_to_flashinfer_impl:
                super().init_forward_metadata(forward_batch)

            seq_lens = forward_batch.seq_lens - forward_batch.extend_prefix_lens
            cum_seq_lens_q = torch.cat(
                (
                    torch.zeros(
                        1, dtype=torch.int32, device=forward_batch.seq_lens.device
                    ),
                    torch.cumsum(seq_lens, dim=0),
                )
            ).int()
            max_seq_len = max(forward_batch.extend_seq_lens_cpu)
            self.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(
                max_seq_len,
                cum_seq_lens_q,
                seq_lens,
                fallback_to_flashinfer_impl,
            )
        elif (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            bs = forward_batch.batch_size
            self.forward_decode_metadata = TRTLLMMLADecodeMetadata()
            
            
            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                self.forward_prefill_metadata = None
            
            if getattr(forward_batch, "seq_lens_cpu", None) is not None:
                max_seq = forward_batch.seq_lens_cpu.max().item()
            else:
                max_seq = forward_batch.seq_lens.max().item()

            seq_lens = forward_batch.seq_lens

            if forward_batch.forward_mode.is_target_verify():
                max_seq = max_seq + self.num_draft_tokens
                seq_lens = seq_lens + self.num_draft_tokens
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)
            elif forward_batch.forward_mode.is_draft_extend(include_v2=True):
                sum_seq_lens_q = sum(forward_batch.extend_seq_lens_cpu)
                max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(
                        forward_batch.extend_seq_lens, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                
                seq_lens = seq_lens - forward_batch.extend_seq_lens + max_seq_len_q

                self.forward_decode_metadata.max_seq_len_q = max_seq_len_q
                self.forward_decode_metadata.sum_seq_lens_q = sum_seq_lens_q
                self.forward_decode_metadata.cu_seqlens_q = cu_seqlens_q
                self.forward_decode_metadata.seq_lens_q = forward_batch.extend_seq_lens
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)

            max_seqlen_pad = self._calc_padded_blocks(max_seq)
            block_kv_indices = self._create_block_kv_indices(
                bs,
                max_seqlen_pad,
                forward_batch.req_pool_indices,
                seq_lens,
                seq_lens.device,
            )

            self.forward_decode_metadata.block_kv_indices = block_kv_indices
            self.forward_decode_metadata.max_seq_len_k = int(max_seq)
            self.forward_decode_metadata.batch_size = bs

            forward_batch.decode_trtllm_mla_metadata = self.forward_decode_metadata
        else:
            return super().init_forward_metadata(forward_batch)

    def init_mha_chunk_metadata(self, forward_batch: ForwardBatch):
        super().init_mha_chunk_metadata(forward_batch, disable_flashinfer_ragged=True)

    def pad_draft_extend_query(
        self,
        q: torch.Tensor,
        padded_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        """Pad draft extended query using Triton kernel."""
        batch_size = cu_seqlens_q.shape[0] - 1
        max_seq_len_q = padded_q.shape[1]
        num_heads = padded_q.shape[2]
        head_dim = padded_q.shape[3]

        
        BLOCK_SIZE = 64
        num_head_blocks = triton.cdiv(num_heads, BLOCK_SIZE)
        num_dim_blocks = triton.cdiv(head_dim, BLOCK_SIZE)
        grid = (batch_size * max_seq_len_q, num_head_blocks, num_dim_blocks)

        pad_draft_extend_query_kernel[grid](
            q_ptr=q,
            padded_q_ptr=padded_q,
            seq_lens_q_ptr=seq_lens_q,
            cumsum_ptr=cu_seqlens_q,
            batch_size=batch_size,
            max_seq_len=max_seq_len_q,
            num_heads=num_heads,
            head_dim=head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return padded_q

    def unpad_draft_extend_output(
        self,
        raw_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        sum_seq_lens_q: int,
    ) -> torch.Tensor:
        """Unpad draft extended output using Triton kernel."""
        
        batch_size = seq_lens_q.shape[0]
        token_per_batch = raw_out.shape[1]  
        tp_q_head_num = raw_out.shape[2]  
        v_head_dim = raw_out.shape[3]  
        total_tokens = sum_seq_lens_q

        
        if self.unpad_output_buffer is not None:
            
            output = self.unpad_output_buffer[:total_tokens, :, :].to(
                dtype=raw_out.dtype
            )
        else:
            
            output = torch.empty(
                (total_tokens, tp_q_head_num, v_head_dim),
                dtype=raw_out.dtype,
                device=raw_out.device,
            )

        
        BLOCK_SIZE = 64
        num_head_blocks = triton.cdiv(tp_q_head_num, BLOCK_SIZE)
        num_dim_blocks = triton.cdiv(v_head_dim, BLOCK_SIZE)
        grid = (batch_size * token_per_batch, num_head_blocks, num_dim_blocks)

        unpad_draft_extend_output_kernel[grid](
            raw_out_ptr=raw_out,
            output_ptr=output,
            accept_length_ptr=seq_lens_q,
            cumsum_ptr=cu_seqlens_q,
            batch_size=batch_size,
            token_per_batch=token_per_batch,
            tp_q_head_num=tp_q_head_num,
            v_head_dim=v_head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output[:total_tokens, :, :]

    def forward_decode(
        self,
        q: torch.Tensor,  
        k: torch.Tensor,  
        v: torch.Tensor,  
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run forward for decode using TRTLLM MLA kernel."""
        merge_query = q_rope is not None
        if self.data_type == torch.float8_e4m3fn:
            
            
            assert all(
                x is not None for x in [q_rope, k_rope, cos_sin_cache]
            ), "For FP8 path and using flashinfer.rope.mla_rope_quantize we need all of q_rope, k_rope and cos_sin_cache to be not None."
            q, k, k_rope = mla_quantize_and_rope_for_fp8(
                q,
                q_rope,
                k.squeeze(1),
                k_rope.squeeze(1),
                forward_batch.positions,
                cos_sin_cache,
                is_neox,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
            merge_query = False

        
        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        
        if merge_query:
            
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            query = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
        else:
            
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        
        if llama_4_scaling is not None:
            query = query.to(self.q_data_type) * llama_4_scaling
            query = query.to(self.data_type)

        
        if query.dim() == 3:
            query = query.unsqueeze(1)

        
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

        
        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_decode_metadata
        )

        
        
        
        batch_size = getattr(metadata, "batch_size", None)
        if batch_size is not None and batch_size < forward_batch.batch_size:
            self.init_forward_metadata(forward_batch)
            metadata = forward_batch.decode_trtllm_mla_metadata

        
        
        
        
        
        
        
        
        q_scale = 1.0
        if self.data_type == torch.float8_e4m3fn:
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
        else:
            if getattr(layer, "k_scale_float", None) is not None:
                logger.warning_once(
                    "Checkpoint has k_scale but KV cache dtype is not FP8. "
                    "Ignoring k_scale for BMM1 (k_scale=%.4f, kv_dtype=%s).",
                    layer.k_scale_float,
                    self.data_type,
                )
            k_scale = 1.0

        bmm1_scale = q_scale * k_scale * layer.scaling

        
        raw_out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=metadata.block_kv_indices,
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            max_seq_len=metadata.max_seq_len_k,
            bmm1_scale=bmm1_scale,
        )

        
        output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        return output

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if (
            self.forward_prefill_metadata is not None
            and self.forward_prefill_metadata.fallback_to_flashinfer_impl
        ):
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, q_rope, k_rope
            )

        
        merge_query = q_rope is not None
        if (
            self.data_type == torch.float8_e4m3fn
        ) and forward_batch.forward_mode.is_target_verify():
            
            
            assert all(
                x is not None for x in [q_rope, k_rope, cos_sin_cache]
            ), "For FP8 path and using flashinfer.rope.mla_rope_quantize we need all of q_rope, k_rope and cos_sin_cache to be not None."
            q, k, k_rope = mla_quantize_and_rope_for_fp8(
                q,
                q_rope,
                k.squeeze(1),
                k_rope.squeeze(1),
                forward_batch.positions,
                cos_sin_cache,
                is_neox,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
            merge_query = False

        
        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        
        
        if merge_query:
            
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            q = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)

        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        
        if llama_4_scaling is not None:
            q = q.to(self.q_data_type) * llama_4_scaling
            q = q.to(self.data_type)

        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            metadata = (
                getattr(forward_batch, "decode_trtllm_mla_metadata", None)
                or self.forward_decode_metadata
            )

            
            
            
            batch_size = getattr(metadata, "batch_size", None)
            if batch_size is not None and batch_size < forward_batch.batch_size:
                self.init_forward_metadata(forward_batch)
                metadata = forward_batch.decode_trtllm_mla_metadata

            
            bs = forward_batch.batch_size

            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

            q_scale = 1.0
            if self.data_type == torch.float8_e4m3fn:
                k_scale = (
                    layer.k_scale_float
                    if getattr(layer, "k_scale_float", None) is not None
                    else 1.0
                )
            else:
                if getattr(layer, "k_scale_float", None) is not None:
                    logger.warning_once(
                        "Checkpoint has k_scale but KV cache dtype is not FP8. "
                        "Ignoring k_scale for BMM1 (k_scale=%.4f, kv_dtype=%s).",
                        layer.k_scale_float,
                        self.data_type,
                    )
                k_scale = 1.0
            q = q.to(self.data_type)

            bmm1_scale = q_scale * k_scale * layer.scaling
            if forward_batch.forward_mode.is_target_verify():
                max_seq_len = (
                    metadata.max_seq_len_k + forward_batch.spec_info.draft_token_num
                )
                
                q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
                needs_unpad = False
            else:
                
                
                total_tokens = q.shape[0]
                tokens_per_seq = total_tokens // bs if bs > 0 else 0
                can_direct_view = bs > 0 and (total_tokens % bs == 0)

                if can_direct_view:
                    max_seq_len = metadata.max_seq_len_k + tokens_per_seq
                    q = q.view(bs, tokens_per_seq, layer.tp_q_head_num, layer.head_dim)
                    needs_unpad = False
                else:
                    
                    actual_seq_lens_q = forward_batch.extend_seq_lens
                    actual_max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                    max_seq_len = metadata.max_seq_len_k + actual_max_seq_len_q

                    actual_cu_seqlens_q = torch.nn.functional.pad(
                        torch.cumsum(actual_seq_lens_q, dim=0, dtype=torch.int32),
                        (1, 0),
                    )

                    if self.padded_q_buffer is not None:
                        padded_q = self.padded_q_buffer[
                            :bs, :actual_max_seq_len_q, :, :
                        ].to(dtype=q.dtype)
                        padded_q.zero_()
                    else:
                        padded_q = torch.zeros(
                            (
                                bs,
                                actual_max_seq_len_q,
                                layer.tp_q_head_num,
                                layer.head_dim,
                            ),
                            dtype=q.dtype,
                            device=q.device,
                        )

                    q = self.pad_draft_extend_query(
                        q, padded_q, actual_seq_lens_q, actual_cu_seqlens_q
                    )
                    needs_unpad = True
                    unpad_seq_lens_q = actual_seq_lens_q
                    unpad_cu_seqlens_q = actual_cu_seqlens_q
                    unpad_sum_seq_lens_q = total_tokens

            assert kv_cache.dtype == self.data_type

            raw_out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                block_tables=metadata.block_kv_indices,
                seq_lens=metadata.seq_lens_k,
                max_seq_len=max_seq_len,
                bmm1_scale=bmm1_scale,
            )

            if needs_unpad:
                
                
                output = self.unpad_draft_extend_output(
                    raw_out,
                    unpad_cu_seqlens_q,
                    unpad_seq_lens_q,
                    unpad_sum_seq_lens_q,
                )
                output = output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            else:
                output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            return output

        if k_rope is not None:
            k = torch.cat([k, k_rope], dim=-1)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)

        q_scale = k_scale = v_scale = 1.0
        if self.data_type == torch.float8_e4m3fn:
            q, k, v, k_scale, v_scale = _quantize_fp8_qkv(q, k, v, layer)

        common_trtllm_args = {
            "query": q,
            "key": k,
            "value": v,
            "workspace_buffer": self.workspace_buffer,
            "batch_size": forward_batch.batch_size,
            "window_left": -1,
            "enable_pdl": False,
            "max_q_len": self.forward_prefill_metadata.max_seq_len,
            "bmm1_scale": q_scale * k_scale * layer.scaling,
            "bmm2_scale": v_scale,
            "cum_seq_lens_q": self.forward_prefill_metadata.cum_seq_lens,
        }

        
        if forward_batch.attn_attend_prefix_cache:
            
            assert forward_batch.prefix_chunk_idx is not None
            assert forward_batch.prefix_chunk_cu_seq_lens is not None
            assert q_rope is None
            assert k_rope is None
            chunk_idx = forward_batch.prefix_chunk_idx

            out = torch.zeros(
                q.shape[0],
                layer.tp_q_head_num,
                layer.v_head_dim,
                dtype=self.q_data_type,
                device=q.device,
            )
            return flashinfer.prefill.trtllm_ragged_attention_deepseek(
                **common_trtllm_args,
                seq_lens=forward_batch.prefix_chunk_seq_lens[chunk_idx],
                max_kv_len=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                o_sf_scale=-1.0,
                cum_seq_lens_kv=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                is_causal=False,
                return_lse=True,
                out=out,
            )
        else:
            out = torch.zeros(
                q.shape[0],
                q.shape[1],
                v.shape[2],
                device=q.device,
                dtype=self.q_data_type,
            )
            return flashinfer.prefill.trtllm_ragged_attention_deepseek(
                **common_trtllm_args,
                seq_lens=self.forward_prefill_metadata.seq_lens,
                max_kv_len=self.forward_prefill_metadata.max_seq_len,
                o_sf_scale=1.0,
                cum_seq_lens_kv=self.forward_prefill_metadata.cum_seq_lens,
                is_causal=True,
                return_lse=forward_batch.mha_return_lse,
                out=out,
            )


class TRTLLMMLAMultiStepDraftBackend(FlashInferMLAMultiStepDraftBackend):
    """Multi-step draft backend for TRT-LLM MLA used by EAGLE."""

    def __init__(
        self, model_runner: "ModelRunner", topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TRTLLMMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
            )
