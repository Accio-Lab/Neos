# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Logits processing."""

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    attn_tp_all_gather,
    attn_tp_all_gather_into_tensor,
    dp_gather_replicate,
    dp_scatter,
    get_attention_dp_rank,
    get_attention_dp_size,
    get_attention_tp_size,
    get_dp_device,
    get_dp_dtype,
    get_dp_hidden_size,
)
from sglang.srt.layers.utils.logprob import (
    InputLogprobsResult,
    compute_temp_top_p_normalized_logprobs,
    get_token_ids_logprobs_chunk,
    get_token_ids_logprobs_prefill,
    get_top_logprobs_chunk,
    get_top_logprobs_prefill,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.common import is_npu, use_intel_amx_backend

logger = logging.getLogger(__name__)

_is_npu = is_npu()


@dataclasses.dataclass
class LogitsProcessorOutput:
    
    
    
    next_token_logits: Optional[torch.Tensor]
    
    
    hidden_states: Optional[torch.Tensor] = None

    
    
    next_token_logprobs: Optional[torch.Tensor] = None
    
    next_token_top_logprobs_val: Optional[List] = None
    next_token_top_logprobs_idx: Optional[List] = None
    
    
    next_token_token_ids_logprobs_val: Optional[
        List[Union[List[float], torch.Tensor]]
    ] = None
    next_token_token_ids_logprobs_idx: Optional[List] = None

    
    
    input_token_logprobs: Optional[torch.Tensor] = None
    
    input_top_logprobs_val: Optional[List] = None
    input_top_logprobs_idx: Optional[List] = None
    
    
    input_token_ids_logprobs_val: Optional[List[Union[List[float], torch.Tensor]]] = (
        None
    )
    input_token_ids_logprobs_idx: Optional[List] = None

    
    full_logits: Optional[torch.Tensor] = None

    
    customized_info: Optional[Dict[str, List[Any]]] = None

    mm_input_embeds: Optional[torch.Tensor] = None


@dataclasses.dataclass
class LogitsMetadata:
    forward_mode: ForwardMode
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL
    next_token_logits_buffer: Optional[torch.Tensor] = None

    extend_return_logprob: bool = False
    extend_return_top_logprob: bool = False
    extend_token_ids_logprob: bool = False
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None
    top_logprobs_nums: Optional[List[int]] = None
    extend_input_logprob_token_ids_gpu: Optional[torch.Tensor] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    
    temp_scaled_logprobs: bool = False
    temperature: torch.Tensor = None
    top_p_normalized_logprobs: bool = False
    top_p: torch.Tensor = None

    
    
    global_num_tokens_gpu: Optional[torch.Tensor] = None
    
    dp_local_start_pos: Optional[torch.Tensor] = None
    dp_local_num_tokens: Optional[torch.Tensor] = None
    global_dp_buffer_len: Optional[int] = None
    
    global_num_tokens_for_logprob_cpu: Optional[torch.Tensor] = None
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor] = None
    
    dp_padding_mode: Optional[DpPaddingMode] = None
    
    padded_static_len: int = -1

    
    is_prefill_only: bool = False

    mm_input_embeds: Optional[torch.Tensor] = None

    @classmethod
    def from_forward_batch(cls, forward_batch: ForwardBatch):
        if (
            forward_batch.forward_mode.is_extend()
            and forward_batch.return_logprob
            and not forward_batch.forward_mode.is_target_verify()
        ):
            extend_return_top_logprob = any(
                x > 0 for x in forward_batch.top_logprobs_nums
            )
            extend_token_ids_logprob = any(
                x is not None for x in forward_batch.token_ids_logprobs
            )
            extend_return_logprob = False
            extend_logprob_pruned_lens_cpu = []
            for extend_len, start_len in zip(
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_logprob_start_lens_cpu,
            ):
                if extend_len - start_len > 0:
                    extend_return_logprob = True
                extend_logprob_pruned_lens_cpu.append(extend_len - start_len)
        else:
            extend_return_logprob = extend_return_top_logprob = (
                extend_token_ids_logprob
            ) = extend_logprob_pruned_lens_cpu = False

        return cls(
            forward_mode=forward_batch.forward_mode,
            capture_hidden_mode=forward_batch.capture_hidden_mode,
            next_token_logits_buffer=forward_batch.next_token_logits_buffer,
            extend_return_logprob=extend_return_logprob,
            extend_return_top_logprob=extend_return_top_logprob,
            extend_token_ids_logprob=extend_token_ids_logprob,
            extend_seq_lens=forward_batch.extend_seq_lens,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,
            extend_logprob_pruned_lens_cpu=extend_logprob_pruned_lens_cpu,
            top_logprobs_nums=forward_batch.top_logprobs_nums,
            token_ids_logprobs=forward_batch.token_ids_logprobs,
            extend_input_logprob_token_ids_gpu=forward_batch.extend_input_logprob_token_ids_gpu,
            padded_static_len=forward_batch.padded_static_len,
            is_prefill_only=forward_batch.is_prefill_only,
            global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,
            dp_local_start_pos=forward_batch.dp_local_start_pos,
            dp_local_num_tokens=forward_batch.dp_local_num_tokens,
            global_dp_buffer_len=forward_batch.global_dp_buffer_len,
            global_num_tokens_for_logprob_cpu=forward_batch.global_num_tokens_for_logprob_cpu,
            global_num_tokens_for_logprob_gpu=forward_batch.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.SUM_LEN,
            mm_input_embeds=forward_batch.mm_input_embeds,
        )

    def compute_dp_attention_metadata(self):

        cumtokens = torch.cumsum(self.global_num_tokens_for_logprob_gpu, dim=0)
        dp_rank = get_attention_dp_rank()
        if dp_rank == 0:
            dp_local_start_pos = torch.zeros_like(
                self.global_num_tokens_for_logprob_gpu[0]
            )
        else:
            dp_local_start_pos = cumtokens[dp_rank - 1]

        self.dp_local_start_pos = dp_local_start_pos
        self.dp_local_num_tokens = self.global_num_tokens_for_logprob_gpu[dp_rank]

        hidden_size = get_dp_hidden_size()
        dtype = get_dp_dtype()
        device = get_dp_device()

        if self.global_num_tokens_for_logprob_cpu is not None:
            
            self.global_dp_buffer_len = sum(self.global_num_tokens_for_logprob_cpu)
        else:
            self.global_dp_buffer_len = self.global_dp_buffer_len

        self.gathered_buffer = torch.empty(
            (
                self.global_dp_buffer_len,
                hidden_size,
            ),
            dtype=dtype,
            device=device,
        )


class LogitsProcessor(nn.Module):
    def __init__(
        self,
        config,
        skip_all_gather: bool = False,
        logit_scale: Optional[float] = None,
        return_full_logits: bool = False,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.logit_scale = logit_scale
        self.use_attn_tp_group = get_global_server_args().enable_dp_lm_head
        self.use_fp32_lm_head = get_global_server_args().enable_fp32_lm_head
        if self.use_attn_tp_group:
            self.attn_tp_size = get_attention_tp_size()
            self.do_tensor_parallel_all_gather = (
                not skip_all_gather and self.attn_tp_size > 1
            )
            self.do_tensor_parallel_all_gather_dp_attn = False
        else:
            self.do_tensor_parallel_all_gather = (
                not skip_all_gather and get_tensor_model_parallel_world_size() > 1
            )
            self.do_tensor_parallel_all_gather_dp_attn = (
                self.do_tensor_parallel_all_gather and get_attention_dp_size() != 1
            )
        self.final_logit_softcapping = getattr(
            self.config, "final_logit_softcapping", None
        )
        if (
            self.final_logit_softcapping is not None
            and self.final_logit_softcapping < 0
        ):
            self.final_logit_softcapping = None

        self.return_full_logits = return_full_logits
        self.multi_item_delimiter = (
            get_global_server_args().multi_item_scoring_delimiter
        )

        
        self.enable_logprobs_chunk = envs.SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK.get()
        
        self.logprobs_chunk_size = envs.SGLANG_LOGITS_PROCESSER_CHUNK_SIZE.get()

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
        aux_hidden_states: Optional[torch.Tensor] = None,
        hidden_states_before_norm: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        if isinstance(logits_metadata, ForwardBatch):
            logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)

        
        if self.multi_item_delimiter is not None and logits_metadata.is_prefill_only:
            return self.compute_logprobs_for_multi_item_scoring(
                input_ids,
                hidden_states,
                lm_head,
                logits_metadata,
                self.multi_item_delimiter,
            )

        
        if logits_metadata.forward_mode.is_dllm_extend():
            return self._get_dllm_logits(hidden_states, lm_head, logits_metadata)

        
        (
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            input_logprob_indices,
            token_to_seq_idx,
        ) = self._get_pruned_states(
            hidden_states,
            hidden_states_before_norm,
            aux_hidden_states,
            logits_metadata,
        )

        hidden_states_to_store = self._get_hidden_states_to_store(
            hidden_states,
            hidden_states_before_norm,
            aux_hidden_states,
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            logits_metadata,
        )
        del hidden_states

        if not logits_metadata.extend_return_logprob:
            
            logits = self._get_logits(pruned_states, lm_head, logits_metadata)
            sampled_logits = (
                logits[sample_indices] if sample_indices is not None else logits
            )

            
            return LogitsProcessorOutput(
                next_token_logits=sampled_logits,
                hidden_states=hidden_states_to_store,
                mm_input_embeds=logits_metadata.mm_input_embeds,
            )

        
        
        self._expand_metadata_for_logprobs(logits_metadata, pruned_states.device)

        
        
        
        
        
        should_skip_chunking = (
            not self.enable_logprobs_chunk
            or pruned_states.shape[0] <= self.logprobs_chunk_size
            or self.do_tensor_parallel_all_gather_dp_attn
        )

        if should_skip_chunking:
            
            logits = self._get_logits(pruned_states, lm_head, logits_metadata)
            sampled_logits = (
                logits[sample_indices] if sample_indices is not None else logits
            )
            input_logits = logits[input_logprob_indices]
            del logits

            logprobs_result = self.process_input_logprobs(input_logits, logits_metadata)
        else:
            logprobs_result, sampled_logits = self.process_input_logprobs_by_chunk(
                pruned_states,
                sample_indices,
                input_logprob_indices,
                token_to_seq_idx,
                lm_head,
                logits_metadata,
            )

        return LogitsProcessorOutput(
            next_token_logits=sampled_logits,
            hidden_states=hidden_states_to_store,
            input_token_logprobs=logprobs_result.input_token_logprobs,
            input_top_logprobs_val=logprobs_result.input_top_logprobs_val,
            input_top_logprobs_idx=logprobs_result.input_top_logprobs_idx,
            input_token_ids_logprobs_val=logprobs_result.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=logprobs_result.input_token_ids_logprobs_idx,
            mm_input_embeds=logits_metadata.mm_input_embeds,
        )

    def _get_pruned_states(
        self,
        hidden_states: torch.Tensor,
        hidden_states_before_norm: Optional[torch.Tensor],
        aux_hidden_states: Optional[torch.Tensor],
        logits_metadata: LogitsMetadata,
    ):
        pruned_states_before_norm: Optional[torch.Tensor] = None
        aux_pruned_states = None
        token_to_seq_idx = []

        if (
            logits_metadata.forward_mode.is_decode_or_idle()
            or logits_metadata.forward_mode.is_target_verify()
            or logits_metadata.forward_mode.is_draft_extend_v2()
        ):
            pruned_states = hidden_states
            pruned_states_before_norm = hidden_states_before_norm
            if aux_hidden_states is not None:
                aux_pruned_states = [hidden for hidden in aux_hidden_states]
            sample_indices = None
            input_logprob_indices = None

        elif (
            logits_metadata.forward_mode.is_extend()
            and not logits_metadata.extend_return_logprob
        ):
            
            if logits_metadata.padded_static_len < 0:
                last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
            else:
                
                
                
                idx = torch.arange(
                    len(logits_metadata.extend_seq_lens),
                    device=logits_metadata.extend_seq_lens.device,
                )
                last_index = (
                    idx * logits_metadata.padded_static_len
                    + logits_metadata.extend_seq_lens
                    - 1
                )
            pruned_states = hidden_states[last_index]
            if hidden_states_before_norm is not None:
                pruned_states_before_norm = hidden_states_before_norm[last_index]
            if aux_hidden_states is not None:
                aux_pruned_states = [hidden[last_index] for hidden in aux_hidden_states]
            sample_indices = None
            input_logprob_indices = None
        else:
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

            sample_index_pt = -1
            sample_indices = []
            input_logprob_indices_pt = 0
            input_logprob_indices = []
            pt, pruned_states_list, pruned_states_before_norm_list = 0, [], []

            for idx, (extend_logprob_start_len, extend_len) in enumerate(
                zip(
                    logits_metadata.extend_logprob_start_lens_cpu,
                    logits_metadata.extend_seq_lens_cpu,
                )
            ):
                
                
                if extend_len == extend_logprob_start_len:
                    start_len = extend_logprob_start_len - 1
                else:
                    start_len = extend_logprob_start_len

                
                
                assert extend_len > start_len
                pruned_states_list.append(
                    hidden_states[pt + start_len : pt + extend_len]
                )
                if hidden_states_before_norm is not None:
                    pruned_states_before_norm_list.append(
                        hidden_states_before_norm[pt + start_len : pt + extend_len]
                    )
                
                
                token_to_seq_idx.extend([idx] * (extend_len - start_len))
                pt += extend_len
                sample_index_pt += extend_len - start_len
                sample_indices.append(sample_index_pt)
                input_logprob_indices.extend(
                    [
                        input_logprob_indices_pt + i
                        for i in range(extend_len - extend_logprob_start_len)
                    ]
                )
                input_logprob_indices_pt += extend_len - start_len

            
            token_to_seq_idx.append(len(logits_metadata.extend_seq_lens_cpu) - 1)
            pruned_states = torch.cat(pruned_states_list)
            if hidden_states_before_norm is not None:
                pruned_states_before_norm = torch.cat(pruned_states_before_norm_list)
            sample_indices = torch.tensor(
                sample_indices, device=pruned_states.device, dtype=torch.int64
            )
            input_logprob_indices = torch.tensor(
                input_logprob_indices, device=pruned_states.device, dtype=torch.int64
            )

        return (
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            input_logprob_indices,
            token_to_seq_idx,
        )

    def _get_hidden_states_to_store(
        self,
        hidden_states: torch.Tensor,
        hidden_states_before_norm: Optional[torch.Tensor],
        aux_hidden_states: Optional[List[torch.Tensor]],
        pruned_states: torch.Tensor,
        pruned_states_before_norm: Optional[torch.Tensor],
        aux_pruned_states: Optional[List[torch.Tensor]],
        sample_indices: Optional[torch.Tensor],
        logits_metadata: LogitsMetadata,
    ) -> Optional[torch.Tensor]:
        hidden_states_to_store: Optional[torch.Tensor] = None
        hidden_states_to_store_before_norm: Optional[torch.Tensor] = None
        if logits_metadata.capture_hidden_mode.need_capture():
            if logits_metadata.capture_hidden_mode.is_full():
                if aux_hidden_states is not None:
                    aux_hidden_states = torch.cat(aux_hidden_states, dim=-1)
                    hidden_states_to_store = aux_hidden_states
                else:
                    hidden_states_to_store = hidden_states
                hidden_states_to_store_before_norm = hidden_states_before_norm
            elif logits_metadata.capture_hidden_mode.is_last():
                
                
                if aux_hidden_states is not None:
                    aux_pruned_states = torch.cat(aux_pruned_states, dim=-1)
                    hidden_states_to_store = (
                        aux_pruned_states[sample_indices]
                        if sample_indices is not None
                        else aux_pruned_states
                    )
                else:
                    hidden_states_to_store = (
                        pruned_states[sample_indices]
                        if sample_indices is not None
                        else pruned_states
                    )
                    if hidden_states_before_norm is not None:
                        hidden_states_to_store_before_norm = (
                            pruned_states_before_norm[sample_indices]
                            if sample_indices is not None
                            else pruned_states_before_norm
                        )
            else:
                assert False, "Should never reach"

        if hidden_states_to_store_before_norm is not None:
            
            
            hidden_states_to_store = hidden_states_to_store_before_norm

        return hidden_states_to_store

    def _expand_metadata_for_logprobs(
        self, logits_metadata: LogitsMetadata, device: torch.device
    ):
        pruned_lens = torch.tensor(
            logits_metadata.extend_logprob_pruned_lens_cpu,
            device=device,
        )
        if logits_metadata.temp_scaled_logprobs:
            logits_metadata.temperature = torch.repeat_interleave(
                logits_metadata.temperature.view(-1),
                pruned_lens,
            ).view(-1, 1)
        if logits_metadata.top_p_normalized_logprobs:
            logits_metadata.top_p = torch.repeat_interleave(
                logits_metadata.top_p,
                pruned_lens,
            )

    def process_input_logprobs(self, input_logits, logits_metadata: LogitsMetadata):
        input_logprobs = compute_temp_top_p_normalized_logprobs(
            input_logits, logits_metadata
        )

        
        if logits_metadata.extend_return_top_logprob:
            (
                input_top_logprobs_val,
                input_top_logprobs_idx,
            ) = get_top_logprobs_prefill(input_logprobs, logits_metadata)
        else:
            input_top_logprobs_val = input_top_logprobs_idx = None

        
        if logits_metadata.extend_token_ids_logprob:
            (
                input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs_prefill(input_logprobs, logits_metadata)
        else:
            input_token_ids_logprobs_val = input_token_ids_logprobs_idx = None

        input_token_logprobs = input_logprobs[
            torch.arange(input_logprobs.shape[0], device=input_logprobs.device),
            logits_metadata.extend_input_logprob_token_ids_gpu,
        ]

        return InputLogprobsResult(
            input_token_logprobs=input_token_logprobs,
            input_top_logprobs_val=input_top_logprobs_val,
            input_top_logprobs_idx=input_top_logprobs_idx,
            input_token_ids_logprobs_val=input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
        )

    def process_input_logprobs_by_chunk(
        self,
        pruned_states: torch.Tensor,
        sample_indices: torch.Tensor,
        input_logprob_indices: torch.Tensor,
        token_to_seq_idx: list[int],
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
    ) -> Tuple[InputLogprobsResult, torch.Tensor]:
        """
        compute logprobs for the output token from the hidden states.
        To avoid using too much memory, we split pruned_states into chunks of
        rows to compute input_logprobs separately, then concatenate the results.

        Returns:
            InputLogprobsResult: logprobs result
            torch.Tensor: sampled logits
        """

        
        chunk_size = self.logprobs_chunk_size
        total_size = pruned_states.shape[0]
        num_chunks = (total_size + chunk_size - 1) // chunk_size

        input_token_logprobs = []
        if logits_metadata.extend_return_top_logprob:
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
        else:
            input_top_logprobs_val = None
            input_top_logprobs_idx = None
        if logits_metadata.extend_token_ids_logprob:
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
        else:
            input_token_ids_logprobs_val = None
            input_token_ids_logprobs_idx = None

        
        
        split_len_topk = 0
        split_len_token_ids = 0

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_size)

            
            chunk_mask = (input_logprob_indices >= start_idx) & (
                input_logprob_indices < end_idx
            )
            global_indices = input_logprob_indices[chunk_mask]
            chunk_indices = global_indices - start_idx
            
            
            mask_indices = torch.nonzero(chunk_mask, as_tuple=True)[0]

            
            chunk_states = pruned_states[start_idx:end_idx]
            chunk_logits = self._get_logits(chunk_states, lm_head, logits_metadata)

            
            if i == 0:
                sampled_logits = torch.empty(
                    (sample_indices.shape[0], chunk_logits.shape[1]),
                    dtype=chunk_logits.dtype,
                    device=chunk_logits.device,
                )

            
            
            chunk_sample_mask = (sample_indices >= start_idx) & (
                sample_indices < end_idx
            )
            if chunk_sample_mask.any():
                chunk_sample_indices = sample_indices[chunk_sample_mask] - start_idx
                sampled_logits[chunk_sample_mask] = chunk_logits[chunk_sample_indices]

            
            if chunk_indices.numel() == 0:
                continue

            
            chunk_input_logprobs = chunk_logits[chunk_indices]
            
            
            
            chunk_temperature = (
                logits_metadata.temperature[global_indices]
                if logits_metadata.temp_scaled_logprobs
                and logits_metadata.temperature is not None
                else None
            )
            chunk_top_p = (
                logits_metadata.top_p[global_indices]
                if logits_metadata.top_p_normalized_logprobs
                and logits_metadata.top_p is not None
                else None
            )
            chunk_input_logprobs = compute_temp_top_p_normalized_logprobs(
                chunk_input_logprobs,
                logits_metadata,
                chunk_top_p,
                chunk_temperature,
            )

            
            chunk_slice = slice(
                token_to_seq_idx[start_idx], token_to_seq_idx[end_idx] + 1
            )

            
            if logits_metadata.extend_return_top_logprob:
                top_k_nums = logits_metadata.top_logprobs_nums[chunk_slice]
                pruned_lens = logits_metadata.extend_logprob_pruned_lens_cpu[
                    chunk_slice
                ]
                split_len_topk = get_top_logprobs_chunk(
                    chunk_input_logprobs,
                    logits_metadata,
                    top_k_nums,
                    pruned_lens,
                    input_top_logprobs_val,
                    input_top_logprobs_idx,
                    split_len_topk,
                )

            
            if logits_metadata.extend_token_ids_logprob:
                token_ids_logprobs = logits_metadata.token_ids_logprobs[chunk_slice]
                pruned_lens = logits_metadata.extend_logprob_pruned_lens_cpu[
                    chunk_slice
                ]
                split_len_token_ids = get_token_ids_logprobs_chunk(
                    chunk_input_logprobs,
                    token_ids_logprobs,
                    pruned_lens,
                    input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx,
                    split_len_token_ids,
                )

            
            chunk_input_token_logprobs = chunk_input_logprobs[
                torch.arange(
                    chunk_input_logprobs.shape[0], device=chunk_input_logprobs.device
                ),
                logits_metadata.extend_input_logprob_token_ids_gpu[mask_indices],
            ]
            input_token_logprobs.append(chunk_input_token_logprobs)

        
        input_token_logprobs = torch.cat(input_token_logprobs, dim=0)

        return (
            InputLogprobsResult(
                input_token_logprobs=input_token_logprobs,
                input_top_logprobs_val=input_top_logprobs_val,
                input_top_logprobs_idx=input_top_logprobs_idx,
                input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
            ),
            sampled_logits,
        )

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get logits from hidden_states.

        If sampled_logits_only is True, it means hidden_states only contain the
        last position (e.g., extend without input logprobs). The caller should
        guarantee the given hidden_states follow this constraint.
        """
        hidden_states, local_hidden_states = self._gather_dp_attn_hidden_states(
            hidden_states, logits_metadata
        )

        logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)

        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)

        if self.do_tensor_parallel_all_gather:
            if self.use_attn_tp_group:
                logits = self._gather_attn_tp_logits(logits)
            else:
                logits = tensor_model_parallel_all_gather(logits)

        logits = self._scatter_dp_attn_logits(
            logits, local_hidden_states, logits_metadata
        )

        logits = self._copy_logits_to_buffer(logits, logits_metadata)

        if self.final_logit_softcapping:
            if not _is_npu:
                fused_softcap(logits, self.final_logit_softcapping)
            else:
                logits = self.final_logit_softcapping * torch.tanh(
                    logits / self.final_logit_softcapping
                )

        return logits

    def _compute_lm_head(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"):
            
            logits = lm_head(hidden_states)
        elif hasattr(lm_head, "weight"):
            
            if self.use_fp32_lm_head:
                logits = torch.matmul(
                    hidden_states.to(torch.float32), lm_head.weight.to(torch.float32).T
                )
            elif use_intel_amx_backend(lm_head):
                logits = torch.ops.sgl_kernel.weight_packed_linear(
                    hidden_states.to(lm_head.weight.dtype),
                    lm_head.weight,
                    None,  
                    True,  
                )
            elif get_global_server_args().rl_on_policy_target is not None:
                
                logits = torch.matmul(
                    hidden_states.bfloat16(), lm_head.weight.T.bfloat16()
                )
            else:
                logits = torch.matmul(
                    hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
                )
        else:
            
            
            if self.use_fp32_lm_head:
                with torch.cuda.amp.autocast(enabled=False):
                    logits = lm_head.quant_method.apply(
                        lm_head, hidden_states.to(torch.float32), embedding_bias
                    )
            else:
                logits = lm_head.quant_method.apply(
                    lm_head, hidden_states, embedding_bias
                )
        return logits

    def _gather_dp_attn_hidden_states(
        self, hidden_states: torch.Tensor, logits_metadata: LogitsMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.do_tensor_parallel_all_gather_dp_attn:
            logits_metadata.compute_dp_attention_metadata()
            local_hidden_states = hidden_states
            hidden_states = logits_metadata.gathered_buffer
            dp_gather_replicate(hidden_states, local_hidden_states, logits_metadata)
            return hidden_states, local_hidden_states
        return hidden_states, hidden_states

    def _gather_attn_tp_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.vocab_size % self.attn_tp_size == 0:
            global_logits = torch.empty(
                (
                    self.attn_tp_size,
                    logits.shape[0],
                    self.vocab_size // self.attn_tp_size,
                ),
                device=logits.device,
                dtype=logits.dtype,
            )
            attn_tp_all_gather_into_tensor(global_logits, logits)
            global_logits = global_logits.permute(1, 0, 2).reshape(
                logits.shape[0], self.vocab_size
            )
        else:
            global_logits = torch.empty(
                (self.vocab_size, logits.shape[0]),
                device=logits.device,
                dtype=logits.dtype,
            )
            global_logits = global_logits.T
            attn_tp_all_gather(
                list(global_logits.tensor_split(self.attn_tp_size, dim=-1)),
                logits,
            )
        return global_logits

    def _scatter_dp_attn_logits(
        self,
        logits: torch.Tensor,
        local_hidden_states: torch.Tensor,
        logits_metadata: LogitsMetadata,
    ) -> torch.Tensor:
        if self.do_tensor_parallel_all_gather_dp_attn:
            global_logits = logits
            logits = torch.empty(
                (local_hidden_states.shape[0], global_logits.shape[1]),
                device=global_logits.device,
                dtype=global_logits.dtype,
            )
            dp_scatter(logits, global_logits, logits_metadata)
        return logits

    def _copy_logits_to_buffer(
        self, logits: torch.Tensor, logits_metadata: LogitsMetadata
    ) -> torch.Tensor:
        if logits_metadata.next_token_logits_buffer is not None:
            logits_buffer = logits_metadata.next_token_logits_buffer
            assert logits_buffer.dtype == torch.float
            logits_buffer.copy_(logits[:, : self.vocab_size])
            logits = logits_buffer
        else:
            logits = logits[:, : self.vocab_size].float()
        return logits

    def _get_dllm_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
    ) -> LogitsProcessorOutput:
        assert self.return_full_logits
        full_logits = self._get_logits(hidden_states, lm_head, logits_metadata)
        return LogitsProcessorOutput(
            full_logits=full_logits,
            next_token_logits=None,
        )

    def compute_logprobs_for_multi_item_scoring(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
        delimiter_token: int,
    ):
        """
        Compute logprobs for multi-item scoring using delimiter-based token extraction.

        This method is designed for scenarios where you want to score multiple items/candidates
        against a single query by combining them into one sequence separated by delimiters.

        Sequence format: Query<delimiter>Item1<delimiter>Item2<delimiter>...
        Scoring positions: Extracts logprobs at positions before each <delimiter>

        Args:
            input_ids (torch.Tensor): Input token IDs containing query and items separated by delimiters.
                Shape: [total_sequence_length] for single request or [batch_total_length] for batch.
            hidden_states (torch.Tensor): Hidden states from the model.
                Shape: [sequence_length, hidden_dim].
            lm_head (VocabParallelEmbedding): Language model head for computing logits.
            logits_metadata (Union[LogitsMetadata, ForwardBatch]): Metadata containing batch info
                and token ID specifications for logprob extraction.
            delimiter_token (int): Token ID used as delimiter between query and items.

        Returns:
            LogitsProcessorOutput: Contains:
                - next_token_logits: None (not needed for scoring-only requests)
                - input_token_logprobs: Logprobs of delimiter tokens at scoring positions
                - input_top_logprobs_val: Top-k logprobs at delimiter positions (if requested)
                - input_top_logprobs_idx: Top-k token indices at delimiter positions (if requested)
                - input_token_ids_logprobs_val: Logprobs for user-requested token IDs (if any)
                - input_token_ids_logprobs_idx: Indices for user-requested token IDs (if any)
        """
        multi_item_indices = (input_ids == delimiter_token).nonzero(as_tuple=True)[
            0
        ] - 1
        
        sliced_hidden = hidden_states[multi_item_indices]

        sliced_logits = self._get_logits(sliced_hidden, lm_head, logits_metadata)
        sliced_logprobs = torch.nn.functional.log_softmax(sliced_logits, dim=-1)

        
        input_token_ids_logprobs_val = []
        input_token_ids_logprobs_idx = []
        input_top_logprobs_val = None
        input_top_logprobs_idx = None

        
        
        if (
            logits_metadata.token_ids_logprobs
            or logits_metadata.extend_return_top_logprob
        ):
            logits_metadata.extend_logprob_pruned_lens_cpu = []

            if logits_metadata.extend_seq_lens_cpu is not None:
                
                input_pt = 0
                for req_seq_len in logits_metadata.extend_seq_lens_cpu:
                    req_input_ids = input_ids[input_pt : input_pt + req_seq_len]
                    delimiter_count = (req_input_ids == delimiter_token).sum().item()
                    logits_metadata.extend_logprob_pruned_lens_cpu.append(
                        delimiter_count
                    )
                    input_pt += req_seq_len
            else:
                
                total_delimiters = (input_ids == delimiter_token).sum().item()
                logits_metadata.extend_logprob_pruned_lens_cpu = [total_delimiters]

        
        if logits_metadata.extend_token_ids_logprob:
            (
                input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs_prefill(
                sliced_logprobs, logits_metadata, delay_cpu_copy=True
            )

        
        if logits_metadata.extend_return_top_logprob:
            (
                input_top_logprobs_val,
                input_top_logprobs_idx,
            ) = get_top_logprobs_prefill(sliced_logprobs, logits_metadata)

        
        input_token_logprobs = sliced_logprobs[:, delimiter_token]

        return LogitsProcessorOutput(
            next_token_logits=None,  
            input_token_logprobs=input_token_logprobs,
            input_top_logprobs_val=input_top_logprobs_val,
            input_top_logprobs_idx=input_top_logprobs_idx,
            input_token_ids_logprobs_val=input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
            
            
            
            mm_input_embeds=logits_metadata.mm_input_embeds,
        )


@triton.jit
def fused_softcap_kernel(
    full_logits_ptr,
    softcapping_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    
    x = tl.load(full_logits_ptr + offsets, mask=mask)

    
    x = x / softcapping_value

    
    exp2x = tl.exp(2 * x)
    x = (exp2x - 1) / (exp2x + 1)

    x = x * softcapping_value

    
    tl.store(full_logits_ptr + offsets, x, mask=mask)


def fused_softcap(full_logits, final_logit_softcapping):
    n_elements = full_logits.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)

    fused_softcap_kernel[grid](
        full_logits_ptr=full_logits,
        softcapping_value=final_logit_softcapping,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return full_logits
